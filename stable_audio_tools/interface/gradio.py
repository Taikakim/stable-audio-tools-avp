import gc
import numpy as np
import gradio as gr
import json 
import re
import subprocess
import torch
import torchaudio
import threading
import os, time

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..interface.aeiou import audio_spectrogram_image
from ..inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

from .interfaces.diffusion_cond import create_diffusion_cond_ui, generate_with_model

# Global variables for models
model = None
model_type = None
sample_rate = 32000
sample_size = 1920000
model_instances = {}
active_model_idx = 0

def find_config_for_checkpoint(ckpt_path, explicit_config_path=None):
    """Find the most appropriate config file for a checkpoint using various strategies"""
    
    # 1. If explicit config is provided and exists, use it
    if explicit_config_path and os.path.exists(explicit_config_path):
        print(f"Using explicitly provided config: {explicit_config_path}")
        return explicit_config_path
        
    # Get checkpoint directory and basename
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)
    base_name = os.path.splitext(ckpt_name)[0]
    
    # 2. Look for same-named config in the same directory
    same_name_config = os.path.join(ckpt_dir, f"{base_name}.json")
    if os.path.exists(same_name_config):
        print(f"Found matching config file: {same_name_config}")
        return same_name_config
    
    # 3. Look for config files in the same directory
    if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
        config_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.json')]
        if config_files:
            config_path = os.path.join(ckpt_dir, config_files[0])
            print(f"Using config file from same directory: {config_path}")
            return config_path
    
    # 4. Check parent directory
    parent_dir = os.path.dirname(ckpt_dir)
    if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
        # Check for same-named config in parent directory
        parent_same_name = os.path.join(parent_dir, f"{base_name}.json")
        if os.path.exists(parent_same_name):
            print(f"Found matching config in parent directory: {parent_same_name}")
            return parent_same_name
            
        # Check for any config in parent directory
        parent_configs = [f for f in os.listdir(parent_dir) if f.endswith('.json')]
        if parent_configs:
            config_path = os.path.join(parent_dir, parent_configs[0])
            print(f"Using config from parent directory: {config_path}")
            return config_path
    
    # 5. Fall back to default location if explicit config was provided
    if explicit_config_path:
        print(f"Warning: Config file {explicit_config_path} not found, but continuing with this path")
        return explicit_config_path
        
    # No suitable config found
    raise FileNotFoundError(f"Could not find a config file for checkpoint: {ckpt_path}")

def update_gpu_memory():
    """Returns current GPU memory usage with more accurate reporting"""
    if torch.cuda.is_available():
        # Force CUDA synchronization for accuracy
        torch.cuda.synchronize()
        
        try:
            # Try to use nvidia-smi for most accurate reporting
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
            values = result.decode('utf-8').strip().split(',')
            mem_used = int(values[0].strip()) / 1024  # Convert to GB
            mem_total = int(values[1].strip()) / 1024  # Convert to GB
        except:
            # Fall back to torch reporting if nvidia-smi isn't available
            mem_used = torch.cuda.memory_allocated() / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return f"GPU Memory: {mem_used:.2f}GB / {mem_total:.2f}GB ({(mem_used/mem_total)*100:.1f}%)"
    else:
        return "CUDA not available"

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size, model_type
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        
        # Load the JSON config file if model_config is a string (path)
        if isinstance(model_config, str):
            try:
                with open(model_config, 'r') as f:
                    model_config_dict = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load config file {model_config}: {e}")
        else:
            model_config_dict = model_config
                
        model = create_model_from_config(model_config_dict)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        
        # Use the loaded config dict going forward
        model_config = model_config_dict

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]

    # Only try to load pretransform if a valid path is provided
    if pretransform_ckpt_path is not None and pretransform_ckpt_path.strip() != "":
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")
    elif hasattr(model, 'pretransform') and model.pretransform is not None:
        # If there's a pretransform but no checkpoint specified
        print("Model has a pretransform, but no checkpoint specified. Using default initialization.")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config  

def generate_uncond(
        steps=250,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        batch_size=1,
        preview_every=None
        ):

    global preview_images

    preview_images = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        # Turn into torch tensor, converting from int16 to float32
        # init_audio = torch.from_numpy(init_audio).float().div(32767)
        if init_audio.dtype == np.float32:
            init_audio = torch.from_numpy(init_audio)
        elif init_audio.dtype == np.int16:
            init_audio = torch.from_numpy(init_audio).float().div(32767)
        elif init_audio.dtype == np.int32:
            init_audio = torch.from_numpy(init_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:

            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:

            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)

            denoised = rearrange(denoised, "b d n -> d (b n)")

            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)

            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    audio = generate_diffusion_uncond(
        model, 
        steps=steps,
        batch_size=batch_size,
        sample_size=input_sample_size,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        callback = progress_callback if preview_every is not None else None
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram, *preview_images])

def generate_lm(
        temperature=1.0,
        top_p=0.95,
        top_k=0,    
        batch_size=1,
        ):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    audio = model.generate_audio(
        batch_size=batch_size,
        max_gen_len = sample_size//model.pretransform.downsampling_ratio,
        conditioning=None,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True
    )

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    return ("output.wav", [audio_spectrogram])


def create_uncond_sampling_ui(model_config):   
    generate_button = gr.Button("Generate", variant='primary', scale=1)
    
    with gr.Row(equal_height=False):
        with gr.Column():            
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")

            with gr.Accordion("Sampler params", open=False):
            
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

            # Sampler params
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")

            with gr.Accordion("Init audio", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use init audio")
                init_audio_input = gr.Audio(label="Init audio")
                init_noise_level_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])
    
    generate_button.click(fn=generate_uncond, 
        inputs=[
            steps_slider, 
            seed_textbox, 
            sampler_type_dropdown, 
            sigma_min_slider, 
            sigma_max_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
        ], 
        outputs=[
            audio_output, 
            audio_spectrogram_output
        ], 
        api_name="generate")

def create_diffusion_uncond_ui(model_config):
    with gr.Blocks() as ui:
        create_uncond_sampling_ui(model_config)
    
    return ui

def autoencoder_process(audio, latent_noise, n_quantizers):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    if audio.dtype == np.float32:
        audio = torch.from_numpy(audio)
    elif audio.dtype == np.int16:
        audio = torch.from_numpy(audio).float().div(32767)
    elif audio.dtype == np.int32:
        audio = torch.from_numpy(audio).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio.dtype}")

    audio = audio.to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM, 
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model. 
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def on_parallel_generate_click(
    prompt, negative_prompt, selected_models, 
    seconds_start, seconds_total,
    steps, cfg_scale, seed, 
    sampler_type, sigma_min, sigma_max, rho,
    cfg_interval_min, cfg_interval_max, cfg_rescale,
    file_format, file_naming, cut_to_seconds_total,
    preview_every, init_audio, init_noise_level
):
    """Handle generation with multiple models in parallel"""
    # Convert selected model names to indices
    model_indices = [int(model_name.split(" ")[1]) - 1 for model_name in selected_models]
    
    if not model_indices:
        return [None], *[gr.update(visible=False) for _ in range(6)], *[None for _ in range(6)]
    
    # Run generation in parallel
    kwargs = {
        "negative_prompt": negative_prompt,
        "seconds_start": seconds_start,
        "seconds_total": seconds_total,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sampler_type": sampler_type,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
        "cfg_interval_min": cfg_interval_min,
        "cfg_interval_max": cfg_interval_max,
        "cfg_rescale": cfg_rescale,
        "file_format": file_format,
        "file_naming": file_naming,
        "cut_to_seconds_total": cut_to_seconds_total,
        "preview_every": preview_every,
        "init_audio": init_audio,
        "init_noise_level": init_noise_level
    }
    
    results = generate_parallel(
        prompt=prompt,
        model_indices=model_indices,
        **kwargs
    )
    
    # Collect gallery items and audio files
    gallery_items = []
    audio_accordion_updates = []
    audio_player_updates = []
    
    # Initialize updates (set all invisible by default)
    for i in range(6):
        audio_accordion_updates.append(gr.update(visible=False, label=f"Model {i+1} Output"))
        audio_player_updates.append(None)
    
    # Update with results
    for idx, result in results:
        if result:
            audio_file, spectrograms = result
            model_name = f"Model {idx+1}"
            
            # Add the spectrogram with model name
            if spectrograms and len(spectrograms) > 0:
                gallery_items.append((spectrograms[0][0], f"{model_name}: {prompt}"))
            
            # Update the audio player for this model
            audio_accordion_updates[idx] = gr.update(
                visible=True, 
                label=f"{model_name}: {prompt}"
            )
            audio_player_updates[idx] = audio_file
    
    return [gallery_items] + audio_accordion_updates + audio_player_updates

def create_autoencoder_ui(model_config):

    is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

    if is_dac_rvq:
        n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
    else:
        n_quantizers = 0

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        n_quantizers_slider = gr.Slider(minimum=1, maximum=n_quantizers, step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
        latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=autoencoder_process, inputs=[input_audio, latent_noise_slider, n_quantizers_slider], outputs=output_audio, api_name="process")

    return ui

def diffusion_prior_process(audio, steps, sampler_type, sigma_min, sigma_max):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.unsqueeze(0)

    audio = model.stereoize(audio, in_sr, steps, sampler_kwargs={"sampler_type": sampler_type, "sigma_min": sigma_min, "sigma_max": sigma_max})

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_diffusion_prior_ui(model_config):
    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        # Sampler params
        with gr.Row():
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=diffusion_prior_process, inputs=[input_audio, steps_slider, sampler_type_dropdown, sigma_min_slider, sigma_max_slider], outputs=output_audio, api_name="process")    

    return ui

def create_lm_ui(model_config):
    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_lm, 
            inputs=[
                temperature_slider, 
                top_p_slider, 
                top_k_slider
            ], 
            outputs=[output_audio, audio_spectrogram_output],
            api_name="generate"
        )

    return ui

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=device)
    
    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
        ui = create_diffusion_cond_ui(model_config, model, in_model_half=model_half)
    elif model_type == "diffusion_uncond":
        ui = create_diffusion_uncond_ui(model_config)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(model_config)
    elif model_type == "lm":
        ui = create_lm_ui(model_config)
        
    return ui

def load_model_slot(path, config_path=None, slot_index=0, model_half=True, pretransform_path=None):
    """Loads a model into a specific slot"""
    global model_instances, active_model_idx, model
    
    try:
        # Clear existing model in this slot if any
        if slot_index in model_instances:
            del model_instances[slot_index]
            torch.cuda.empty_cache()
            gc.collect()
        
        # Handle empty path
        if not path or path.strip() == "":
            return f"Slot {slot_index+1} cleared", update_gpu_memory()
        
        # Clean up pretransform path if it's empty
        if not pretransform_path or pretransform_path.strip() == "":
            pretransform_path = None
        
        # Load the model
        if os.path.exists(path) or path.endswith((".safetensors", ".ckpt", ".pth")):
            # It's a checkpoint path
            try:
                config_to_use = find_config_for_checkpoint(path, config_path)
            except FileNotFoundError as e:
                return f"Error: {str(e)}", update_gpu_memory()
                
            new_model, model_config = load_model(
                model_config=config_to_use,
                model_ckpt_path=path,
                pretrained_name=None,
                pretransform_ckpt_path=pretransform_path,
                model_half=model_half
            )
        else:
            # It's a pretrained name
            new_model, model_config = load_model(
                model_config=None,
                model_ckpt_path=None,
                pretrained_name=path,
                pretransform_ckpt_path=pretransform_path,
                model_half=model_half
            )
        
        # Store the model
        model_instances[slot_index] = {
            "model": new_model,
            "config": model_config,
            "path": path,
            "type": model_config["model_type"]
        }
        
        # Make this the active model if it's the first one loaded
        if active_model_idx not in model_instances:
            active_model_idx = slot_index
            model = new_model
            return f"★ ACTIVE: {model_config['model_type']} ★", update_gpu_memory()
        elif active_model_idx == slot_index:
            # If we're reloading the active model, keep it active
            model = new_model
            return f"★ ACTIVE: {model_config['model_type']} ★", update_gpu_memory()
        else:
            return f"Loaded: {model_config['model_type']}", update_gpu_memory()
        
    except Exception as e:
        return f"Error loading model in slot {slot_index+1}: {str(e)}", update_gpu_memory()

def set_active_model(slot_index):
    """Sets a loaded model as the active one and updates all status indicators"""
    global model_instances, active_model_idx, model, model_type, sample_rate, sample_size
    
    # Create status updates for all model slots
    status_updates = []
    for i in range(6):
        if i in model_instances:
            if i == slot_index:
                status_updates.append(f"★ ACTIVE: {model_instances[i]['type']} ★")
            else:
                status_updates.append(f"Loaded: {model_instances[i]['type']}")
        else:
            status_updates.append("Not loaded")
    
    # Set the active model if it exists
    if slot_index in model_instances:
        active_model_idx = slot_index
        model = model_instances[slot_index]["model"]
        model_config = model_instances[slot_index]["config"]
        model_type = model_config["model_type"] 
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        print(f"Active model changed to slot {slot_index+1}, type: {model_type}")
        message = f"Model {slot_index+1} is now active ({model_type})"
    else:
        message = f"No model loaded in slot {slot_index+1}"
    
    return [message, update_gpu_memory()] + status_updates

def create_ui_with_model_manager(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=True):
    """Creates the UI with compact model management capabilities"""
    global model_instances, active_model_idx, model
    
    # Clean up pretransform path if it's empty
    if not pretransform_ckpt_path or pretransform_ckpt_path.strip() == "":
        pretransform_ckpt_path = None
    
    # Load initial model if provided
    initial_config = None
    if pretrained_name is not None or (model_config_path is not None and ckpt_path is not None):
        try:
            if model_config_path is not None and ckpt_path is not None:
                config_to_use = find_config_for_checkpoint(ckpt_path, model_config_path)
                initial_model, initial_config = load_model(
                    model_config=config_to_use, 
                    model_ckpt_path=ckpt_path, 
                    pretrained_name=None, 
                    pretransform_ckpt_path=pretransform_ckpt_path, 
                    model_half=model_half
                )
            else:
                initial_model, initial_config = load_model(
                    model_config=None, 
                    model_ckpt_path=None, 
                    pretrained_name=pretrained_name, 
                    pretransform_ckpt_path=pretransform_ckpt_path, 
                    model_half=model_half
                )
            
            model_instances[0] = {
                "model": initial_model,
                "config": initial_config,
                "path": pretrained_name or ckpt_path,
                "type": initial_config["model_type"]
            }
            
            active_model_idx = 0
            model = initial_model
        except Exception as e:
            print(f"Error loading initial model: {str(e)}")
            print("Starting with no model loaded.")
    
    with gr.Blocks() as ui:
        # Add a hidden textbox to hold extra status messages
        status_message = gr.Textbox(visible=False)
        
        # Compact GPU memory display
        with gr.Row():
            gpu_memory = gr.Textbox(label="GPU Memory", value=update_gpu_memory())
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(fn=update_gpu_memory, inputs=[], outputs=[gpu_memory])
        
        # More compact model management section
        with gr.Accordion("Model Management", open=True):
            # Create table-like header with fixed column widths
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Model**")
                with gr.Column(scale=5):
                    gr.Markdown("**Path/Name**")
                with gr.Column(scale=3):
                    gr.Markdown("**Config**")
                with gr.Column(scale=3):
                    gr.Markdown("**Controls**")
                with gr.Column(scale=2):
                    gr.Markdown("**Status**")
            
            # For tracking status boxes
            status_boxes = []
            
            # Create compact model slots
            for i in range(6):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"**#{i+1}**")
                    
                    with gr.Column(scale=5):
                        model_path = gr.Textbox(
                            show_label=False,
                            placeholder="Model path or name",
                            value=pretrained_name or ckpt_path if i == 0 and (pretrained_name or ckpt_path) else ""
                        )
                    
                    with gr.Column(scale=3):
                        with gr.Row():
                            config_path = gr.Textbox(
                                show_label=False,
                                placeholder="Config path (optional)",
                                value=model_config_path if i == 0 and model_config_path else ""
                            )
                        
                        with gr.Row():
                            pretransform_path = gr.Textbox(
                                show_label=False,
                                placeholder="Pretransform (optional)",
                                value=pretransform_ckpt_path if i == 0 and pretransform_ckpt_path else ""
                            )
                    
                    with gr.Column(scale=3):
                        with gr.Row():
                            load_btn = gr.Button("Load", variant="primary")
                            active_btn = gr.Button("Set Active", variant="secondary")
                    
                    with gr.Column(scale=2):
                        # Create status text directly in place
                        initial_status = "★ ACTIVE ★" if i == active_model_idx and i in model_instances else "Not loaded"
                        if i in model_instances and i != active_model_idx:
                            initial_status = f"Loaded: {model_instances[i]['type']}"
                        status = gr.Textbox(show_label=False, value=initial_status)
                        status_boxes.append(status)
                    
                    # Connect the buttons
                    load_btn.click(
                        fn=load_model_slot,
                        inputs=[model_path, config_path, gr.Number(value=i, visible=False), gr.Checkbox(value=model_half, visible=False), pretransform_path],
                        outputs=[status_boxes[i], gpu_memory]
                    )
                    
                    active_btn.click(
                        fn=lambda idx=i: set_active_model(idx),
                        inputs=[],
                        outputs=[status_message, gpu_memory] + status_boxes
                    )
        
        # Create tabs for different functionalities
        with gr.Tabs() as tabs:
            with gr.Tab("Generation"):
                if initial_config and initial_config["model_type"] in ["diffusion_cond", "diffusion_cond_inpaint"]:
                    create_diffusion_cond_ui(initial_config, model, model_half)
                else:
                    gr.Markdown("Load a conditional diffusion model to use this tab.")
            
            with gr.Tab("Unconditional"):
                if initial_config and initial_config["model_type"] == "diffusion_uncond":
                    create_uncond_sampling_ui(initial_config)
                else:
                    gr.Markdown("Load an unconditional diffusion model to use this tab.")
            
            with gr.Tab("Autoencoder"):
                if initial_config and initial_config["model_type"] in ["autoencoder", "diffusion_autoencoder"]:
                    create_autoencoder_ui(initial_config)
                else:
                    gr.Markdown("Load an autoencoder model to use this tab.")
            
            with gr.Tab("Parallel Generation"):
                create_parallel_generation_ui()

        # Add tab selection event handler that's flexible with arguments
        def on_tab_select(*args):
            # Try to get the tab index from args, default to None if not available
            tab_index = args[0] if args else None
            
            # If we have a valid tab index and it's not the generation tab
            if tab_index is not None and tab_index != 0:
                # Import here to avoid circular imports
                from stable_audio_tools.interface.interfaces.diffusion_cond import clear_bracketing_model_copies
                clear_bracketing_model_copies()
                print(f"Leaving generation tab, clearing bracketing model copies")
            return None
        
        # Connect the tab selection event with change event
        tabs.change(fn=on_tab_select, inputs=None, outputs=None)
    
    return ui
    
def generate_parallel(prompt, model_indices, **kwargs):
    """Run generation with multiple models in parallel"""
    global model_instances
    
    results = []
    threads = []
    thread_results = {}
    
    # Filter out cfg_interval parameters that shouldn't be passed to models
    generation_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['cfg_interval_min', 'cfg_interval_max']}
    
    # Add cfg_interval as a tuple if the individual min/max values were provided
    if 'cfg_interval_min' in kwargs and 'cfg_interval_max' in kwargs:
        generation_kwargs['cfg_interval'] = (kwargs['cfg_interval_min'], kwargs['cfg_interval_max'])
    
    # Create a thread for each model
    for idx in model_indices:
        if idx in model_instances:
            thread_results[idx] = None
            
            def worker(model_idx=idx):  # Use default arg to capture current idx value
                try:
                    # Get this thread's model
                    temp_model = model_instances[model_idx]["model"]
                    temp_config = model_instances[model_idx]["config"]
                    
                    # Generate with proper parameters
                    result = generate_with_model(temp_model, temp_config, prompt, **generation_kwargs)
                    thread_results[model_idx] = result
                except Exception as e:
                    print(f"Error in worker for model {model_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results in order
    for idx in model_indices:
        if idx in thread_results and thread_results[idx] is not None:
            results.append((idx, thread_results[idx]))
    
    return results

def create_parallel_generation_ui():
    """Creates the UI for parallel generation with full options"""
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column(scale=6):
                prompt = gr.Textbox(show_label=False, placeholder="Prompt")
                negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
            
            selected_models = gr.CheckboxGroup(
                label="Select Models to Use",
                choices=["Model 1", "Model 2", "Model 3", "Model 4", "Model 5", "Model 6"],
                value=["Model 1"],
                scale=2
            )
        
        with gr.Row():
            # Timing controls
            seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start")
            seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=30, label="Seconds total")
        
        with gr.Row():
            # Steps slider
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            # CFG scale 
            cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG scale")

        with gr.Accordion("Sampler params", open=False):
            with gr.Row():
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")
                cfg_interval_min_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG interval min")
                cfg_interval_max_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=1.0, label="CFG interval max")

            with gr.Row():
                cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")

            with gr.Row():
                # Sampler params
                sampler_types = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
                sampler_type_dropdown = gr.Dropdown(sampler_types, label="Sampler type", value="dpmpp-3m-sde")
                sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.01, label="Sigma min")
                sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=100, label="Sigma max")
                rho_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.01, value=1.0, label="Sigma curve strength")

        with gr.Accordion("Output params", open=False):
            # Output params
            with gr.Row():
                file_format_dropdown = gr.Dropdown(["wav", "flac", "mp3 320k", "mp3 v0", "mp3 128k", "m4a aac_he_v2 64k", "m4a aac_he_v2 32k"], label="File format", value="wav")
                file_naming_dropdown = gr.Dropdown(["verbose", "prompt", "output.wav"], label="File naming", value="output.wav")
                cut_to_seconds_total_checkbox = gr.Checkbox(label="Cut to seconds total", value=True)
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Spec Preview Every N Steps")

        with gr.Accordion("Init audio", open=False):
            init_audio_input = gr.Audio(label="Init audio")
            init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

        generate_button = gr.Button("Generate with All Selected Models", variant='primary')
        
        # Display area for results
        with gr.Row():
            output_gallery = gr.Gallery(label="Generated Spectrograms")
        
        # Create audio players for all possible models
        audio_players = []
        for i in range(6):
            with gr.Accordion(f"Model {i+1} Output", visible=False, open=True) as audio_accordion:
                audio_player = gr.Audio(label=f"Model {i+1} Audio")
                audio_players.append((audio_accordion, audio_player))
        
        # Connect the generate button
        generate_button.click(
            fn=on_parallel_generate_click,
            inputs=[
                prompt, negative_prompt, selected_models, 
                seconds_start_slider, seconds_total_slider,
                steps_slider, cfg_scale_slider, seed_textbox, 
                sampler_type_dropdown, sigma_min_slider, sigma_max_slider, rho_slider,
                cfg_interval_min_slider, cfg_interval_max_slider, cfg_rescale_slider,
                file_format_dropdown, file_naming_dropdown, cut_to_seconds_total_checkbox,
                preview_every_slider, init_audio_input, init_noise_level_slider
            ],
            outputs=[output_gallery] + [acc for acc, _ in audio_players] + [player for _, player in audio_players]
        )
    
    return ui
