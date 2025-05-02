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

from ..aeiou import audio_spectrogram_image
from ...inference.generation import generate_diffusion_cond

from ...models.factory import create_model_from_config

# Global variables for model management
model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = True
model_instances = {}
active_model_idx = 0
output_directory = os.getcwd()  # Default to current directory

bracketing_model_copies = {}  # format: {count: [model_copies]}
active_bracketing_count = 0   # track currently allocated number of copies
last_active_model_idx = None  # track which model was used for copies

# when using a prompt in a filename
def condense_prompt(prompt):
    pattern = r'[\\/:*?"<>|]'
    # Replace special characters with hyphens
    prompt = re.sub(pattern, '-', prompt)
    # set a character limit 
    prompt = prompt[:150]
    # zero length prompts may lead to filenames (ie ".wav") which seem cause problems with gradio
    if len(prompt)==0:
        prompt = "_"
    return prompt

def generate_with_model(specific_model, model_config, prompt, negative_prompt=None, steps=100, cfg_scale=7.0, seed=-1, **kwargs):
    """Generate audio with a specific model - used for parallel generation"""
    device = next(specific_model.parameters()).device
    temp_sample_rate = model_config["sample_rate"]
    temp_sample_size = model_config["sample_size"]
    temp_model_type = model_config["model_type"]
    
    # Extract seconds parameters for conditioning
    seconds_start = kwargs.pop("seconds_start", 0)
    seconds_total = kwargs.pop("seconds_total", temp_sample_size // temp_sample_rate)
    
    # Basic validation
    if temp_model_type not in ["diffusion_cond", "diffusion_cond_inpaint"]:
        return None, [(None, f"Model type {temp_model_type} not supported for conditional generation")]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create conditioning with seconds parameters
    conditioning_dict = {"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}
    conditioning = [conditioning_dict]
    
    if negative_prompt:
        negative_conditioning_dict = {"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}
        negative_conditioning = [negative_conditioning_dict]
    else:
        negative_conditioning = None
    
    # Setup seed for reproducibility
    seed = int(seed)
    if seed == -1:
        # This should now be handled in generate_parallel for consistent random seeds
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    
    # Ensure deterministic results by setting global seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Using seed: {seed}")
    
    # Extract specific generation parameters
    sampler_type = kwargs.pop("sampler_type", "dpmpp-3m-sde")
    sigma_min = kwargs.pop("sigma_min", 0.03)
    sigma_max = kwargs.pop("sigma_max", 100)
    cfg_interval = kwargs.pop("cfg_interval", (0.0, 1.0))
    cfg_rescale = kwargs.pop("cfg_rescale", 0.0)
    rho = kwargs.pop("rho", 1.0)
    
    # Generate arguments with only the parameters expected by generate_diffusion_cond
    generate_args = {
        "model": specific_model,
        "conditioning": conditioning,
        "negative_conditioning": negative_conditioning,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": cfg_interval,
        "batch_size": 1,
        "sample_size": temp_sample_size,
        "seed": seed,
        "device": device,
        "sampler_type": sampler_type,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "scale_phi": cfg_rescale,
        "rho": rho
    }
    
    # Add other valid parameters
    init_audio = kwargs.pop("init_audio", None)
    if init_audio is not None:
        generate_args["init_audio"] = init_audio
        
    init_noise_level = kwargs.pop("init_noise_level", 1.0)
    generate_args["init_noise_level"] = init_noise_level
    
    # Handle inpainting parameters if needed
    if temp_model_type == "diffusion_cond_inpaint":
        inpaint_audio = kwargs.pop("inpaint_audio", None)
        if inpaint_audio is not None:
            mask_start = kwargs.pop("mask_maskstart", 0)
            mask_end = kwargs.pop("mask_maskend", temp_sample_size // temp_sample_rate)
            
            # Convert to sample indices
            mask_start_samples = int(mask_start * temp_sample_rate)
            mask_end_samples = int(mask_end * temp_sample_rate)
            
            # Create mask
            inpaint_mask = torch.ones(1, temp_sample_size, device=device)
            inpaint_mask[:, mask_start_samples:mask_end_samples] = 0
            
            generate_args["inpaint_audio"] = inpaint_audio
            generate_args["inpaint_mask"] = inpaint_mask
    
    # Generate audio
    if temp_model_type == "diffusion_cond":
        audio = generate_diffusion_cond(**generate_args)
    elif temp_model_type == "diffusion_cond_inpaint":
        audio = generate_diffusion_cond_inpaint(**generate_args)
    
    # Process output
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    # Save file to the output directory
    filename = f"output_model_{np.random.randint(10000)}.wav"
    filepath = os.path.join(output_directory, filename)
    print(f"Saving parallel generation to: {filepath}")
    torchaudio.save(filepath, audio, temp_sample_rate)
    
    # Create spectrogram
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=temp_sample_rate)
    
    return filepath, [(audio_spectrogram, f"Model result for: {prompt}")]

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=48,
        cfg_scale=7.0,
        steps=100,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.01,
        sigma_max=1000,
        rho=1.0,
        cfg_interval_min=0.0,
        cfg_interval_max=1.0,
        cfg_rescale=0.0,
        file_format="wav",
        file_naming="verbose",
        cut_to_seconds_total=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_maskstart=None,
        mask_maskend=None,
        inpaint_audio=None,
        batch_size=1    
    ):

    global model_instances, active_model_idx, model, model_type, sample_rate, sample_size

     # Explicitly get the active model and update all globals
    if active_model_idx in model_instances:
        model = model_instances[active_model_idx]["model"]
        model_config = model_instances[active_model_idx]["config"] 
        model_type = model_config["model_type"]
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        print(f"Generating with model in slot {active_model_idx+1}: {model_type}")
    else:
        return "No active model available", None
    
    # Check if the model type is appropriate for this function
    if model_type not in ["diffusion_cond", "diffusion_cond_inpaint"]:
        return f"Active model is type {model_type}, but this function requires a conditional diffusion model", None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Return fake stereo audio
    conditioning_dict = {"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}

    conditioning = [conditioning_dict] * batch_size

    if negative_prompt:
        negative_conditioning_dict = {"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}

        negative_conditioning = [negative_conditioning_dict] * batch_size
    else:
        negative_conditioning = None
        
    # Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)
    # if seed is -1, define the seed value now, randomly, so we can save it in the filename
    if(seed==-1):
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    
    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio

        if init_audio.dtype == np.float32:
            init_audio = torch.from_numpy(init_audio)
        elif init_audio.dtype == np.int16:
            init_audio = torch.from_numpy(init_audio).float().div(32767)
        elif init_audio.dtype == np.int32:
            init_audio = torch.from_numpy(init_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")

        if model_half:
            init_audio = init_audio.to(torch.float16)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0) # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device).to(init_audio.dtype)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            init_audio = init_audio[:, :sample_size]

        init_audio = (sample_rate, init_audio)

    if inpaint_audio is not None:
        in_sr, inpaint_audio = inpaint_audio
        
        if inpaint_audio.dtype == np.float32:
            inpaint_audio = torch.from_numpy(inpaint_audio)
        elif inpaint_audio.dtype == np.int16:
            inpaint_audio = torch.from_numpy(inpaint_audio).float().div(32767)
        elif inpaint_audio.dtype == np.int32:
            inpaint_audio = torch.from_numpy(inpaint_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {inpaint_audio.dtype}")

        if model_half:
            inpaint_audio = inpaint_audio.to(torch.float16)
        
        if inpaint_audio.dim() == 1:
            inpaint_audio = inpaint_audio.unsqueeze(0) # [1, n]
        elif inpaint_audio.dim() == 2:
            inpaint_audio = inpaint_audio.transpose(0, 1) # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(inpaint_audio.device).to(inpaint_audio.dtype)
            inpaint_audio = resample_tf(inpaint_audio)

        audio_length = inpaint_audio.shape[-1]

        if audio_length > sample_size:
            inpaint_audio = inpaint_audio[:, :sample_size]

        inpaint_audio = (sample_rate, inpaint_audio)

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

    # Handle cfg_interval - could be individual parameters or a tuple
    cfg_interval_to_use = (cfg_interval_min, cfg_interval_max)
    if isinstance(cfg_interval_min, tuple) and len(cfg_interval_min) == 2:
        # First arg is already a tuple
        cfg_interval_to_use = cfg_interval_min
        
    generate_args = {
        "model": model,
        "conditioning": conditioning,
        "negative_conditioning": negative_conditioning,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": cfg_interval_to_use,  # Using the tuple
        "batch_size": batch_size,
        "sample_size": input_sample_size,
        "seed": seed,
        "device": device,
        "sampler_type": sampler_type,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "init_audio": init_audio,
        "init_noise_level": init_noise_level,
        "callback": progress_callback if preview_every is not None else None,
        "scale_phi": cfg_rescale,
        "rho": rho
    }

    # If inpainting, send mask args
    # This will definitely change in the future
    if model_type == "diffusion_cond":
        # Do the audio generation
        audio = generate_diffusion_cond(**generate_args)

    elif model_type == "diffusion_cond_inpaint":
        if inpaint_audio is not None:
            # Convert mask start and end from percentages to sample indices
            mask_start = int(mask_maskstart * sample_rate)
            mask_end = int(mask_maskend * sample_rate)

            inpaint_mask = torch.ones(1, sample_size, device=device)
            inpaint_mask[:, mask_start:mask_end] = 0

            generate_args.update({
                "inpaint_audio": inpaint_audio,
                "inpaint_mask": inpaint_mask
            })

        audio = generate_diffusion_cond_inpaint(**generate_args)

    # Use global output directory
    global output_directory
    
    # Filenaming convention
    prompt_condensed = condense_prompt(prompt) 
    if file_naming=="verbose":
        cfg_filename = "cfg%s" % (cfg_scale)
        seed_filename = seed
        if negative_prompt:
            prompt_condensed += ".neg-%s" % condense_prompt(negative_prompt)
        basename = "%s.%s.%s" % (prompt_condensed, cfg_filename, seed_filename)
    elif file_naming=="prompt":
        basename = prompt_condensed
    else:
        # simple e.g. "output.wav"
        basename = "output" 

    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else: 
        filename_extension = "wav"
    
    # Use the output directory for file paths
    output_filename = os.path.join(output_directory, f"{basename}.{filename_extension}")
    output_wav = os.path.join(output_directory, f"{basename}.wav")

    # Cut the extra silence off the end, if the user requested a smaller seconds_total
    if cut_to_seconds_total:
        audio = audio[:,:,:seconds_total*sample_rate]

    # Encode the audio to WAV format
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # save as wav file
    print(f"Saving audio to: {output_wav}")
    torchaudio.save(output_wav, audio, sample_rate)

    # If file_format is other than wav, convert to other file format
    cmd = ""
    if file_format == "m4a aac_he_v2 32k":
        # note: need to compile ffmpeg with --enable-libfdk_aac
        cmd = f"ffmpeg -i \"{output_wav}\" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 32k -y \"{output_filename}\""
    elif file_format == "m4a aac_he_v2 64k":
        cmd = f"ffmpeg -i \"{output_wav}\" -c:a libfdk_aac -profile:a aac_he_v2 -b:a 64k -y \"{output_filename}\""
    elif file_format == "flac":
        cmd = f"ffmpeg -i \"{output_wav}\" -y \"{output_filename}\""
    elif file_format == "mp3 320k":
        cmd = f"ffmpeg -i \"{output_wav}\" -b:a 320k -y \"{output_filename}\""
    elif file_format == "mp3 128k":
        cmd = f"ffmpeg -i \"{output_wav}\" -b:a 128k -y \"{output_filename}\""
    elif file_format == "mp3 v0":
        cmd = f"ffmpeg -i \"{output_wav}\" -q:a 0 -y \"{output_filename}\""
    else: # wav
        pass
    if cmd:
        cmd += " -loglevel error" # make output less verbose in the cmd window
        subprocess.run(cmd, shell=True, check=True)
    
    # Let's look at a nice spectrogram too
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    # Return the full path to the files
    return (output_filename, [audio_spectrogram, *preview_images])
    
# Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes. 
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file
    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start() 

def create_sampling_ui(model_config):
    global model, model_type, sample_size, sample_rate
    
    has_inpainting = model_type == "diffusion_cond_inpaint"
    
    model_conditioning_config = model_config["model"].get("conditioning", None)

    diffusion_objective = model.diffusion_objective

    is_rf = diffusion_objective == "rectified_flow"

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True
                
# Add bracketing controls in an accordion

    with gr.Accordion("Parameter Bracketing", open=False):
        with gr.Row():
            bracket_count = gr.Slider(minimum=0, maximum=16, step=1, value=0, 
                                    label="Number of Variations (0 for single generation)")
        
        with gr.Column():
            with gr.Row():
                bracket_steps = gr.Number(value=0, label="Steps Increment", precision=0)
                bracket_cfg = gr.Number(value=0.0, label="CFG Scale Increment", precision=2)
            
            with gr.Row():
                bracket_seed = gr.Number(value=0, label="Seed Increment", precision=0)
                bracket_noise = gr.Number(value=0.0, label="Noise Level Increment", precision=2)
            
            with gr.Row():
                bracket_cfg_rescale = gr.Number(value=0.0, label="CFG Rescale Increment", precision=2)
                bracket_sigma_min = gr.Number(value=0.0, label="Sigma Min Increment", precision=3)
            
            with gr.Row():
                bracket_sigma_max = gr.Number(value=0.0, label="Sigma Max Increment", precision=1)
                bracket_rho = gr.Number(value=0.0, label="Rho Increment", precision=2)
            
            bracket_warning = gr.Markdown(
                "⚠️ **Note**: Bracketing creates multiple variations with increasing parameter values. "
                "Results will appear as separate audio players below.", 
                visible=True
            )

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(show_label=False, placeholder="Prompt")
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
        generate_button = gr.Button("Generate", variant='primary', scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row(visible = has_seconds_start or has_seconds_total):
                # Timing controls
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=sample_size//sample_rate, label="Seconds total", visible=has_seconds_total)
            
            with gr.Row():
                # Steps slider
                default_steps = 50 if is_rf else 100
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=default_steps, label="Steps")
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
                    if is_rf:
                        sampler_types = ["euler", "rk4", "dpmpp"]
                        default_sampler_type = "euler"
                    else:
                        sampler_types = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
                        default_sampler_type = "dpmpp-3m-sde"
                    sampler_type_dropdown = gr.Dropdown(sampler_types, label="Sampler type", value=default_sampler_type)
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.01, label="Sigma min", visible=not is_rf)
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=100, label="Sigma max", visible=not is_rf)
                    rho_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.01, value=1.0, label="Sigma curve strength", visible=not is_rf)

            with gr.Accordion("Output params", open=False):
                # Output params
                with gr.Row():
                    file_format_dropdown = gr.Dropdown(["wav", "flac", "mp3 320k", "mp3 v0", "mp3 128k", "m4a aac_he_v2 64k", "m4a aac_he_v2 32k"], label="File format", value="mp3 320k")
                    file_naming_dropdown = gr.Dropdown(["verbose", "prompt", "output.wav"], label="File naming", value="prompt")
                    cut_to_seconds_total_checkbox = gr.Checkbox(label="Cut to seconds total", value=True)
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Spec Preview Every N Steps")
                    
            # Default generation tab
            with gr.Accordion("Init audio", open=False):
                init_audio_input = gr.Audio(label="Init audio")
                min_noise_level = 0.01 if is_rf else 0.1
                max_noise_level = 1.0 if is_rf else 100.0
                init_noise_level_slider = gr.Slider(minimum=min_noise_level, maximum=max_noise_level, step=0.01, value=0.1, label="Init noise level")

            with gr.Accordion("Inpainting", open=False, visible=has_inpainting):
                inpaint_audio_input = gr.Audio(label="Inpaint audio")
                mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=10, label="Mask Start (sec)")
                mask_maskend_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=sample_size//sample_rate, label="Mask End (sec)")

            inputs = [
                prompt, 
                negative_prompt,
                seconds_start_slider, 
                seconds_total_slider, 
                cfg_scale_slider, 
                steps_slider, 
                preview_every_slider, 
                seed_textbox, 
                sampler_type_dropdown, 
                sigma_min_slider, 
                sigma_max_slider,
                rho_slider,
                cfg_interval_min_slider,
                cfg_interval_max_slider,
                cfg_rescale_slider,
                file_format_dropdown,
                file_naming_dropdown,
                cut_to_seconds_total_checkbox,
                init_audio_input,
                init_noise_level_slider,
                mask_maskstart_slider,
                mask_maskend_slider,
                inpaint_audio_input,
                bracket_count, 
                bracket_steps, 
                bracket_cfg,
                bracket_noise, 
                bracket_cfg_rescale,
                bracket_seed,
                bracket_sigma_min,
                bracket_sigma_max,
                bracket_rho
            ]

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

            if has_inpainting:
                send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
                send_to_inpaint_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[inpaint_audio_input])
                    
        # Add container for bracketed results (initially hidden)
        with gr.Column(visible=False) as bracketed_container:
            bracketed_gallery = gr.Gallery(label="Parameter Variations")
            
            # Create placeholders for dynamic audio players
            bracketed_audio_players = []
            for i in range(16):  # Max 16 variations
                audio_player = gr.Audio(label=f"Variation {i+1}", visible=False)
                bracketed_audio_players.append(audio_player)
    
    generate_button.click(
        fn=generate_with_bracketing,
        inputs=inputs,
        outputs=[
            audio_output, audio_spectrogram_output,
            bracketed_container, bracketed_gallery,
            *bracketed_audio_players
        ]
    )

def generate_with_bracketing(
        prompt, negative_prompt, seconds_start, seconds_total, 
        cfg_scale, steps, preview_every, seed, sampler_type,
        sigma_min, sigma_max, rho, cfg_interval_min, cfg_interval_max,
        cfg_rescale, file_format, file_naming, cut_to_seconds_total,
        init_audio, init_noise_level, mask_maskstart, mask_maskend,
        inpaint_audio, bracket_count, bracket_steps, bracket_cfg,
        bracket_noise, bracket_cfg_rescale, bracket_seed, bracket_sigma_min,
        bracket_sigma_max, bracket_rho
    ):
    """Generate audio with parameter bracketing using discrete model copies for true parallelism"""
    global model_instances, active_model_idx, bracketing_model_copies, active_bracketing_count, last_active_model_idx
    
    try:
        # Convert parameters to appropriate types
        try:
            steps = int(steps)
            cfg_scale = float(cfg_scale)
            seed = int(seed) if seed != "-1" else -1
            sigma_min = float(sigma_min)
            sigma_max = float(sigma_max)
            rho = float(rho)
            cfg_rescale = float(cfg_rescale)
            init_noise_level = float(init_noise_level)
            
            # Convert bracketing parameters
            bracket_count = int(bracket_count) if bracket_count not in [None, ""] else 0
            bracket_steps = int(bracket_steps) if bracket_steps not in [None, ""] else 0
            bracket_cfg = float(bracket_cfg) if bracket_cfg not in [None, ""] else 0.0
            bracket_seed = int(bracket_seed) if bracket_seed not in [None, ""] else 0
            bracket_noise = float(bracket_noise) if bracket_noise not in [None, ""] else 0.0
            bracket_cfg_rescale = float(bracket_cfg_rescale) if bracket_cfg_rescale not in [None, ""] else 0.0
            bracket_sigma_min = float(bracket_sigma_min) if bracket_sigma_min not in [None, ""] else 0.0
            bracket_sigma_max = float(bracket_sigma_max) if bracket_sigma_max not in [None, ""] else 0.0
            bracket_rho = float(bracket_rho) if bracket_rho not in [None, ""] else 0.0
        except (ValueError, TypeError) as e:
            print(f"Error converting parameters: {e}")
            # Use default values if conversion fails
            if 'bracket_count' not in locals(): bracket_count = 0
            if 'bracket_steps' not in locals(): bracket_steps = 0
            if 'bracket_cfg' not in locals(): bracket_cfg = 0.0
            if 'bracket_seed' not in locals(): bracket_seed = 0
            if 'bracket_noise' not in locals(): bracket_noise = 0.0
            if 'bracket_cfg_rescale' not in locals(): bracket_cfg_rescale = 0.0
            if 'bracket_sigma_min' not in locals(): bracket_sigma_min = 0.0
            if 'bracket_sigma_max' not in locals(): bracket_sigma_max = 0.0
            if 'bracket_rho' not in locals(): bracket_rho = 0.0
        
        # Check if bracketing is enabled
        is_bracketing = (bracket_count > 0 and 
                       (bracket_steps != 0 or bracket_cfg != 0 or bracket_seed != 0 or
                        bracket_noise != 0 or bracket_cfg_rescale != 0 or
                        bracket_sigma_min != 0 or bracket_sigma_max != 0 or bracket_rho != 0))
        
        print(f"Bracketing enabled: {is_bracketing}")
        
        if not is_bracketing:
            # Standard single generation
            result = generate_cond(
                prompt, negative_prompt, seconds_start, seconds_total,
                cfg_scale, steps, preview_every, seed, sampler_type,
                sigma_min, sigma_max, rho, cfg_interval_min, cfg_interval_max,
                cfg_rescale, file_format, file_naming, cut_to_seconds_total,
                init_audio, init_noise_level, mask_maskstart, mask_maskend,
                inpaint_audio
            )
            return [result[0], result[1], gr.update(visible=False), None] + [None] * 16
        
        # Get the current active model and config
        if active_model_idx not in model_instances:
            return "No active model available", None, gr.update(visible=False), None, [None] * 16
        
        original_model = model_instances[active_model_idx]["model"]
        model_config = model_instances[active_model_idx]["config"]
        
        # Determine device from parameters rather than model attribute
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Memory management: Let's start with a clean state
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f}GB / {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
        
        # Let's do a sequential generation approach instead of parallel to avoid OOM issues
        results = []
        
        # Print base values
        print(f"Base values - Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}, "
              f"Noise: {init_noise_level}, Rescale: {cfg_rescale}, "
              f"Sigma Min: {sigma_min}, Sigma Max: {sigma_max}, Rho: {rho}")
        
        # For each variation
        for i in range(bracket_count):
            var_idx = i + 1
            
            # Calculate parameter values for this variation with bounds checking
            var_steps = max(1, min(steps + (bracket_steps * var_idx), 500))
            var_cfg = max(0, min(cfg_scale + (bracket_cfg * var_idx), 25.0))
            
            # Handle seed specially
            if seed == -1:
                var_seed = -1  # Will generate a random seed in generate_cond
            else:
                var_seed = seed + (bracket_seed * var_idx)
            
            var_noise = max(0.01, min(init_noise_level + (bracket_noise * var_idx), 100.0))
            var_rescale = max(-1.0, min(cfg_rescale + (bracket_cfg_rescale * var_idx), 1.0))
            var_sigma_min = max(0.0, min(sigma_min + (bracket_sigma_min * var_idx), 2.0))
            var_sigma_max = max(var_sigma_min, min(sigma_max + (bracket_sigma_max * var_idx), 1000.0))
            var_rho = max(0.0, min(rho + (bracket_rho * var_idx), 10.0))
            
            # Print what we're generating for debugging
            print(f"Variation {var_idx}: Steps={var_steps}, CFG={var_cfg}, Noise={var_noise}, " +
                  f"SigMin={var_sigma_min}, SigMax={var_sigma_max}, Rho={var_rho}")
            
            # Create a unique base filename for this variation
            timestamp = int(time.time() * 1000) % 10000
            unique_basename = f"var{var_idx}_{timestamp}"
            
            # Call generate_cond sequentially (no threads or model copies)
            try:
                result = generate_cond(
                    prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    seconds_start=seconds_start, 
                    seconds_total=seconds_total,
                    cfg_scale=var_cfg, 
                    steps=var_steps, 
                    preview_every=preview_every, 
                    seed=var_seed, 
                    sampler_type=sampler_type,
                    sigma_min=var_sigma_min, 
                    sigma_max=var_sigma_max, 
                    rho=var_rho, 
                    cfg_interval_min=cfg_interval_min,
                    cfg_interval_max=cfg_interval_max,
                    cfg_rescale=var_rescale, 
                    file_format=file_format, 
                    file_naming=unique_basename, 
                    cut_to_seconds_total=cut_to_seconds_total,
                    init_audio=init_audio, 
                    init_noise_level=var_noise, 
                    mask_maskstart=mask_maskstart, 
                    mask_maskend=mask_maskend,
                    inpaint_audio=inpaint_audio
                )
                results.append((i, result))
                
                # Force memory cleanup after each generation
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Memory after variation {var_idx}: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")
                
            except Exception as e:
                print(f"Error generating variation {var_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Process results for the UI
        gallery_items = []
        audio_files = []
        
        for i, result in results:
            if result:
                audio_file, spectrograms = result
                
                # Store the audio file
                audio_files.append(audio_file)
                
                # Create parameter variation label
                var_idx = i + 1
                var_steps = steps + (bracket_steps * var_idx)
                var_cfg = cfg_scale + (bracket_cfg * var_idx)
                var_seed = "random" if seed == -1 else seed + (bracket_seed * var_idx)
                var_noise = init_noise_level + (bracket_noise * var_idx)
                var_rescale = cfg_rescale + (bracket_cfg_rescale * var_idx)
                var_sigma_min = sigma_min + (bracket_sigma_min * var_idx)
                var_sigma_max = sigma_max + (bracket_sigma_max * var_idx)
                var_rho = rho + (bracket_rho * var_idx)
                
                # Only show changed parameters in the label
                label = f"Variation {var_idx}: "
                if bracket_steps != 0:
                    label += f"Steps={var_steps}, "
                if bracket_cfg != 0:
                    label += f"CFG={var_cfg:.1f}, "
                if bracket_seed != 0 and seed != -1:
                    label += f"Seed={var_seed}, "
                if bracket_noise != 0:
                    label += f"Noise={var_noise:.2f}, "
                if bracket_cfg_rescale != 0:
                    label += f"Rescale={var_rescale:.2f}, "
                if bracket_sigma_min != 0:
                    label += f"SigMin={var_sigma_min:.3f}, "
                if bracket_sigma_max != 0:
                    label += f"SigMax={var_sigma_max:.1f}, "
                if bracket_rho != 0:
                    label += f"Rho={var_rho:.2f}, "
                label = label.rstrip(", ")
                
                # Handle spectrograms safely
                if spectrograms:
                    if isinstance(spectrograms, list) and spectrograms:
                        first_item = spectrograms[0]
                        # Handle different spectrogram formats
                        if isinstance(first_item, tuple):
                            gallery_items.append((first_item[0], label))
                        else:
                            gallery_items.append((first_item, label))
                    else:
                        gallery_items.append((spectrograms, label))
        
        # Debug the files we've generated
        print(f"Generated {len(audio_files)} audio files")
        
        # Prepare outputs for the UI
        std_audio = audio_files[0] if audio_files else None
        std_gallery = results[0][1][1] if results else None
        
        # Update audio players for each variation
        audio_updates = []
        for i in range(16):
            if i < len(audio_files) and audio_files[i]:
                var_idx = i + 1
                
                # Create label for audio player
                label = f"Variation {var_idx}: "
                if bracket_steps != 0:
                    label += f"Steps={steps + (bracket_steps * var_idx)}, "
                if bracket_cfg != 0:
                    label += f"CFG={cfg_scale + (bracket_cfg * var_idx):.1f}, "
                if bracket_seed != 0 and seed != -1:
                    label += f"Seed={seed + (bracket_seed * var_idx)}, "
                if bracket_noise != 0:
                    label += f"Noise={init_noise_level + (bracket_noise * var_idx):.2f}, "
                if bracket_cfg_rescale != 0:
                    label += f"Rescale={cfg_rescale + (bracket_cfg_rescale * var_idx):.2f}, "
                if bracket_sigma_min != 0:
                    label += f"SigMin={sigma_min + (bracket_sigma_min * var_idx):.3f}, "
                if bracket_sigma_max != 0:
                    label += f"SigMax={sigma_max + (bracket_sigma_max * var_idx):.1f}, "
                if bracket_rho != 0:
                    label += f"Rho={rho + (bracket_rho * var_idx):.2f}, "
                label = label.rstrip(", ")
                
                audio_updates.append(gr.update(value=audio_files[i], visible=True, label=label))
            else:
                audio_updates.append(gr.update(visible=False))
        
        return [std_audio, std_gallery, gr.update(visible=True), gallery_items] + audio_updates
    
    except Exception as e:
        print(f"Critical error in bracketing: {str(e)}")
        import traceback
        traceback.print_exc()
        return ["Error: " + str(e), None, gr.update(visible=False), None] + [None] * 16

def clear_bracketing_model_copies():
    """Clears model copies to free VRAM"""
    global bracketing_model_copies, active_bracketing_count, last_active_model_idx
    
    if active_bracketing_count > 0:
        print(f"Clearing {active_bracketing_count} model copies to free VRAM")
        
        if active_bracketing_count in bracketing_model_copies:
            for model_copy in bracketing_model_copies[active_bracketing_count]:
                del model_copy
            
            bracketing_model_copies[active_bracketing_count] = []
            torch.cuda.empty_cache()
            gc.collect()
        
        active_bracketing_count = 0
        last_active_model_idx = None
        print("Model copies cleared")

def create_diffusion_cond_ui(model_config, in_model, in_model_half=True):
    global model, sample_size, sample_rate, model_type, model_half, model_instances, active_model_idx

    # Store the model in the model_instances dictionary if it's not already there
    if not model_instances:
        model_instances[0] = {
            "model": in_model,
            "config": model_config,
            "path": "initial_model",
            "type": model_config["model_type"]
        }
        active_model_idx = 0

    # Always use the active model directly
    model = model_instances[active_model_idx]["model"] if active_model_idx in model_instances else in_model
    model_config = model_instances[active_model_idx]["config"] if active_model_idx in model_instances else model_config
    
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_type = model_config["model_type"]
    model_half = in_model_half

    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config) 
    return ui
