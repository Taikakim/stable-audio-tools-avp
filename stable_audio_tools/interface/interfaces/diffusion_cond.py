import gc
import numpy as np
import gradio as gr
import json 
import re
import subprocess
import torch
import torchaudio
import threading 
import os, time, math
import glob

from einops import rearrange
from torchaudio import transforms as T

from ..aeiou import audio_spectrogram_image
from ...inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint
from ...inference.filename_utils import extract_params_from_filename

model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = True
diffusion_objective = None

# Navigation state - tracks generated files per model
generated_files_by_model = {}
current_file_index_by_model = {}

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

def generate_cond(
        prompt,
        negative_prompt=None,
        seconds_start=0,
        seconds_total=30,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
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
        batch_size=1,
        bracket_samplers=False,
        bracket_sigma=False,
        *sampler_selections,
        sigma_min_list="0.01, 0.03, 0.1",
        sigma_max_list="50, 100, 200"
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")
    
    # Handle bracketing parameters
    samplers_to_test = []
    sigma_mins_to_test = []
    sigma_maxs_to_test = []
    
    # Determine which samplers to test
    if bracket_samplers:
        sampler_types_list = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
        if model.diffusion_objective == "rectified_flow":
            sampler_types_list = ["euler", "rk4", "dpmpp"]
        elif model.diffusion_objective == "rf_denoiser":
            sampler_types_list = ["pingpong"]
            
        for i, selected in enumerate(sampler_selections):
            if selected and i < len(sampler_types_list):
                samplers_to_test.append(sampler_types_list[i])
    
    if not samplers_to_test:
        samplers_to_test = [sampler_type]
    
    # Determine sigma values to test
    if bracket_sigma:
        try:
            sigma_mins_to_test = [float(x.strip()) for x in sigma_min_list.split(',') if x.strip()]
            sigma_maxs_to_test = [float(x.strip()) for x in sigma_max_list.split(',') if x.strip()]
        except ValueError:
            # Fallback to single values if parsing fails
            sigma_mins_to_test = [sigma_min]
            sigma_maxs_to_test = [sigma_max]
    else:
        sigma_mins_to_test = [sigma_min]
        sigma_maxs_to_test = [sigma_max]
    
    # If bracketing is enabled, generate multiple outputs
    if bracket_samplers or bracket_sigma:
        results = []
        for test_sampler in samplers_to_test:
            for test_sigma_min in sigma_mins_to_test:
                for test_sigma_max in sigma_maxs_to_test:
                    print(f"Testing: sampler={test_sampler}, sigma_min={test_sigma_min}, sigma_max={test_sigma_max}")
                    
                    # Recursive call with single parameters
                    result = generate_cond(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seconds_start=seconds_start,
                        seconds_total=seconds_total,
                        cfg_scale=cfg_scale,
                        steps=steps,
                        preview_every=preview_every,
                        seed=seed,
                        sampler_type=test_sampler,
                        sigma_min=test_sigma_min,
                        sigma_max=test_sigma_max,
                        rho=rho,
                        cfg_interval_min=cfg_interval_min,
                        cfg_interval_max=cfg_interval_max,
                        cfg_rescale=cfg_rescale,
                        file_format=file_format,
                        file_naming=file_naming,
                        cut_to_seconds_total=cut_to_seconds_total,
                        init_audio=init_audio,
                        init_noise_level=init_noise_level,
                        mask_maskstart=mask_maskstart,
                        mask_maskend=mask_maskend,
                        inpaint_audio=inpaint_audio,
                        batch_size=batch_size,
                        bracket_samplers=False,  # Disable bracketing for recursive calls
                        bracket_sigma=False,
                        *[False] * len(sampler_selections),  # All sampler checkboxes disabled
                        sigma_min_list=sigma_min_list,
                        sigma_max_list=sigma_max_list
                    )
                    results.append(result)
        
        # Return the last result (for now - could be enhanced to return multiple)
        if results:
            return results[-1]

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
        
    #Get the device from the model
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

            #input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            init_audio = init_audio[:, :input_sample_size]

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

            #input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
            inpaint_audio = inpaint_audio[:, :input_sample_size]

        inpaint_audio = (sample_rate, inpaint_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        t = callback_info["t"]
        sigma = callback_info["sigma"]

        if diffusion_objective == "v":
            alphas, sigmas = math.cos(t * math.pi / 2), math.sin(t * math.pi / 2)
            log_snr = math.log((alphas / sigmas) + 1e-6)
        elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            log_snr = math.log(((1 - sigma) / sigma) + 1e-6)

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f} logSNR={log_snr:.3f}"))

    generate_args = {
        "model": model,
        "conditioning": conditioning,
        "negative_conditioning": negative_conditioning,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "cfg_interval": (cfg_interval_min, cfg_interval_max),
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
    output_filename = "%s.%s" % (basename, filename_extension)
    output_wav = "%s.wav" % basename

    # Cut the extra silence off the end, if the user requested a smaller seconds_total
    if cut_to_seconds_total:
        audio = audio[:,:,:seconds_total*sample_rate]

    # Encode the audio to WAV format
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # save as wav file
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

    # Store the generated file for navigation
    add_generated_file(output_filename, [audio_spectrogram, *preview_images])
    
    # Asynchronously delete the files after returning the output file, so as to prevent clutter in the directory
    if file_naming in ["verbose", "prompt"]:
        delete_files_async([output_wav, output_filename], 30)

    return (output_filename, [audio_spectrogram, *preview_images])

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes. 
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file
    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start() 

def get_model_id():
    """Get a unique identifier for the current model"""
    global model
    if hasattr(model, 'model_config'):
        return id(model.model_config)
    return id(model)

def add_generated_file(audio_file, spectrogram):
    """Add a newly generated file to the model's file list"""
    global generated_files_by_model, current_file_index_by_model
    
    model_id = get_model_id()
    if model_id not in generated_files_by_model:
        generated_files_by_model[model_id] = []
        current_file_index_by_model[model_id] = 0
    
    generated_files_by_model[model_id].append((audio_file, spectrogram))
    current_file_index_by_model[model_id] = len(generated_files_by_model[model_id]) - 1

def navigate_files(direction):
    """Navigate through generated files for current model"""
    global generated_files_by_model, current_file_index_by_model
    
    model_id = get_model_id()
    if model_id not in generated_files_by_model or not generated_files_by_model[model_id]:
        return None, None
    
    files = generated_files_by_model[model_id]
    current_idx = current_file_index_by_model[model_id]
    
    if direction == "prev":
        new_idx = (current_idx - 1) % len(files)
    else:  # next
        new_idx = (current_idx + 1) % len(files)
    
    current_file_index_by_model[model_id] = new_idx
    return files[new_idx]

def navigate_prev():
    """Navigate to previous file"""
    return navigate_files("prev")

def navigate_next():
    """Navigate to next file"""
    return navigate_files("next")

def create_sampling_ui(model_config):
    global diffusion_objective
    has_inpainting = model_config["model_type"] == "diffusion_cond_inpaint"
    
    model_conditioning_config = model_config["model"].get("conditioning", None)

    diffusion_objective = model.diffusion_objective if model is not None else "v"

    is_rf = diffusion_objective == "rectified_flow"

    is_rf_denoiser = diffusion_objective == "rf_denoiser"

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True

    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(show_label=False, placeholder="Prompt")
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
        generate_button = gr.Button("Generate", variant='primary', scale=1)

    with gr.Row(equal_height=False):
        with gr.Column():
            with gr.Row(visible = has_seconds_start or has_seconds_total):
                # Timing controls
                seconds_start_slider = gr.Slider(minimum=0, maximum=700, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                seconds_total_slider = gr.Slider(minimum=0, maximum=700, step=1, value=sample_size//sample_rate, label="Seconds total", visible=has_seconds_total)
            
            with gr.Row():
                # Steps slider
                if is_rf:
                    default_steps = 50
                elif is_rf_denoiser:
                    default_steps = 8
                else:
                    default_steps = 100
                    
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=default_steps, label="Steps")
                # CFG scale 
                default_cfg_scale = 1.0 if is_rf_denoiser else 7.0
                cfg_scale_slider = gr.Slider(
                    minimum=0.0, maximum=25.0, step=0.1, value=default_cfg_scale, 
                    label="CFG scale",
                    info="Classifier-Free Guidance strength. Higher values (7-15) follow prompts more closely but may reduce diversity. 1.0=no guidance. Use CFG interval/rescale parameters below to fine-tune behavior."
                )

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    # Seed
                    seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                    cfg_interval_min_slider = gr.Slider(
                        minimum=0.0, maximum=1, step=0.01, value=0.0, 
                        label="CFG interval min",
                        info="Start applying CFG when noise level ≥ this value. Higher values (0.5+) apply CFG only during early/noisy steps, affecting broad structure while preserving fine details."
                    )
                    cfg_interval_max_slider = gr.Slider(
                        minimum=0.0, maximum=1, step=0.01, value=1.0, 
                        label="CFG interval max",
                        info="Stop applying CFG when noise level > this value. Lower values (≤0.5) apply CFG only during late/clean steps, affecting fine details while preserving overall structure."
                    )

                with gr.Row():
                    cfg_rescale_slider = gr.Slider(
                        minimum=0.0, maximum=1, step=0.01, value=0.0, 
                        label="CFG rescale amount",
                        info="Prevents over-saturation at high CFG scales by normalizing output variance. 0.0=off, 1.0=full rescaling. Helps maintain diversity and prevent artifacts with high CFG values."
                    )

                with gr.Row():
                    # Sampler params
                    if is_rf:
                        sampler_types = ["euler", "rk4", "dpmpp"]
                        default_sampler_type = "euler"
                    elif is_rf_denoiser:
                        sampler_types = ["pingpong"]
                        default_sampler_type = "pingpong"
                    else:
                        sampler_types = ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"]
                        default_sampler_type = "dpmpp-3m-sde"
                        
                    sampler_type_dropdown = gr.Dropdown(
                        sampler_types, label="Sampler type", value=default_sampler_type,
                        info="Denoising algorithm. 'dpmpp-3m-sde' is generally recommended for quality. Different samplers may require different step counts for optimal results."
                    )
                    sigma_min_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.01, value=0.01, 
                        label="Sigma min", visible=not (is_rf or is_rf_denoiser),
                        info="Minimum noise level. Lower values create cleaner outputs but may lose fine details. Typical range: 0.01-0.1."
                    )
                    sigma_max_slider = gr.Slider(
                        minimum=0.0, maximum=1000.0, step=0.1, value=100, 
                        label="Sigma max", visible=not (is_rf or is_rf_denoiser),
                        info="Maximum noise level (starting point). Higher values allow more creativity but may reduce stability. Typical range: 50-500."
                    )
                    rho_slider = gr.Slider(
                        minimum=0.0, maximum=10.0, step=0.01, value=1.0, 
                        label="Sigma curve strength", visible=not (is_rf or is_rf_denoiser),
                        info="Controls the noise schedule curve. 1.0=linear, >1.0=more time on high noise (more creative), <1.0=more time on low noise (more refinement)."
                    )

            with gr.Accordion("Output params", open=False):
                # Output params
                with gr.Row():
                    file_format_dropdown = gr.Dropdown(["wav", "flac", "mp3 320k", "mp3 v0", "mp3 128k", "m4a aac_he_v2 64k", "m4a aac_he_v2 32k"], label="File format", value="wav")
                    file_naming_dropdown = gr.Dropdown(["verbose", "prompt", "output.wav"], label="File naming", value="output.wav")
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Spec Preview Every")
                
                    cut_to_seconds_total_checkbox = gr.Checkbox(label="Cut to seconds total", value=True)
                    autoplay_checkbox = gr.Checkbox(label="Autoplay", value=False, elem_id="autoplay")
                    infinite_radio_checkbox = gr.Checkbox(label="Infinite Radio", value=False, elem_id="infinite-radio")
                    automatic_download_checkbox = gr.Checkbox(label="Auto Download", value=False, elem_id="automatic-download")

            # Default generation tab
            with gr.Accordion("Init audio", open=False):
                init_audio_input = gr.Audio(label="Init audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                min_noise_level = 0.01 if (is_rf or is_rf_denoiser) else 0.1
                max_noise_level = 1.0 if (is_rf or is_rf_denoiser) else 100.0

                init_noise_level_slider = gr.Slider(minimum=min_noise_level, maximum=max_noise_level, step=0.01, value=0.1, label="Init noise level")

            with gr.Accordion("Inpainting", open=False, visible=has_inpainting):
                inpaint_audio_input = gr.Audio(label="Inpaint audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
                mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=10, label="Mask Start (sec)")
                mask_maskend_slider = gr.Slider(minimum=0.0, maximum=sample_size//sample_rate, step=0.1, value=sample_size//sample_rate, label="Mask End (sec)")

            with gr.Accordion("Bracketing (Multi-Parameter Testing)", open=False):
                gr.Markdown("Select multiple values to test different parameter combinations. Each enabled parameter will generate multiple outputs.")
                
                with gr.Row():
                    bracket_samplers_checkbox = gr.Checkbox(label="Enable sampler bracketing", value=False)
                    bracket_sigma_checkbox = gr.Checkbox(label="Enable sigma bracketing", value=False, visible=not (is_rf or is_rf_denoiser))
                
                # Sampler type checkboxes
                gr.Markdown("**Sampler Types** (select multiple to test)")
                sampler_checkboxes = []
                # Organize samplers in rows of 3-4 for better layout
                samplers_per_row = 3
                for i in range(0, len(sampler_types), samplers_per_row):
                    with gr.Row():
                        for j in range(samplers_per_row):
                            if i + j < len(sampler_types):
                                sampler = sampler_types[i + j]
                                is_default = sampler == default_sampler_type
                                checkbox = gr.Checkbox(label=sampler, value=is_default)
                                sampler_checkboxes.append(checkbox)
                
                # Sigma parameter lists
                with gr.Row():
                    sigma_min_list = gr.Textbox(
                        label="Sigma min values (comma-separated)", 
                        value="0.01, 0.03, 0.1",
                        visible=not (is_rf or is_rf_denoiser),
                        info="e.g. 0.01, 0.03, 0.1, 0.3"
                    )
                    sigma_max_list = gr.Textbox(
                        label="Sigma max values (comma-separated)", 
                        value="50, 100, 200",
                        visible=not (is_rf or is_rf_denoiser),
                        info="e.g. 50, 100, 200, 500"
                    )

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
                bracket_samplers_checkbox,
                bracket_sigma_checkbox,
                *sampler_checkboxes,
                sigma_min_list,
                sigma_max_list
            ]

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False, 
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False))
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            
            # Navigation buttons
            with gr.Row():
                prev_button = gr.Button("← Previous", scale=1)
                next_button = gr.Button("Next →", scale=1)
            
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

            if has_inpainting:
                send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
                send_to_inpaint_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[inpaint_audio_input])
            
            # Connect navigation buttons
            prev_button.click(fn=navigate_prev, inputs=[], outputs=[audio_output, audio_spectrogram_output])
            next_button.click(fn=navigate_next, inputs=[], outputs=[audio_output, audio_spectrogram_output])
    
    generate_button.click(fn=generate_cond, 
        inputs=inputs,
        outputs=[
            audio_output, 
            audio_spectrogram_output
        ], 
        api_name="generate")

def create_diffusion_cond_ui(model_config, in_model, in_model_half=True, gradio_title=""):
    global model, sample_size, sample_rate, model_type, model_half

    model = in_model
    sample_size = model_config["sample_size"]
    sample_rate = model_config["sample_rate"]
    model_type = model_config["model_type"]

    model_half = in_model_half

    js ="""function run_javascript_on_page_load(){
        const generateBtn = Array.from(document.querySelectorAll('button'))
            .find(btn => btn.innerText.trim() === 'Generate');
        function getAudioOutputPlayer () {
            return [...document.querySelectorAll('label')].find(label => label.textContent.trim() === 'Output audio')?.parentElement.querySelector('audio');
        }
        const infiniteRadio = document.querySelector('#infinite-radio input[type="checkbox"]');
        const autoplay = document.querySelector('#autoplay input[type="checkbox"]');
        const automaticDownload = document.querySelector('#automatic-download input[type="checkbox"]');
        let radioAutoStart = false;
        let listenersSetup = false;
        const setupListeners = () => {
            const audioEl = getAudioOutputPlayer();
            if (!audioEl) return;
            audioEl.addEventListener('loadedmetadata', () => {
                if(automaticDownload.checked){
                    downloadAudio(audioEl);
                }
                if(autoplay.checked || radioAutoStart){
                    audioEl.play();
                    radioAutoStart = false;
                }
                if(infiniteRadio.checked){
                    audioEl.addEventListener('timeupdate', function checkAudioEnd() {
                        if (audioEl.duration - audioEl.currentTime <= 1) {                            
                            generateBtn.click();
                            radioAutoStart = true;
                            audioEl.removeEventListener('timeupdate', checkAudioEnd);
                        }
                    });
                }
            });
            listenersSetup = true;
        };
        generateBtn.addEventListener('click', () => {
            if(listenersSetup) return;
            const interval = setInterval(() => {
                console.log("...")
                const audioEl = document.querySelector('audio');
                if (audioEl?.src && audioEl.src !== window.location.href) {
                    setupListeners();
                    clearInterval(interval);
                }
            }, 100);
        });
        // Respond to >> button on MacBookPro and on steering wheel during CarPlay
        if ('mediaSession' in navigator) {
            navigator.mediaSession.setActionHandler('nexttrack', () => generateBtn.click());
            navigator.mediaSession.setActionHandler('play', () => getAudioOutputPlayer()?.play());
            navigator.mediaSession.setActionHandler('pause', () => getAudioOutputPlayer()?.pause());
        }
        // Automatic Download
        function downloadAudio(audioEl) {
            const audioSrc = audioEl.src;
            const link = document.createElement('a');
            link.href = audioSrc;
            link.download = audioSrc.substring(audioSrc.lastIndexOf('/') + 1);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }  
    """

    with gr.Blocks(js=js, theme=gr.themes.Base()) as ui:
        if gradio_title:
            gr.Markdown("### %s" % gradio_title)
        with gr.Tab("Generation"):
            create_sampling_ui(model_config) 

        # JavaScript to autoplay audio immediately after generation (if autoplay enabled)
    return ui