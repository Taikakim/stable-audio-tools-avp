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

from einops import rearrange
from torchaudio import transforms as T
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from ..aeiou import audio_spectrogram_image
from ...inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint


def _read_latch_metadata(latch_dir, model_name: str) -> dict:
    """Best-effort metadata read for a LatCH .pt file. Returns {} on any failure.
    Strips ``state_dict`` from the returned payload to avoid keeping ~20 MB of
    weight tensors alive on every UI dropdown change."""
    if not model_name or model_name == "none":
        return {}
    try:
        import torch
        path = latch_dir / model_name
        if not path.exists():
            return {}
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "feature_stats" in raw:
            return {k: v for k, v in raw.items() if k != "state_dict"}
    except Exception:
        pass
    return {}


model = None
model_type = None
sample_size = 2097152
sample_rate = 44100
model_half = True
diffusion_objective = None
preview_images = []

# Global sigma debug settings - these get set before generation and read by dit.py
sigma_debug_settings = {
    "print_sigma": False,
    "chart_sigma": False,
    "cfg_scale": 1.0,
    "cfg_rescale": 0.0,
    "cfg_interval": (0.0, 1.0),
    "data": []  # Collected during generation: list of (step, sigma, progress, cfg_active)
}

def reset_sigma_debug_data():
    """Reset the collected sigma data before a new generation."""
    sigma_debug_settings["data"] = []

def collect_sigma_data(step, sigma, progress, cfg_active):
    """Called from dit.py to collect sigma data for charting."""
    sigma_debug_settings["data"].append({
        "step": step,
        "sigma": sigma,
        "progress": progress,
        "cfg_active": cfg_active
    })

def create_sigma_chart(steps, cfg_interval, cfg_scale, cfg_rescale, latch_windows=None):
    """
    Create a chart showing sigma progression and CFG application.
    
    Args:
        steps: Total number of steps
        cfg_interval: Tuple of (min, max) for CFG interval
        cfg_scale: The CFG scale value
        cfg_rescale: The CFG rescale amount
    
    Returns:
        PIL Image of the chart
    """
    data = sigma_debug_settings["data"]
    
    if not data:
        # No data collected, return a placeholder
        fig = Figure(figsize=(5, 4), dpi=100, facecolor='black')
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(facecolor='black')
        ax.text(0.5, 0.5, 'No sigma data collected', color='white', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return Image.fromarray(rgba)
    
    # Extract data
    step_nums = [d["step"] for d in data]
    sigmas = [d["sigma"] for d in data]
    progresses = [d["progress"] for d in data]
    cfg_actives = [d["cfg_active"] for d in data]
    
    # Create figure with dark background
    fig = Figure(figsize=(5, 4), dpi=100, facecolor='black')
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(facecolor='black')
    
    # Style the axes
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    # Plot sigma as blue line
    ax.plot(step_nums, sigmas, color='#4A90D9', linewidth=2, label='Sigma (noise level)')
    
    # Plot progress as cyan dashed line
    ax.plot(step_nums, progresses, color='cyan', linewidth=1, linestyle='--', alpha=0.7, label='Progress (1-sigma)')
    
    # Find step indices where CFG interval boundaries are crossed
    cfg_start_progress = cfg_interval[0]
    cfg_end_progress = cfg_interval[1]
    
    # Find the step where progress first exceeds cfg_start_progress (CFG starts)
    cfg_start_step = None
    cfg_end_step = None
    
    for i, p in enumerate(progresses):
        if cfg_start_step is None and p >= cfg_start_progress:
            cfg_start_step = step_nums[i]
        if cfg_end_step is None and p > cfg_end_progress:
            cfg_end_step = step_nums[i]
    
    # Draw vertical lines for CFG interval boundaries
    if cfg_start_step is not None:
        ax.axvline(x=cfg_start_step, color='#00FF00', linewidth=2, linestyle='-', 
                   label=f'CFG Start (progress={cfg_start_progress:.2f})')
    
    if cfg_end_step is not None:
        ax.axvline(x=cfg_end_step, color='#FF4444', linewidth=2, linestyle='-',
                   label=f'CFG End (progress={cfg_end_progress:.2f})')
    
    # Draw horizontal dashed line for CFG rescale if non-zero
    if cfg_rescale > 0:
        ax.axhline(y=cfg_rescale, color='#FFD700', linewidth=1, linestyle=':',
                   label=f'CFG Rescale ({cfg_rescale:.2f})')
    
    # Shade the region where CFG is active
    if cfg_start_step is not None or cfg_end_step is not None:
        start_x = cfg_start_step if cfg_start_step is not None else step_nums[0]
        end_x = cfg_end_step if cfg_end_step is not None else step_nums[-1]
        ax.axvspan(start_x, end_x, alpha=0.2, color='#00FF00', label='CFG Active Region')
    
    # LatCH guidance window(s): yellow band, orange where it overlaps the CFG-active region.
    # start_pct/end_pct are fractions of the step schedule, so they map directly to step x.
    if latch_windows:
        max_step = max(step_nums) if step_nums else steps
        cfg_lo = cfg_start_step if cfg_start_step is not None else step_nums[0]
        cfg_hi = cfg_end_step if cfg_end_step is not None else step_nums[-1]
        win_labeled = ov_labeled = False
        for ls, le in latch_windows:
            lo, hi = ls * max_step, le * max_step
            if hi <= lo:
                continue
            ax.axvspan(lo, hi, alpha=0.18, color='#FFD400',
                       label=None if win_labeled else 'LatCH window')
            win_labeled = True
            ov_lo, ov_hi = max(lo, cfg_lo), min(hi, cfg_hi)
            if ov_hi > ov_lo:
                ax.axvspan(ov_lo, ov_hi, alpha=0.35, color='#FF8C00',
                           label=None if ov_labeled else 'LatCH ∩ CFG')
                ov_labeled = True

    # Set labels and title
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Value (0-1)', fontsize=10)
    ax.set_title(f'Sigma & CFG (scale={cfg_scale})', fontsize=11, color='white', pad=10)
    
    # Set axis limits
    ax.set_xlim(0, max(step_nums) if step_nums else steps)
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    # Create legend with smaller font, positioned below
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                       ncol=2, fontsize=7, facecolor='black', edgecolor='white',
                       labelcolor='white')
    
    # Adjust layout to make room for legend
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba)


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
        print_sigma=False,
        chart_sigma=False,
        file_format="wav",
        file_naming="verbose",
        cut_to_seconds_total=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_maskstart=None,
        mask_maskend=None,
        inpaint_audio=None,
        latch_enable=False,
        latch_model_1="none",
        latch_target_1=1.0,
        latch_weight_1=1.0,
        latch_start_1=0.0,
        latch_end_1=0.20,
        latch_model_2="none",
        latch_target_2=1.0,
        latch_weight_2=1.0,
        latch_start_2=0.0,
        latch_end_2=0.20,
        latch_rho=1.0,
        latch_mu=1.0,
        latch_gamma=0.3,
        latch_n_iter=4,
        latch_log_norms=False,
        latch_kind_1="constant",
        latch_kind_2="constant",
        batch_size=1,
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Set up sigma debug settings BEFORE generation
    sigma_debug_settings["print_sigma"] = print_sigma
    sigma_debug_settings["chart_sigma"] = chart_sigma
    sigma_debug_settings["cfg_scale"] = cfg_scale
    sigma_debug_settings["cfg_rescale"] = cfg_rescale
    sigma_debug_settings["cfg_interval"] = (cfg_interval_min, cfg_interval_max)
    reset_sigma_debug_data()
    
    # Reset the step counter in dit.py
    try:
        from ...models.dit import reset_step_counter
        reset_step_counter()
    except ImportError:
        pass

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

    if latch_enable:
        from pathlib import Path
        latch_dir = Path(__file__).parent.parent.parent.parent / "latch_weights"
        latch_configs = []
        for m_name, kind, value, weight, start, end in [
            (latch_model_1, latch_kind_1, latch_target_1, latch_weight_1, latch_start_1, latch_end_1),
            (latch_model_2, latch_kind_2, latch_target_2, latch_weight_2, latch_start_2, latch_end_2),
        ]:
            if m_name != "none":
                latch_configs.append({
                    "model_path": str(latch_dir / m_name),
                    "kind": kind,
                    "value": value,
                    "weight": weight,
                    "start_pct": start,
                    "end_pct": end,
                })
        if latch_configs:
            generate_args["latch_configs"] = latch_configs
            generate_args["latch_hparams"] = {
                "rho": float(latch_rho),
                "mu": float(latch_mu),
                "gamma": float(latch_gamma),
                "n_iter": int(round(float(latch_n_iter))),
                "log_norms": bool(latch_log_norms),
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

    # Generate sigma chart if requested
    sigma_chart_image = None
    if chart_sigma:
        latch_windows = []
        if latch_enable:
            if latch_model_1 and latch_model_1 != "none":
                latch_windows.append((float(latch_start_1), float(latch_end_1)))
            if latch_model_2 and latch_model_2 != "none":
                latch_windows.append((float(latch_start_2), float(latch_end_2)))
        sigma_chart_image = create_sigma_chart(
            steps=steps,
            cfg_interval=(cfg_interval_min, cfg_interval_max),
            cfg_scale=cfg_scale,
            cfg_rescale=cfg_rescale,
            latch_windows=latch_windows,
        )

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

    # 1. Ensure Float32
    audio = audio.to(torch.float32)

    # 2. Normalize safely (avoid div by zero if audio is silent)
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio.div(peak)

    # 3. Clamp strictly between -1 and 1
    audio = audio.clamp(-1, 1).cpu()

    # 4. Save directly as Float. Torchaudio will handle the 16-bit PCM encoding.
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
    # audio_spectrogram_image uses power_to_db with db_range=[35,120] which expects int16-scaled audio
    audio_int16 = audio.clamp(-1, 1).mul(32767).to(torch.int16)
    audio_spectrogram = audio_spectrogram_image(audio_int16, sample_rate=sample_rate)

    # Build output images list
    output_images = [(audio_spectrogram, "Final Spectrogram")]
    if sigma_chart_image is not None:
        output_images.append((sigma_chart_image, "Sigma & CFG Chart"))
    output_images.extend(preview_images)

    # Asynchronously delete the files after returning the output file, so as to prevent clutter in the directory
    if file_naming in ["verbose", "prompt"]:
        delete_files_async([output_wav, output_filename], 30)

    # Return audio as (sample_rate, numpy_array) for Gradio 6 compatibility
    # Gradio 6 can't serve relative file paths; passing audio data directly works reliably
    audio_numpy = audio.numpy().T  # [channels, samples] -> [samples, channels] for Gradio
    return ((sample_rate, audio_numpy), output_images)

#  Asynchronously delete the given list of filenames after delay seconds. Sets up thread that sleeps for delay then deletes. 
def delete_files_async(filenames, delay):
    def delete_files_after_delay(filenames, delay):
        time.sleep(delay)  # Wait for the specified delay
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)  # Delete the file
    threading.Thread(target=delete_files_after_delay, args=(filenames, delay)).start() 

def create_sampling_ui(model_config):
    global diffusion_objective
    has_inpainting = model_config["model_type"] == "diffusion_cond_inpaint"
    
    model_conditioning_config = model_config["model"].get("conditioning", None)

    diffusion_objective = model.diffusion_objective

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
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds start", visible=has_seconds_start)
                seconds_total_slider = gr.Slider(minimum=0, maximum=512, step=1, value=sample_size//sample_rate, label="Seconds total", visible=has_seconds_total)
            
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
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=default_cfg_scale, label="CFG scale")

            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    # Seed
                    seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                    cfg_interval_min_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG interval min (progress)")
                    cfg_interval_max_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=1.0, label="CFG interval max (progress)")

                with gr.Row():
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")
                    # Debug checkboxes
                    print_sigma_checkbox = gr.Checkbox(label="Print sigma", value=False)
                    chart_sigma_checkbox = gr.Checkbox(label="Chart sigma", value=False)

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
                        
                    sampler_type_dropdown = gr.Dropdown(sampler_types, label="Sampler type", value=default_sampler_type)
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.01, label="Sigma min", visible=not (is_rf or is_rf_denoiser))
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=100, label="Sigma max", visible=not (is_rf or is_rf_denoiser))
                    rho_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.01, value=1.0, label="Sigma curve strength", visible=not (is_rf or is_rf_denoiser))

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

            with gr.Accordion("LatCH Guidance", open=False):
                from pathlib import Path
                latch_dir = Path(__file__).parent.parent.parent.parent / "latch_weights"
                l_models = ["none"]
                if latch_dir.exists():
                    l_models.extend(sorted([f.name for f in latch_dir.glob("*.pt")]))

                latch_enable_checkbox = gr.Checkbox(label="Enable LatCH (Overrides Sampler)", value=False)
                with gr.Row():
                    latch_rho_slider     = gr.Slider(minimum=0.0, maximum=5.0, step=0.05,
                                                     value=1.0, label="Variance ρ")
                    latch_mu_slider      = gr.Slider(minimum=0.0, maximum=5.0, step=0.05,
                                                     value=1.0, label="Mean μ")
                    latch_gamma_slider   = gr.Slider(minimum=0.0, maximum=2.0, step=0.05,
                                                     value=0.3, label="Noise γ")
                    latch_n_iter_slider  = gr.Slider(minimum=1, maximum=8, step=1,
                                                     value=4, label="Mean iters")
                    latch_log_checkbox   = gr.Checkbox(label="Log gradient norms", value=False)

                gr.Markdown("**Slot 1**")
                with gr.Row():
                    latch_model_1_dropdown = gr.Dropdown(l_models, label="Model", value="none")
                    latch_kind_1_dropdown  = gr.Dropdown(
                        ["constant", "ramp_up", "ramp_down", "beat_grid"],
                        label="Target kind", value="constant")
                with gr.Row():
                    latch_target_1_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.05,
                                                      value=1.0, label="Target value")
                    latch_weight_1_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.05,
                                                      value=1.0, label="Weight")
                with gr.Row():
                    latch_start_1_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                     value=0.0, label="Start %")
                    latch_end_1_slider   = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                     value=0.20, label="End %")

                gr.Markdown("**Slot 2**")
                with gr.Row():
                    latch_model_2_dropdown = gr.Dropdown(l_models, label="Model", value="none")
                    latch_kind_2_dropdown  = gr.Dropdown(
                        ["constant", "ramp_up", "ramp_down", "beat_grid"],
                        label="Target kind", value="constant")
                with gr.Row():
                    latch_target_2_slider = gr.Slider(minimum=0.0, maximum=5.0, step=0.05,
                                                      value=1.0, label="Target value")
                    latch_weight_2_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.05,
                                                      value=1.0, label="Weight")
                with gr.Row():
                    latch_start_2_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                     value=0.0, label="Start %")
                    latch_end_2_slider   = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                     value=0.20, label="End %")

                def _on_latch_model_change(model_name, kind):
                    md = _read_latch_metadata(latch_dir, model_name)
                    stats = md.get("feature_stats", {}) if md else {}
                    default_kind = md.get("target_kind_default", "constant") if md else "constant"
                    if kind == "beat_grid":
                        # target value is BPM, not the feature value
                        slider_min, slider_max, slider_step, slider_val = 30.0, 240.0, 1.0, 120.0
                    elif md and md.get("slider_min") is not None and md.get("slider_max") is not None:
                        # robust p1/p99 bounds from training (outlier-safe; preferred).
                        # NOTE: slider_scale=="log" (spectral_flatness/kurtosis) still renders
                        # linearly for now — a true log slider is pending (no log head trained yet).
                        slider_min = float(md["slider_min"])
                        slider_max = float(md["slider_max"])
                        slider_step = max((slider_max - slider_min) / 100.0, 1e-4)
                        slider_val = float(stats.get("mean", (slider_min + slider_max) / 2.0))
                        slider_val = min(max(slider_val, slider_min), slider_max)
                    elif stats and "min" in stats and "max" in stats:
                        # legacy heads without robust bounds
                        slider_min = float(stats["min"])
                        slider_max = float(stats["max"]) * 2.0 if stats["max"] > 0 else 1.0
                        slider_step = max((slider_max - slider_min) / 100.0, 1e-4)
                        slider_val = float(stats.get("mean", (slider_min + slider_max) / 2.0))
                    else:
                        slider_min, slider_max, slider_step, slider_val = 0.0, 5.0, 0.05, 1.0
                    return (
                        gr.update(minimum=slider_min, maximum=slider_max,
                                  step=slider_step, value=slider_val),
                        gr.update(value=default_kind if model_name != "none" else kind),
                    )

                latch_model_1_dropdown.change(
                    fn=_on_latch_model_change,
                    inputs=[latch_model_1_dropdown, latch_kind_1_dropdown],
                    outputs=[latch_target_1_slider, latch_kind_1_dropdown],
                )
                latch_model_2_dropdown.change(
                    fn=_on_latch_model_change,
                    inputs=[latch_model_2_dropdown, latch_kind_2_dropdown],
                    outputs=[latch_target_2_slider, latch_kind_2_dropdown],
                )

                def _on_latch_kind_change(model_name, kind):
                    return _on_latch_model_change(model_name, kind)[0]

                latch_kind_1_dropdown.change(
                    fn=_on_latch_kind_change,
                    inputs=[latch_model_1_dropdown, latch_kind_1_dropdown],
                    outputs=[latch_target_1_slider],
                )
                latch_kind_2_dropdown.change(
                    fn=_on_latch_kind_change,
                    inputs=[latch_model_2_dropdown, latch_kind_2_dropdown],
                    outputs=[latch_target_2_slider],
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
                print_sigma_checkbox,
                chart_sigma_checkbox,
                file_format_dropdown,
                file_naming_dropdown,
                cut_to_seconds_total_checkbox,
                init_audio_input,
                init_noise_level_slider,
                mask_maskstart_slider,
                mask_maskend_slider,
                inpaint_audio_input,
                latch_enable_checkbox,
                latch_model_1_dropdown,
                latch_target_1_slider,
                latch_weight_1_slider,
                latch_start_1_slider,
                latch_end_1_slider,
                latch_model_2_dropdown,
                latch_target_2_slider,
                latch_weight_2_slider,
                latch_start_2_slider,
                latch_end_2_slider,
                latch_rho_slider,
                latch_mu_slider,
                latch_gamma_slider,
                latch_n_iter_slider,
                latch_log_checkbox,
                latch_kind_1_dropdown,
                latch_kind_2_dropdown,
            ]

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False, 
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False))
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)
            send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

            if has_inpainting:
                send_to_inpaint_button = gr.Button("Send to inpaint audio", scale=1)
                send_to_inpaint_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[inpaint_audio_input])
    
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

    js_code ="""
        // --- Gradio 6 Audio src workaround ---
        // Gradio 6.8.0 has a bug where gr.Audio doesn't set the <audio> element's src.
        // This workaround monitors the download link and syncs the URL to the audio element.
        function fixAudioSrc() {
            const outputAudioContainer = [...document.querySelectorAll('label')]
                .find(label => label.textContent.trim() === 'Output audio')?.parentElement;
            if (!outputAudioContainer) return;

            const observer = new MutationObserver(() => {
                const downloadLink = outputAudioContainer.querySelector('a[download]');
                const audioEl = outputAudioContainer.querySelector('audio');
                if (downloadLink && audioEl) {
                    const url = downloadLink.href;
                    if (url && url !== audioEl.src && !url.endsWith('#')) {
                        console.log('[SAT] Fixing audio src:', url);
                        audioEl.src = url;
                        audioEl.load();
                    }
                }
            });
            observer.observe(outputAudioContainer, { childList: true, subtree: true, attributes: true, attributeFilter: ['href'] });
        }
        // Run fix after a short delay to ensure DOM is ready
        setTimeout(fixAudioSrc, 1000);

        // --- Original SAT features ---
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
    """

    with gr.Blocks(head=f"<script>{js_code}</script>") as ui:
        if gradio_title:
            gr.Markdown("### %s" % gradio_title)
        with gr.Tab("Generation"):
            create_sampling_ui(model_config) 

    return ui
