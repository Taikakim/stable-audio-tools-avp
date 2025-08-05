import gc
import numpy as np
import gradio as gr
import json 
import re
import subprocess
import torch
import torchaudio
import os
import time
import glob
from pathlib import Path

from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..interface.aeiou import audio_spectrogram_image
from ..inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import copy_state_dict, load_ckpt_state_dict
from ..inference.utils import prepare_audio

from .interfaces.diffusion_cond import create_diffusion_cond_ui

model = None
model_type = None
model_a = None
model_b = None
model_a_config = None
model_b_config = None
model_a_loaded = False
model_b_loaded = False
blended_model = None
sample_rate = 32000
sample_size = 1920000

# Global storage for discovered models and configs
available_models = []
available_configs = []
gradio_config = None

def load_gradio_config(config_path="./gradio-config.json"):
    """Load gradio configuration and scan for models and configs"""
    global gradio_config, available_models, available_configs
    
    try:
        with open(config_path, 'r') as f:
            gradio_config = json.load(f)
    except FileNotFoundError:
        # Create default config if not found
        gradio_config = {
            "model_folders": ["./unwrapped"],
            "config_folders": ["./stable_audio_tools/configs/model_configs", "./checkpoints"]
        }
        with open(config_path, 'w') as f:
            json.dump(gradio_config, f, indent=4)
        print(f"Created default gradio config at {config_path}")
    
    # Scan for models (only unwrapped models)
    available_models = []
    for folder in gradio_config.get("model_folders", []):
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.ckpt'):
                        full_path = os.path.abspath(os.path.join(root, file))
                        # Only include models that have "unwrapped" in their path (case-insensitive)
                        if "unwrapped" in full_path.lower():
                            available_models.append(full_path)
    
    # Scan for configs
    available_configs = []
    for folder in gradio_config.get("config_folders", []):
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.json'):
                        full_path = os.path.abspath(os.path.join(root, file))
                        if is_model_config(full_path):
                            available_configs.append(full_path)
    
    print(f"Found {len(available_models)} unwrapped model files and {len(available_configs)} config files")
    if len(available_models) == 0:
        print("WARNING: No unwrapped model files found. Model paths must contain 'unwrapped' to be loaded.")
        print("Use unwrap_model.py to convert training checkpoints to inference-ready models.")
    return available_models, available_configs

def is_model_config(json_path):
    """Check if a JSON file is a model config by checking if 'model_type' is the first key"""
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Check if model_type is present (doesn't need to be first key necessarily)
        return "model_type" in config and isinstance(config.get("model_type"), str)
    except:
        return False

def load_selected_model(config_path, model_path):
    """Load a model based on selected config and checkpoint paths"""
    global model, sample_rate, sample_size, model_type
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, model_config = load_model(model_config, model_path, device=device)
        
        return f"Successfully loaded model: {os.path.basename(model_path)} (Type: {model_config['model_type']})"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def get_compatible_configs(model_path):
    """Get compatible config files for a given model checkpoint"""
    global available_configs
    # For now, return all configs - could implement smarter compatibility checking
    return available_configs

def load_model_a(config_path, model_path, model_half=True):
    """Load model A"""
    global model_a, model_a_config, model_a_loaded, sample_rate, sample_size
    
    try:
        if not config_path or not model_path:
            return "Please select both config and model files"
            
        # Load config
        with open(config_path, 'r') as f:
            model_a_config = json.load(f)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_a, _ = load_model(model_a_config, model_path, device=device, model_half=model_half)
        model_a_loaded = True
        
        precision_str = "FP16" if model_half else "FP32"
        return f"Model A loaded: {os.path.basename(model_path)} (Type: {model_a_config['model_type']}, {precision_str})"
    except Exception as e:
        model_a_loaded = False
        return f"Error loading Model A: {str(e)}"

def unload_model_a():
    """Unload model A from memory"""
    global model_a, model_a_config, model_a_loaded
    
    if model_a is not None:
        del model_a
        model_a = None
        model_a_config = None
        model_a_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return "Model A unloaded from memory"
    return "Model A was not loaded"

def load_model_b(config_path, model_path, model_half=True):
    """Load model B"""
    global model_b, model_b_config, model_b_loaded
    
    try:
        if not config_path or not model_path:
            return "Please select both config and model files"
            
        # Load config
        with open(config_path, 'r') as f:
            model_b_config = json.load(f)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_b, _ = load_model(model_b_config, model_path, device=device, model_half=model_half)
        model_b_loaded = True
        
        precision_str = "FP16" if model_half else "FP32"
        return f"Model B loaded: {os.path.basename(model_path)} (Type: {model_b_config['model_type']}, {precision_str})"
    except Exception as e:
        model_b_loaded = False
        return f"Error loading Model B: {str(e)}"

def unload_model_b():
    """Unload model B from memory"""
    global model_b, model_b_config, model_b_loaded
    
    if model_b is not None:
        del model_b
        model_b = None
        model_b_config = None
        model_b_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return "Model B unloaded from memory"
    return "Model B was not loaded"

def create_dual_model_interface():
    """Create interface for dual model comparison"""
    global available_models, available_configs
    
    gr.Markdown("## Dual Model Comparison Interface")
    
    if len(available_models) == 0:
        gr.Markdown("⚠️ **No unwrapped models found!** Model paths must contain 'unwrapped' to appear in the list. Use `unwrap_model.py` to convert training checkpoints to inference-ready models.")
    
    with gr.Row():
        # Model A Selection
        with gr.Column():
            gr.Markdown("### Model A")
            with gr.Row():
                model_a_active = gr.Checkbox(label="Active", value=True)
                model_a_half = gr.Checkbox(label="Half Precision", value=True, info="Use FP16 to save memory")
                
            config_a_dropdown = gr.Dropdown(
                choices=available_configs,
                label="Config A",
                value=available_configs[0] if available_configs else None,
                interactive=True
            )
            model_a_dropdown = gr.Dropdown(
                choices=available_models,
                label="Model A", 
                value=available_models[0] if available_models else None,
                interactive=True
            )
            
            with gr.Row():
                load_a_button = gr.Button("Load A", variant="primary")
                unload_a_button = gr.Button("Unload A", variant="secondary")
            
            status_a = gr.Textbox(label="Status A", interactive=False)
            
        # Model B Selection  
        with gr.Column():
            gr.Markdown("### Model B")
            with gr.Row():
                model_b_active = gr.Checkbox(label="Active", value=False)
                model_b_half = gr.Checkbox(label="Half Precision", value=True, info="Use FP16 to save memory")
                
            config_b_dropdown = gr.Dropdown(
                choices=available_configs,
                label="Config B",
                value=available_configs[1] if len(available_configs) > 1 else (available_configs[0] if available_configs else None),
                interactive=True
            )
            model_b_dropdown = gr.Dropdown(
                choices=available_models,
                label="Model B", 
                value=available_models[1] if len(available_models) > 1 else (available_models[0] if available_models else None),
                interactive=True
            )
            
            with gr.Row():
                load_b_button = gr.Button("Load B", variant="primary")
                unload_b_button = gr.Button("Unload B", variant="secondary")
                
            status_b = gr.Textbox(label="Status B", interactive=False)
    
    # Connect buttons
    load_a_button.click(
        fn=load_model_a,
        inputs=[config_a_dropdown, model_a_dropdown, model_a_half],
        outputs=[status_a]
    )
    
    unload_a_button.click(
        fn=unload_model_a,
        inputs=[],
        outputs=[status_a]
    )
    
    load_b_button.click(
        fn=load_model_b,
        inputs=[config_b_dropdown, model_b_dropdown, model_b_half],
        outputs=[status_b]
    )
    
    unload_b_button.click(
        fn=unload_model_b,
        inputs=[],
        outputs=[status_b]
    )
    
    # Refresh button
    refresh_button = gr.Button("Refresh Model Lists", variant="secondary")
    
    def refresh_lists():
        global available_models, available_configs
        load_gradio_config()
        return (
            gr.Dropdown.update(choices=available_configs),
            gr.Dropdown.update(choices=available_models),
            gr.Dropdown.update(choices=available_configs),
            gr.Dropdown.update(choices=available_models),
            f"Found {len(available_models)} models and {len(available_configs)} configs"
        )
    
    refresh_button.click(
        fn=refresh_lists,
        inputs=[],
        outputs=[config_a_dropdown, model_a_dropdown, config_b_dropdown, model_b_dropdown, status_a]
    )
    
    return model_a_active, model_b_active, model_a_half, model_b_half

def load_preset(preset_name):
    """Load preset from presets.json"""
    try:
        with open('./presets.json', 'r') as f:
            presets = json.load(f)
        
        if preset_name in presets:
            preset = presets[preset_name]
            return (
                preset.get('prompt', ''),
                preset.get('negative_prompt', ''),
                preset.get('steps', 100),
                preset.get('cfg_scale', 7.0),
                preset.get('cfg_rescale', 0.0),
                preset.get('sigma_min', 0.03),
                preset.get('sigma_max', 500),
                preset.get('sampler_type', 'dpmpp-3m-sde'),
                f"Loaded preset: {preset_name}"
            )
    except:
        pass
    
    return ("", "", 100, 7.0, 0.0, 0.03, 500, "dpmpp-3m-sde", "Failed to load preset")

def save_preset(preset_name, prompt, negative_prompt, steps, cfg_scale, cfg_rescale, sigma_min, sigma_max, sampler_type):
    """Save current parameters as preset"""
    try:
        # Load existing presets
        try:
            with open('./presets.json', 'r') as f:
                presets = json.load(f)
        except:
            presets = {}
        
        # Add new preset
        presets[preset_name] = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'cfg_rescale': cfg_rescale,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'sampler_type': sampler_type
        }
        
        # Save presets
        with open('./presets.json', 'w') as f:
            json.dump(presets, f, indent=2)
        
        return f"Saved preset: {preset_name}"
    except Exception as e:
        return f"Failed to save preset: {str(e)}"

def get_preset_list():
    """Get list of available presets"""
    try:
        with open('./presets.json', 'r') as f:
            presets = json.load(f)
        return list(presets.keys())
    except:
        return []

def calculate_generation_count(steps_list, cfg_list, cfg_rescale_list, sigma_min_list, sigma_max_list, *samplers_and_models):
    """Calculate how many files will be generated based on bracketing settings"""
    try:
        # Extract sampler checkboxes and model states from variable arguments
        # Last two arguments should be model states
        if len(samplers_and_models) >= 2:
            model_a_active = samplers_and_models[-2]
            model_b_active = samplers_and_models[-1]
            samplers_selected = samplers_and_models[:-2] if len(samplers_and_models) > 2 else []
        else:
            model_a_active = False
            model_b_active = False
            samplers_selected = []
        
        steps_count = len([x.strip() for x in steps_list.split(',') if x.strip()]) if steps_list.strip() else 1
        cfg_count = len([x.strip() for x in cfg_list.split(',') if x.strip()]) if cfg_list.strip() else 1
        cfg_rescale_count = len([x.strip() for x in cfg_rescale_list.split(',') if x.strip()]) if cfg_rescale_list.strip() else 1
        sigma_min_count = len([x.strip() for x in sigma_min_list.split(',') if x.strip()]) if sigma_min_list.strip() else 1
        sigma_max_count = len([x.strip() for x in sigma_max_list.split(',') if x.strip()]) if sigma_max_list.strip() else 1
        sampler_count = sum(samplers_selected) if any(samplers_selected) else 1
        
        combinations = steps_count * cfg_count * cfg_rescale_count * sigma_min_count * sigma_max_count * sampler_count
        
        active_models = sum([model_a_active, model_b_active])
        total_files = combinations * max(1, active_models)
        
        return f"Will generate {total_files} files ({combinations} combinations × {active_models} models)"
    except:
        return "Error calculating file count"

def create_comparison_sampling_ui(model_a_active, model_b_active):
    """Create main sampling interface with bracketing and dual model support"""
    
    gr.Markdown("## Generation Parameters")
    
    # Preset management
    with gr.Row():
        with gr.Column(scale=2):
            preset_dropdown = gr.Dropdown(
                choices=get_preset_list(),
                label="Load Preset",
                interactive=True
            )
        with gr.Column(scale=2):
            preset_name_input = gr.Textbox(label="Preset Name", placeholder="Enter name to save")
        with gr.Column(scale=1):
            load_preset_btn = gr.Button("Load")
            save_preset_btn = gr.Button("Save")
        with gr.Column(scale=2):
            preset_status = gr.Textbox(label="Preset Status", interactive=False)
    
    # Main parameters
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here")
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, placeholder="Enter negative prompt (optional)")
        
        with gr.Column(scale=1):
            generate_button = gr.Button("Generate", variant='primary', size='lg')
            
    # Bracketing parameters
    with gr.Accordion("Bracketing Settings", open=True):
        gr.Markdown("Enter comma-separated values for each parameter to test multiple values")
        
        with gr.Row():
            steps_list = gr.Textbox(label="Steps", value="100", placeholder="e.g. 50, 100, 150")
            cfg_list = gr.Textbox(label="CFG Scale", value="7.0", placeholder="e.g. 5.0, 7.0, 10.0")
            cfg_rescale_list = gr.Textbox(label="CFG Rescale", value="0.0", placeholder="e.g. 0.0, 0.2, 0.5")
        
        with gr.Row():
            sampler_type_dropdown = gr.Dropdown(
                ["dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-adaptive", "k-dpm-fast", "v-ddim", "v-ddim-cfgpp"], 
                label="Default Sampler", value="dpmpp-3m-sde",
                info="Default sampler when not using bracketing"
            )
        
        with gr.Row():
            sigma_min_list = gr.Textbox(label="Sigma Min", value="0.03", placeholder="e.g. 0.01, 0.03, 0.1")
            sigma_max_list = gr.Textbox(label="Sigma Max", value="500", placeholder="e.g. 100, 300, 500")
        
        # Sampler checkboxes
        gr.Markdown("**Select Samplers to Test:**")
        samplers = ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2"]
        sampler_checkboxes = []
        
        with gr.Row():
            for i, sampler in enumerate(samplers):
                checkbox = gr.Checkbox(label=sampler, value=(i == 1))  # Default to dpmpp-3m-sde
                sampler_checkboxes.append(checkbox)
        
        # Generation count display
        generation_count_display = gr.Textbox(label="Generation Count", interactive=False, value="Will generate 1 files")
        
        # Update count when any parameter changes
        for component in [steps_list, cfg_list, cfg_rescale_list, sigma_min_list, sigma_max_list, model_a_active, model_b_active] + sampler_checkboxes:
            component.change(
                fn=calculate_generation_count,
                inputs=[steps_list, cfg_list, cfg_rescale_list, sigma_min_list, sigma_max_list, *sampler_checkboxes, model_a_active, model_b_active],
                outputs=[generation_count_display]
            )
    
    # Additional parameters
    with gr.Accordion("Advanced Parameters", open=False):
        with gr.Row(visible=True):
            # Timing controls
            seconds_start_slider = gr.Slider(minimum=0, maximum=700, step=1, value=0, label="Seconds start")
            seconds_total = gr.Slider(minimum=1, maximum=700, value=30, step=1, label="Duration (seconds)")
            batch_size = gr.Slider(minimum=1, maximum=8, value=1, step=1, label="Batch Size")
        
        with gr.Row():
            seed_input = gr.Textbox(label="Seed (-1 for random)", value="-1")
            
            cfg_interval_min_slider = gr.Slider(
                minimum=0.0, maximum=1, step=0.01, value=0.0, 
                label="CFG interval min",
                info="Start applying CFG when noise level ≥ this value"
            )
            cfg_interval_max_slider = gr.Slider(
                minimum=0.0, maximum=1, step=0.01, value=1.0, 
                label="CFG interval max", 
                info="Stop applying CFG when noise level > this value"
            )

        with gr.Row():
            rho_slider = gr.Slider(
                minimum=0.0, maximum=10.0, step=0.01, value=1.0, 
                label="Sigma curve strength",
                info="Controls noise schedule curve. 1.0=linear, >1.0=more creative, <1.0=more refinement"
            )
            preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Spec Preview Every")
    
    # Output parameters
    with gr.Accordion("Output Parameters", open=False):
        with gr.Row():
            file_format_dropdown = gr.Dropdown(
                ["wav", "flac", "mp3 320k", "mp3 v0", "mp3 128k", "m4a aac_he_v2 64k", "m4a aac_he_v2 32k"], 
                label="File format", value="wav"
            )
            file_naming_dropdown = gr.Dropdown(
                ["verbose", "prompt", "output.wav"], 
                label="File naming", value="verbose"
            )
            save_permanently_checkbox = gr.Checkbox(
                label="Save files permanently", 
                value=False,
                info="Keep generated files instead of auto-deleting after 30 seconds"
            )
            cut_to_seconds_total_checkbox = gr.Checkbox(label="Cut to seconds total", value=True)

    # Init audio
    with gr.Accordion("Init Audio", open=False):
        init_audio_input = gr.Audio(label="Init audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
        init_noise_level_slider = gr.Slider(minimum=0.01, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

    # Inpainting
    with gr.Accordion("Inpainting", open=False):
        inpaint_audio_input = gr.Audio(label="Inpaint audio", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
        mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=120, step=0.1, value=10, label="Mask Start (sec)")
        mask_maskend_slider = gr.Slider(minimum=0.0, maximum=120, step=0.1, value=40, label="Mask End (sec)")
    
    # Audio outputs section
    audio_output_a, spectrogram_a, params_a, audio_output_b, spectrogram_b, params_b = create_audio_outputs_section()
    
    # Connect generate button
    generate_button.click(
        fn=generate_dual_model_comparison,
        inputs=[
            prompt, negative_prompt, steps_list, cfg_list, cfg_rescale_list,
            sigma_min_list, sigma_max_list, sampler_type_dropdown, *sampler_checkboxes,
            seed_input, seconds_start_slider, seconds_total, batch_size, 
            cfg_interval_min_slider, cfg_interval_max_slider, rho_slider,
            preview_every_slider, file_format_dropdown, file_naming_dropdown,
            save_permanently_checkbox, cut_to_seconds_total_checkbox, init_audio_input, init_noise_level_slider,
            mask_maskstart_slider, mask_maskend_slider, inpaint_audio_input,
            model_a_active, model_b_active
        ],
        outputs=[audio_output_a, spectrogram_a, audio_output_b, spectrogram_b, params_a, params_b]
    )
    
    # Connect preset functionality
    load_preset_btn.click(
        fn=load_preset,
        inputs=[preset_dropdown],
        outputs=[prompt, negative_prompt, steps_list, cfg_list, cfg_rescale_list, sigma_min_list, sigma_max_list, sampler_type_dropdown, preset_status]
    )
    
    save_preset_btn.click(
        fn=save_preset,
        inputs=[preset_name_input, prompt, negative_prompt, steps_list, cfg_list, cfg_rescale_list, sigma_min_list, sigma_max_list, sampler_type_dropdown],
        outputs=[preset_status]
    )

def create_audio_outputs_section():
    """Create dual audio preview containers"""
    gr.Markdown("## Audio Outputs")
    
    with gr.Row():
        # Model A outputs
        with gr.Column():
            gr.Markdown("### Model A Results")
            audio_output_a = gr.Audio(label="Audio A", interactive=False)
            spectrogram_a = gr.Gallery(label="Spectrogram A", show_label=False)
            
            with gr.Row():
                prev_a_btn = gr.Button("← Prev A")
                next_a_btn = gr.Button("Next A →")
            
            # Generation parameters display for Model A
            params_a = gr.Textbox(label="Generation Parameters A", interactive=False, lines=3)
        
        # Model B outputs
        with gr.Column():
            gr.Markdown("### Model B Results")
            audio_output_b = gr.Audio(label="Audio B", interactive=False)
            spectrogram_b = gr.Gallery(label="Spectrogram B", show_label=False)
            
            with gr.Row():
                prev_b_btn = gr.Button("← Prev B")
                next_b_btn = gr.Button("Next B →")
            
            # Generation parameters display for Model B
            params_b = gr.Textbox(label="Generation Parameters B", interactive=False, lines=3)
    
    return audio_output_a, spectrogram_a, params_a, audio_output_b, spectrogram_b, params_b

def create_tooltips_section():
    """Create informational tooltips section at bottom"""
    with gr.Accordion("Parameter Information", open=False):
        gr.Markdown("""
        ### Parameter Explanations
        
        **CFG Scale**: Classifier-Free Guidance strength. Higher values (7-15) follow prompts more closely but may reduce diversity. 1.0=no guidance.
        
        **CFG Rescale**: Prevents over-saturation at high CFG scales by normalizing output variance. 0.0=off, 1.0=full rescaling.
        
        **Steps**: Number of denoising steps. More steps generally improve quality but take longer. 50-150 is typically good.
        
        **Sigma Min/Max**: Noise schedule parameters. Min controls final detail level, Max controls initial creativity.
        
        **Samplers**: Different denoising algorithms. 'dpmpp-3m-sde' is generally recommended for quality.
        
        **Bracketing**: Enable multiple parameter testing. Enter comma-separated values to test different combinations.
        
        **Models**: Load different checkpoints to compare results. Use Active checkbox to enable/disable models for generation.
        """)

def generate_dual_model_comparison(
    prompt,
    negative_prompt,
    steps_list,
    cfg_list, 
    cfg_rescale_list,
    sigma_min_list,
    sigma_max_list,
    *remaining_args
):
    """Generate audio with bracketing for active models"""
    global model_a, model_b, model_a_config, model_b_config, model_a_loaded, model_b_loaded
    
    # Extract arguments from remaining_args
    # Expected order: sampler_checkboxes, seed_input, seconds_start, seconds_total, batch_size,
    # cfg_interval_min, cfg_interval_max, rho, preview_every, file_format, file_naming,
    # save_permanently, cut_to_seconds_total, init_audio, init_noise_level, mask_maskstart, mask_maskend, 
    # inpaint_audio, model_a_active, model_b_active
    
    if len(remaining_args) >= 2:
        # Extract the last 2 arguments as model states
        model_a_active = remaining_args[-2]
        model_b_active = remaining_args[-1]
        
        # Extract other parameters
        if len(remaining_args) >= 12:
            # Extract all parameters
            remaining_params = remaining_args[:-2]  # Remove model states
            
            # Extract default sampler (string) and then sampler checkboxes (booleans)
            default_sampler_type = remaining_params[0] if remaining_params else "dpmpp-3m-sde"
            remaining_after_sampler = remaining_params[1:] if len(remaining_params) > 1 else []
            
            # Find where sampler checkboxes end (they should be boolean values)
            sampler_end_idx = 0
            for i, arg in enumerate(remaining_after_sampler):
                if isinstance(arg, bool):
                    sampler_end_idx = i + 1
                else:
                    break
            
            sampler_checkboxes = remaining_after_sampler[:sampler_end_idx] if sampler_end_idx > 0 else []
            other_params = remaining_after_sampler[sampler_end_idx:]
            
            # Extract parameters in expected order
            if len(other_params) >= 10:
                seed_input = other_params[0]
                seconds_start = other_params[1] 
                seconds_total = other_params[2]
                batch_size = other_params[3]
                cfg_interval_min = other_params[4]
                cfg_interval_max = other_params[5]
                rho = other_params[6]
                preview_every = other_params[7]
                file_format = other_params[8]
                file_naming = other_params[9]
                save_permanently = other_params[10] if len(other_params) > 10 else False
                cut_to_seconds_total = other_params[11] if len(other_params) > 11 else True
                init_audio = other_params[12] if len(other_params) > 12 else None
                init_noise_level = other_params[13] if len(other_params) > 13 else 0.1
                mask_maskstart = other_params[14] if len(other_params) > 14 else 10
                mask_maskend = other_params[15] if len(other_params) > 15 else 40
                inpaint_audio = other_params[16] if len(other_params) > 16 else None
            else:
                # Fallback defaults
                seed_input = "-1"
                seconds_start = 0
                seconds_total = 30
                batch_size = 1
                cfg_interval_min = 0.0
                cfg_interval_max = 1.0
                rho = 1.0
                preview_every = 0
                file_format = "wav"
                file_naming = "verbose"
                save_permanently = False
                cut_to_seconds_total = True
                init_audio = None
                init_noise_level = 0.1
                mask_maskstart = 10
                mask_maskend = 40
                inpaint_audio = None
        else:
            # Fallback for insufficient parameters
            default_sampler_type = "dpmpp-3m-sde"
            sampler_checkboxes = []
            seed_input = "-1"
            seconds_start = 0
            seconds_total = 30
            batch_size = 1
            cfg_interval_min = 0.0
            cfg_interval_max = 1.0
            rho = 1.0
            preview_every = 0
            file_format = "wav"
            file_naming = "verbose"
            save_permanently = False
            cut_to_seconds_total = True
            init_audio = None
            init_noise_level = 0.1
            mask_maskstart = 10
            mask_maskend = 40
            inpaint_audio = None
    else:
        # Fallback defaults
        model_a_active = False
        model_b_active = False
        default_sampler_type = "dpmpp-3m-sde"
        sampler_checkboxes = []
        seed_input = "-1"
        seconds_start = 0
        seconds_total = 30
        batch_size = 1
        cfg_interval_min = 0.0
        cfg_interval_max = 1.0
        rho = 1.0
        preview_every = 0
        file_format = "wav"
        file_naming = "verbose"
        save_permanently = False
        cut_to_seconds_total = True
        init_audio = None
        init_noise_level = 0.1
        mask_maskstart = 10
        mask_maskend = 40
        inpaint_audio = None
    
    if not (model_a_active and model_a_loaded) and not (model_b_active and model_b_loaded):
        return None, None, None, None, "No active models loaded", "No active models loaded"
    
    # Parse bracketing parameters
    try:
        steps_values = [int(x.strip()) for x in steps_list.split(',') if x.strip()]
        cfg_values = [float(x.strip()) for x in cfg_list.split(',') if x.strip()]
        cfg_rescale_values = [float(x.strip()) for x in cfg_rescale_list.split(',') if x.strip()]
        sigma_min_values = [float(x.strip()) for x in sigma_min_list.split(',') if x.strip()]
        sigma_max_values = [float(x.strip()) for x in sigma_max_list.split(',') if x.strip()]
        
        # Get selected samplers from checkboxes or use default
        samplers = ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2"]
        selected_samplers = [samplers[i] for i, selected in enumerate(sampler_checkboxes) if selected and i < len(samplers)]
        
        # If no samplers are selected from checkboxes, use the default sampler
        if not selected_samplers:
            selected_samplers = [default_sampler_type]
            
    except ValueError as e:
        return None, None, None, None, f"Error parsing parameters: {str(e)}", f"Error parsing parameters: {str(e)}"
    
    from ..inference.filename_utils import generate_unique_filename
    import itertools
    import numpy as np
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        steps_values, cfg_values, cfg_rescale_values, 
        sigma_min_values, sigma_max_values, selected_samplers
    ))
    
    results_a = []
    results_b = []
    
    # Set seed
    seed = int(seed_input) if seed_input != "-1" else np.random.randint(0, 2**32 - 1)
    
    # Generate with Model A if active
    if model_a_active and model_a_loaded:
        for i, (steps, cfg, cfg_rescale, sigma_min, sigma_max, sampler) in enumerate(param_combinations):
            try:
                # Use the diffusion_cond generation from the existing interface
                from .interfaces.diffusion_cond import generate_cond
                
                # Temporarily set global model to model_a
                global model, model_type, sample_rate, sample_size
                old_model = model
                old_model_type = model_type
                old_sample_rate = sample_rate
                old_sample_size = sample_size
                
                model = model_a
                model_type = model_a_config["model_type"]
                sample_rate = model_a_config["sample_rate"]
                sample_size = model_a_config["sample_size"]
                
                # Generate unique filename
                base_params = {
                    "prompt": prompt,
                    "steps": steps,
                    "cfg_scale": cfg,
                    "cfg_rescale": cfg_rescale,
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "sampler": sampler,
                    "model_name": f"ModelA_{i:03d}"
                }
                
                filename = generate_unique_filename(
                    base_params=base_params,
                    seed=seed,
                    naming_style="detailed"
                )
                
                # Call model-specific generation function
                audio_file, spectrogram = generate_cond_with_model(
                    model=model_a,
                    model_type=model_a_config["model_type"],
                    sample_rate=model_a_config["sample_rate"],
                    sample_size=model_a_config["sample_size"],
                    model_prefix="ModelA",
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seconds_start=seconds_start,
                    seconds_total=seconds_total,
                    cfg_scale=cfg,
                    steps=steps,
                    preview_every=preview_every,
                    seed=seed,
                    sampler_type=sampler,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=rho,
                    cfg_interval_min=cfg_interval_min,
                    cfg_interval_max=cfg_interval_max,
                    cfg_rescale=cfg_rescale,
                    file_format=file_format,
                    file_naming=file_naming,
                    save_permanently=save_permanently,
                    cut_to_seconds_total=cut_to_seconds_total,
                    init_audio=init_audio,
                    init_noise_level=init_noise_level,
                    mask_maskstart=mask_maskstart,
                    mask_maskend=mask_maskend,
                    inpaint_audio=inpaint_audio,
                    batch_size=batch_size
                )
                
                # Store parameters for display
                param_text = f"Steps: {steps}, CFG: {cfg}, CFG Rescale: {cfg_rescale}, Sigma: {sigma_min}-{sigma_max}, Sampler: {sampler}, Seed: {seed}"
                
                results_a.append({
                    'audio': audio_file,
                    'spectrogram': spectrogram,
                    'params': param_text,
                    'filename': filename
                })
                
            except Exception as e:
                print(f"Error generating with Model A: {str(e)}")
                continue
    
    # Generate with Model B if active (similar logic)
    if model_b_active and model_b_loaded:
        for i, (steps, cfg, cfg_rescale, sigma_min, sigma_max, sampler) in enumerate(param_combinations):
            try:
                # Generate unique filename
                base_params = {
                    "prompt": prompt,
                    "steps": steps,
                    "cfg_scale": cfg,
                    "cfg_rescale": cfg_rescale,
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "sampler": sampler,
                    "model_name": f"ModelB_{i:03d}"
                }
                
                filename = generate_unique_filename(
                    base_params=base_params,
                    seed=seed,
                    naming_style="detailed"
                )
                
                # Call model-specific generation function for Model B
                audio_file, spectrogram = generate_cond_with_model(
                    model=model_b,
                    model_type=model_b_config["model_type"],
                    sample_rate=model_b_config["sample_rate"],
                    sample_size=model_b_config["sample_size"],
                    model_prefix="ModelB",
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seconds_start=seconds_start,
                    seconds_total=seconds_total,
                    cfg_scale=cfg,
                    steps=steps,
                    preview_every=preview_every,
                    seed=seed,
                    sampler_type=sampler,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=rho,
                    cfg_interval_min=cfg_interval_min,
                    cfg_interval_max=cfg_interval_max,
                    cfg_rescale=cfg_rescale,
                    file_format=file_format,
                    file_naming=file_naming,
                    save_permanently=save_permanently,
                    cut_to_seconds_total=cut_to_seconds_total,
                    init_audio=init_audio,
                    init_noise_level=init_noise_level,
                    mask_maskstart=mask_maskstart,
                    mask_maskend=mask_maskend,
                    inpaint_audio=inpaint_audio,
                    batch_size=batch_size
                )
                
                # Store parameters for display
                param_text = f"Steps: {steps}, CFG: {cfg}, CFG Rescale: {cfg_rescale}, Sigma: {sigma_min}-{sigma_max}, Sampler: {sampler}, Seed: {seed}"
                
                results_b.append({
                    'audio': audio_file,
                    'spectrogram': spectrogram,
                    'params': param_text,
                    'filename': filename
                })
                
            except Exception as e:
                print(f"Error generating with Model B: {str(e)}")
                continue
    
    # Return first results for display
    audio_a = results_a[0]['audio'] if results_a else None
    spec_a = results_a[0]['spectrogram'] if results_a else None
    params_a = results_a[0]['params'] if results_a else "No Model A results"
    
    audio_b = results_b[0]['audio'] if results_b else None
    spec_b = results_b[0]['spectrogram'] if results_b else None
    params_b = results_b[0]['params'] if results_b else "No Model B results"
    
    return audio_a, spec_a, audio_b, spec_b, params_a, params_b

def generate_cond_with_model(model, model_type, sample_rate, sample_size, model_prefix, **kwargs):
    """
    Generate audio with explicit model parameters to avoid global state conflicts.
    This function isolates model operations for dual model support.
    """
    import copy
    import subprocess
    import torch
    import torchaudio
    import numpy as np
    import math
    from einops import rearrange
    from torchaudio import transforms as T
    
    # Use the correct relative import pattern for this file location
    from .interfaces.diffusion_cond import condense_prompt, delete_files_async
    from .aeiou import audio_spectrogram_image
    from ..inference.generation import generate_diffusion_cond, generate_diffusion_cond_inpaint
    
    # Extract parameters
    prompt = kwargs.get('prompt', '')
    negative_prompt = kwargs.get('negative_prompt', None)
    seconds_start = kwargs.get('seconds_start', 0)
    seconds_total = kwargs.get('seconds_total', 30)
    cfg_scale = kwargs.get('cfg_scale', 6.0)
    steps = kwargs.get('steps', 250)
    preview_every = kwargs.get('preview_every', None)
    seed = kwargs.get('seed', -1)
    sampler_type = kwargs.get('sampler_type', "dpmpp-3m-sde")
    sigma_min = kwargs.get('sigma_min', 0.03)
    sigma_max = kwargs.get('sigma_max', 1000)
    rho = kwargs.get('rho', 1.0)
    cfg_interval_min = kwargs.get('cfg_interval_min', 0.0)
    cfg_interval_max = kwargs.get('cfg_interval_max', 1.0)
    cfg_rescale = kwargs.get('cfg_rescale', 0.0)
    file_format = kwargs.get('file_format', "wav")
    file_naming = kwargs.get('file_naming', "verbose")
    save_permanently = kwargs.get('save_permanently', False)
    cut_to_seconds_total = kwargs.get('cut_to_seconds_total', False)
    init_audio = kwargs.get('init_audio', None)
    init_noise_level = kwargs.get('init_noise_level', 1.0)
    mask_maskstart = kwargs.get('mask_maskstart', None)
    mask_maskend = kwargs.get('mask_maskend', None)
    inpaint_audio = kwargs.get('inpaint_audio', None)
    batch_size = kwargs.get('batch_size', 1)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[{model_prefix}] Prompt: {prompt}")
    
    # Model-specific preview images list
    preview_images = []
    if preview_every == 0:
        preview_every = None

    # Set up conditioning
    conditioning_dict = {"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}
    conditioning = [conditioning_dict] * batch_size

    if negative_prompt:
        negative_conditioning_dict = {"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}
        negative_conditioning = [negative_conditioning_dict] * batch_size
    else:
        negative_conditioning = None
        
    # Get device from the model
    device = next(model.parameters()).device

    seed = int(seed)
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)

    input_sample_size = sample_size

    # Handle init audio
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

        if hasattr(model, 'dtype') and model.dtype == torch.float16:
            init_audio = init_audio.to(torch.float16)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device).to(init_audio.dtype)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]
        if audio_length > sample_size:
            init_audio = init_audio[:, :input_sample_size]

        init_audio = (sample_rate, init_audio)

    # Handle inpaint audio  
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

        if hasattr(model, 'dtype') and model.dtype == torch.float16:
            inpaint_audio = inpaint_audio.to(torch.float16)
        
        if inpaint_audio.dim() == 1:
            inpaint_audio = inpaint_audio.unsqueeze(0)
        elif inpaint_audio.dim() == 2:
            inpaint_audio = inpaint_audio.transpose(0, 1)

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(inpaint_audio.device).to(inpaint_audio.dtype)
            inpaint_audio = resample_tf(inpaint_audio)

        audio_length = inpaint_audio.shape[-1]
        if audio_length > sample_size:
            inpaint_audio = inpaint_audio[:, :input_sample_size]

        inpaint_audio = (sample_rate, inpaint_audio)

    # Progress callback with model-specific preview images
    def progress_callback(callback_info):
        nonlocal preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        t = callback_info["t"]
        sigma = callback_info["sigma"]

        diffusion_objective = getattr(model, 'diffusion_objective', 'v')
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
            preview_images.append((audio_spectrogram, f"[{model_prefix}] Step {current_step} sigma={sigma:.3f} logSNR={log_snr:.3f}"))

    # Set up generation arguments
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

    # Generate audio
    if model_type == "diffusion_cond":
        audio = generate_diffusion_cond(**generate_args)
    elif model_type == "diffusion_cond_inpaint":
        if inpaint_audio is not None:
            mask_start = int(mask_maskstart * sample_rate)
            mask_end = int(mask_maskend * sample_rate)
            inpaint_mask = torch.ones(1, sample_size, device=device)
            inpaint_mask[:, mask_start:mask_end] = 0
            generate_args.update({
                "inpaint_audio": inpaint_audio,
                "inpaint_mask": inpaint_mask
            })
        audio = generate_diffusion_cond_inpaint(**generate_args)

    # Create model-specific filename
    prompt_condensed = condense_prompt(prompt)
    timestamp = int(time.time())
    
    if file_naming == "verbose":
        cfg_filename = f"cfg{cfg_scale}"
        seed_filename = seed
        if negative_prompt:
            prompt_condensed += f".neg-{condense_prompt(negative_prompt)}"
        basename = f"{model_prefix}_{prompt_condensed}.{cfg_filename}.{seed_filename}.{timestamp}"
    elif file_naming == "prompt":
        basename = f"{model_prefix}_{prompt_condensed}.{timestamp}"
    else:
        basename = f"{model_prefix}_output.{timestamp}"

    if file_format:
        filename_extension = file_format.split(" ")[0].lower()
    else: 
        filename_extension = "wav"
    
    output_filename = f"{basename}.{filename_extension}"
    output_wav = f"{basename}.wav"

    # Cut audio if requested
    if cut_to_seconds_total:
        audio = audio[:,:,:seconds_total*sample_rate]

    # Encode to WAV
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Save WAV file
    torchaudio.save(output_wav, audio, sample_rate)

    # Convert to other formats if needed
    cmd = ""
    if file_format == "m4a aac_he_v2 32k":
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
        
    if cmd:
        cmd += " -loglevel error"
        subprocess.run(cmd, shell=True, check=True)
    
    # Generate spectrogram
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

    # Clean up files after delay (only if not saving permanently)
    if not save_permanently and file_naming in ["verbose", "prompt"]:
        delete_files_async([output_wav, output_filename], 30)
    
    # Create outputs folder if saving permanently and doesn't exist
    if save_permanently:
        outputs_dir = "outputs"
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        
        # Move files to outputs folder for permanent storage
        permanent_path = os.path.join(outputs_dir, os.path.basename(output_filename))
        if os.path.exists(output_filename):
            import shutil
            shutil.move(output_filename, permanent_path)
            output_filename = permanent_path
        
        # Also move WAV file if different from output file
        if output_wav != output_filename and os.path.exists(output_wav):
            permanent_wav_path = os.path.join(outputs_dir, os.path.basename(output_wav))
            shutil.move(output_wav, permanent_wav_path)

    return (output_filename, [audio_spectrogram, *preview_images])

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size, model_type
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
        #model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

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
    # Generate random seed if -1 is provided
    if seed == -1:
        seed = np.random.randint(0, 2**32 - 1, dtype=np.uint32)

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

def create_ui(model_config_path=None, ckpt_path=None, ckpt_files=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False, gradio_title=""):
    # Load gradio configuration and scan for models/configs
    global available_models, available_configs
    load_gradio_config()
    
    # Use legacy parameters if provided, otherwise use configuration system
    if ckpt_files is None:
        ckpt_files = available_models
    
    # Only load initial model if provided via legacy method
    if pretrained_name is not None or (model_config_path is not None and ckpt_path is not None):
        if model_config_path is not None:
            # Load config from json file
            with open(model_config_path) as f:
                model_config = json.load(f)
        else:
            model_config = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, model_half=model_half, device=device)
    else:
        # Use default config for multi-model interface
        model_config = {
            "model_type": "diffusion_cond", 
            "sample_rate": 44100, 
            "sample_size": 4194304,
            "model": {
                "conditioning": {
                    "configs": []
                }
            }
        }
    
    # Create tabbed interface with multi-model support
    with gr.Blocks(theme=gr.themes.Base()) as ui:
        if gradio_title:
            gr.Markdown("### %s" % gradio_title)
        
        # Dual model selection interface at top
        model_a_active, model_b_active, model_a_half, model_b_half = create_dual_model_interface()
        
        # Main sampling interface
        create_comparison_sampling_ui(model_a_active, model_b_active)
        
        # Tooltips section at bottom
        create_tooltips_section()
        
    return ui