"""
Baseline inference: generate audio from the pretrained (unfinetuned) model
using the 4 test prompts from model_config_test.json.
"""
import matplotlib
matplotlib.use('Agg')

import stable_audio_tools.rocm_env  # set HIP/MIOpen/TunableOp env before torch
import torch
import torchaudio
import json
import os
from einops import rearrange
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Paths
MODEL_CONFIG = "models/checkpoints/small/model_config_test.json"
MODEL_CKPT = "models/checkpoints/small/model.ckpt"
OUTPUT_DIR = "baseline_demos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load config
with open(MODEL_CONFIG) as f:
    model_config = json.load(f)

# Create model
print("Creating model...")
model = create_model_from_config(model_config)

# Load checkpoint
print("Loading checkpoint...")
state_dict = load_ckpt_state_dict(MODEL_CKPT)
copy_state_dict(model, state_dict)
del state_dict

model = model.cuda().eval()
model.model.half()  # half precision for the backbone

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

# Test prompts from model_config_test.json
prompts = [
    {"prompt": "an upbeat goa trance tune", "seconds_total": 6},
    {"prompt": "an aggressive acid trance track", "seconds_total": 10},
    {"prompt": "soaring, melodic psychedelic goa trance with soaring arpeggios and filtered lead melodies", "seconds_total": 6},
    {"prompt": "a techno beat", "seconds_total": 1},
]

device = next(model.parameters()).device

for i, cond in enumerate(prompts):
    print(f"\n--- Generating: '{cond['prompt']}' (seconds_total={cond['seconds_total']}) ---")
    
    conditioning = [cond]
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        audio = generate_diffusion_cond(
            model=model,
            conditioning=conditioning,
            steps=8,  # rf_denoiser uses few steps
            cfg_scale=1.0,
            sample_size=sample_size,
            seed=42 + i,
            device=device,
            batch_size=1,
            sampler_type="pingpong",
        )
    
    # audio shape: [batch, channels, samples]
    audio = audio.squeeze(0)  # [channels, samples]
    audio = audio.to(torch.float32)
    
    # Normalize to [-1, 1]
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio / peak
    audio = audio.clamp(-1, 1).cpu()
    
    # Save
    filename = os.path.join(OUTPUT_DIR, f"baseline_{i}_{cond['prompt'][:40].replace(' ', '_')}.wav")
    torchaudio.save(filename, audio, sample_rate)
    print(f"  Saved: {filename}")

print("\nDone! All baseline demos saved to:", OUTPUT_DIR)
