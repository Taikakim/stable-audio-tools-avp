import stable_audio_tools.rocm_env  # set HIP/MIOpen/TunableOp env before torch
import torch
import soundfile as sf
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
import json

model_config_path = "/home/kim/Projects/SAO/stable-audio-tools/models/unwrapped/4rhvtl0m-misty_moon/4rhvtl0m-9000.json"
ckpt_path = "/home/kim/Projects/SAO/stable-audio-tools/models/unwrapped/4rhvtl0m-misty_moon/4rhvtl0m-9000-002.ckpt"

with open(model_config_path) as f:
    model_config = json.load(f)

model = create_model_from_config(model_config)
model.load_state_dict(load_ckpt_state_dict(ckpt_path))
model = model.half().to(device)

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

print(f"Model loaded. Sample rate: {sample_rate}, Sample size: {sample_size}")

# Generate
conditioning = [{
    "prompt": "astral nuns of the non name - pagan vortex machine - {BPM 143} {Genre: Electronic, Goa Trance 8.7, Trance 5.6} {Mood: space 5.7, melodic 6.7, dream 6.1, energetic 7.2",
    "seconds_start": 0,
    "seconds_total": 47
}]

print("Generating...")
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=3,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.05,
    sigma_max=175,
    sampler_type="dpmpp-3m-sde",
    device=device
)

output = output.squeeze(0).cpu().float().numpy().T
sf.write("output.wav", output, sample_rate)
print("Saved to output.wav")
