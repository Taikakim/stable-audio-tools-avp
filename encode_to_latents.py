#!/usr/bin/env python3
"""
Encode audio files to VAE latents using the Stable Audio Open autoencoder.

Standalone script (no PyTorch Lightning dependency) that recursively finds
audio files and encodes them through the frozen autoencoder, saving the
resulting latent tensors as .latent.npy files alongside the originals
(or to a separate --output-dir mirroring the directory structure).

Usage:
    python encode_to_latents.py \
        --model-config models/checkpoints/vae_model_config.json \
        --ckpt-path models/checkpoints/vae_model.ckpt \
        --input-dir /run/media/kim/Mantu1/ai-music/Goa_Separated_crops

    # Or using the full diffusion model checkpoint (same autoencoder):
    python encode_to_latents.py \
        --model-config models/checkpoints/small/base_model_config.json \
        --ckpt-path models/checkpoints/small/base_model.ckpt \
        --input-dir /run/media/kim/Mantu1/ai-music/Goa_Separated_crops
"""

import argparse
import json
import sys
from pathlib import Path

import stable_audio_tools.rocm_env  # set HIP/MIOpen/TunableOp env before torch
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def load_autoencoder(model_config_path: str, ckpt_path: str, model_half: bool = False, device: str = "cuda"):
    """Load the autoencoder model from config and checkpoint.
    
    Handles both standalone autoencoder configs (model_type=autoencoder) and
    full diffusion model configs (model_type=diffusion_cond) — in the latter case,
    the pretransform (autoencoder) is extracted from the diffusion wrapper.
    """
    with open(model_config_path) as f:
        model_config = json.load(f)

    model_type = model_config.get("model_type", "autoencoder")

    print(f"Creating model from config (type: {model_type})")
    model = create_model_from_config(model_config)

    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = load_ckpt_state_dict(ckpt_path)
    copy_state_dict(model, state_dict)

    # If it's a diffusion model, extract the pretransform (autoencoder)
    if model_type in ("diffusion_cond", "diffusion_cond_inpaint", "diffusion_uncond"):
        if hasattr(model, "pretransform") and model.pretransform is not None:
            autoencoder = model.pretransform
            print("Extracted autoencoder from diffusion model pretransform")
        else:
            raise ValueError("Diffusion model has no pretransform/autoencoder")
    else:
        autoencoder = model

    autoencoder.eval().requires_grad_(False)

    if model_half:
        autoencoder.to(torch.float16)

    autoencoder.to(device)

    sample_rate = model_config.get("sample_rate", 44100)
    sample_size = model_config.get("sample_size", 65536)

    print(f"Model loaded — sample_rate={sample_rate}, sample_size={sample_size}")
    return autoencoder, sample_rate, sample_size, model_type


def find_audio_files(input_dir: Path, extensions: set[str]) -> list[Path]:
    """Recursively find all audio files in the input directory."""
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    files.sort()
    return files


def load_and_prepare_audio(audio_path: Path, target_sr: int, target_length: int, num_channels: int = 2) -> torch.Tensor:
    """Load an audio file, resample if needed, and pad/crop to target length.
    
    Returns tensor of shape (1, channels, samples) ready for encoding.
    """
    audio, sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    # Handle channel mismatch
    if audio.shape[0] == 1 and num_channels == 2:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > num_channels:
        audio = audio[:num_channels]

    # Pad or crop to target length
    if audio.shape[1] < target_length:
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[1]))
    elif audio.shape[1] > target_length:
        audio = audio[:, :target_length]

    # Add batch dimension: (channels, samples) -> (1, channels, samples)
    return audio.unsqueeze(0)


@torch.no_grad()
def encode_batch(autoencoder, audio_batch: torch.Tensor, model_type: str) -> np.ndarray:
    """Encode a batch of audio through the autoencoder.
    
    For standalone autoencoder models, calls model.encode() directly.
    For diffusion pretransforms, calls model.encode() which wraps the autoencoder.
    
    Returns numpy array of shape (batch, latent_dim, time_steps).
    """
    latents = autoencoder.encode(audio_batch)
    return latents.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Encode audio files to VAE latents using the Stable Audio autoencoder"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to directory containing audio files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to output directory (mirrors input structure). "
                             "If not set, saves .latent.npy alongside originals.")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of audio samples per chunk (default: from config)")
    parser.add_argument("--model-half", action="store_true",
                        help="Use FP16 inference")
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if .latent.npy already exists")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of files to encode at once")
    parser.add_argument("--extensions", type=str, default="mp3,wav,flac,ogg",
                        help="Comma-separated audio file extensions to process")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    extensions = {f".{ext.strip().lstrip('.')}" for ext in args.extensions.split(",")}

    # Load model
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    autoencoder, sample_rate, config_sample_size, model_type = load_autoencoder(
        args.model_config, args.ckpt_path, args.model_half, device
    )

    sample_size = args.sample_size if args.sample_size is not None else config_sample_size

    # Find audio files
    audio_files = find_audio_files(input_dir, extensions)
    print(f"Found {len(audio_files)} audio files")

    if not audio_files:
        print("No audio files found, exiting.")
        return

    def get_output_path(audio_path: Path) -> Path:
        """Get the .latent.npy output path for a given audio file."""
        if output_dir:
            rel = audio_path.relative_to(input_dir)
            out = output_dir / rel.with_suffix(".latent.npy")
            return out
        return audio_path.with_suffix(".latent.npy")

    # Filter out already encoded files unless --force
    if not args.force:
        to_encode = [f for f in audio_files if not get_output_path(f).exists()]
        skipped = len(audio_files) - len(to_encode)
        if skipped > 0:
            print(f"Skipping {skipped} files with existing .latent.npy")
        audio_files = to_encode

    if not audio_files:
        print("All files already encoded, exiting. Use --force to re-encode.")
        return

    print(f"Encoding {len(audio_files)} files (sample_size={sample_size}, batch_size={args.batch_size})")

    encoded_count = 0
    error_count = 0
    dtype = torch.float16 if args.model_half else torch.float32

    # Process in batches
    for batch_start in tqdm(range(0, len(audio_files), args.batch_size), desc="Encoding"):
        batch_files = audio_files[batch_start : batch_start + args.batch_size]
        
        # Load and prepare audio for the batch
        audio_tensors = []
        valid_files = []
        for audio_path in batch_files:
            try:
                audio = load_and_prepare_audio(audio_path, sample_rate, sample_size)
                audio_tensors.append(audio)
                valid_files.append(audio_path)
            except Exception as e:
                tqdm.write(f"Error loading {audio_path.name}: {e}")
                error_count += 1
                continue

        if not audio_tensors:
            continue

        # Stack into batch and encode
        audio_batch = torch.cat(audio_tensors, dim=0).to(device=device, dtype=dtype)
        
        try:
            latents = encode_batch(autoencoder, audio_batch, model_type)
        except Exception as e:
            tqdm.write(f"Error encoding batch starting at {batch_files[0].name}: {e}")
            error_count += len(valid_files)
            continue

        # Save each latent
        for i, audio_path in enumerate(valid_files):
            latent = latents[i]
            out_path = get_output_path(audio_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), latent)
            encoded_count += 1

    print(f"\nDone! Encoded {encoded_count} files, {error_count} errors.")
    if encoded_count > 0:
        # Print shape of last encoded latent for verification
        print(f"Latent shape: {latent.shape}")


if __name__ == "__main__":
    main()
