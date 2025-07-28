# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is `stable-audio-tools`, a repository for training and inference of generative audio models from Stability AI. It's a PyTorch-based framework using PyTorch Lightning for distributed training.

## Development Commands

### Installation and Setup
```bash
# Install the package (run from repository root)
pip install .

# For development with dependencies
pip install -e .

# Login to Weights & Biases (required for training)
wandb login
```

### Training
```bash
# Start training with dataset and model configs
python3 ./train.py --dataset-config /path/to/dataset/config --model-config /path/to/model/config --name experiment_name

# Additional training flags:
# --batch-size: samples per GPU (default: 8)
# --num-gpus: GPUs per node (default: 1) 
# --num-nodes: number of GPU nodes (default: 1)
# --checkpoint-every: steps between checkpoints (default: 10000)
# --strategy: multi-GPU strategy, use "deepspeed" for ZeRO Stage 2
# --precision: floating-point precision (default: 16)
# --accum-batches: gradient accumulation batches
# --save-dir: checkpoint save directory
```

### Model Management
```bash
# Unwrap trained model checkpoints (removes training wrapper)
python3 ./unwrap_model.py --model-config /path/to/model/config --ckpt-path /path/to/wrapped/ckpt --name model_unwrap

# Pre-encode latents for faster training
python3 ./pre_encode.py --model-config /path/to/autoencoder/config --model-ckpt-path /path/to/autoencoder.ckpt --input-dir /path/to/audio --output-dir /path/to/output
```

### Inference
```bash
# Run Gradio interface for pre-trained models
python3 ./run_gradio.py --pretrained-name stabilityai/stable-audio-open-1.0

# For local models
python3 ./run_gradio.py --model-config /path/to/config --ckpt-path /path/to/unwrapped/ckpt
```

## Architecture Overview

### Core Components
- **Models** (`stable_audio_tools/models/`): Model architectures including autoencoders, diffusion models, and language models
- **Training** (`stable_audio_tools/training/`): PyTorch Lightning wrappers and training logic
- **Data** (`stable_audio_tools/data/`): Dataset loading for local files and WebDataset format
- **Inference** (`stable_audio_tools/inference/`): Generation and sampling utilities
- **Interface** (`stable_audio_tools/interface/`): Gradio web interface

### Model Types
- `autoencoder`: VAE/VQ-VAE models for audio compression
- `diffusion_uncond`: Unconditional diffusion models
- `diffusion_cond`: Conditional diffusion models (text-to-audio)
- `diffusion_cond_inpaint`: Inpainting diffusion models
- `diffusion_autoencoder`: Latent diffusion models
- `lm`: Language models for audio

### Configuration System
- Model configs define architectures and training settings (`stable_audio_tools/configs/model_configs/`)
- Dataset configs specify data sources (`stable_audio_tools/configs/dataset_configs/`)
- All configs are JSON files loaded at runtime

### Training Process
1. Models are wrapped in PyTorch Lightning modules during training
2. Wrapped checkpoints include optimizers, discriminators, EMA copies
3. Use `unwrap_model.py` to extract just the model weights for inference
4. Pre-encoding with autoencoders speeds up latent diffusion training

### Key Files
- `train.py`: Main training script
- `run_gradio.py`: Inference interface
- `unwrap_model.py`: Extract model from training wrapper
- `pre_encode.py`: Pre-encode audio to latents
- `defaults.ini`: Default configuration file

## Development Notes

- Requires PyTorch 2.5+ for Flash Attention and Flex Attention
- Development done in Python 3.10
- Uses PyTorch Lightning for multi-GPU/multi-node training
- Weights & Biases integration for experiment tracking
- Supports both local audio files and S3 WebDataset format
- Model checkpoints can be large due to training wrappers - always unwrap for inference
# Misc practices

- Do not remove any existing functionality without asking first.
- For the code to run, you need to activate :"source stable-audio-tools/sao-5090/bin/activate" to have all depedencies available.