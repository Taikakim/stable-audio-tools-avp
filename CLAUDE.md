# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is `stable-audio-tools`, a repository for training and inference of generative audio models from Stability AI. It's a PyTorch-based framework using PyTorch Lightning for distributed training.

## Virtual Environment Management

### Important: Environment Selection
**Before running any code, you MUST activate the correct virtual environment based on your GPU:**

```bash
# For RTX 5090 (exclusive optimization)
source sao-5090/bin/activate

# For RTX A4500 and all other GPU models  
source sao-a4500/bin/activate
```

**Environment Details:**
- **sao-5090**: Optimized exclusively for RTX 5090 with specific CUDA optimizations
- **sao-a4500**: Compatible with RTX A4500 and all other GPU models (general compatibility)

**⚠️ Critical:** Always check which environment you're connected to before executing any Python commands, model operations, or installations.

## Development Commands

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

- claude.log keeps a track of past changes, update this as you work on the repo. Add info on files changed as well as the purpose of edits.
- 
# Misc practices

- Do not remove any existing functionality without asking first.
- For the code to run, you need to activate :"source stable-audio-tools/sao-5090/bin/activate" to have all depedencies available.
- Before making decisions on implementing changes to core functionality, always double check for potential conflicts, issues and further problems down the road.
- It is VERY important to do the previous check when suggesting changes. Be wary of race conditions, overlap with variables used in different parts of the code, etc.