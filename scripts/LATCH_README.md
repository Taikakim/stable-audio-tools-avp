# Latent-Control Heads (LatCH) Module

This directory contains the implementation of **LatCH-F** (Forward-Simulated Noise Conditioning) for Stable Audio Open, based on the paper *Low-Resource Guidance for Controllable Latent Audio Diffusion*.

## File Structure

### 1. `latch_model.py`
Defines the `LatCH` architecture. It is a lightweight (~7M parameter) Bidirectional Transformer that operates directly on the VAE latents. 
*   **Key Features**: Continuous Rotary Position Embeddings (RoPE), timestep sequence concatenation (stripping it before the final projection layer), and input projections adapted for `[B, 64, 256]` latents.

### 2. `latch_dataset.py`
The PyTorch `Dataset` used to feed latent/feature pairs into the training loop.
*   **Key Features**: Maps SAO `.npy` latents (from your `goa-small` drive) to their corresponding `.INFO` or `.json` metadata files. It dynamically handles the necessary padding or trimming to ensure exactly `T=256` frames and expects time-series arrays for features.

### 3. `train_latch.py`
The noise-conditioned pre-training loop.
*   **Key Features**: Uses a continuous Variance-Preserving (VP) noise schedule matching SAO. It dynamically switches loss functions based on your target (e.g., Sparse-weighted BCE for pitch, BCE for beats/syncopation, and MSE for RMS intensity). 
*   **Usage**: `python scripts/train_latch.py --feature rms_broadband --epochs 10`

### 4. `generate_latch_guided.py`
The inference module demonstrating how to apply LatCH-driven TFG (Training-Free Guidance) to an Euler sampler step.
*   **Key Features**: Applies *Selective TFG* (only applying gradients to the first 20% of the discrete denoising steps). It properly executes both Variance Guidance and Mean Guidance (with `N_iter=4` on the denoised $x_0$ estimate), integrates $\gamma$ noise augmentation, and calculates time-weighted $\mu_t$ and $\rho_t$ coefficients.

## Requirements for MIR Features
To effectively train the LatCH models, the target features provided in the `.INFO` files must be **time-series arrays of exactly 256 frames** (representing a ~21.53 Hz feature sampling rate), rather than scalar averages.

### Keys

For every single entry (track), the available time-series fields (the exact names you can pass to the --feature argument in the training script) are:

beat_activations
downbeat_activations
hpcp (this one is a 12-channel time-series)
onsets_activations
rms_energy_air
rms_energy_bass
rms_energy_body
rms_energy_mid
spectral_flatness
spectral_flux
spectral_kurtosis
spectral_skewness
tonic_strength
tonic

## Checkpoint Format (v2)

Checkpoints saved by `train_latch.py` are now wrapped dicts:

```
{
  "state_dict": {...},
  "feature_name": "spectral_flatness_ts",
  "feature_stats": {"mean": 0.18, "std": 0.09, "min": 0.01, "max": 0.55, "n_samples": 200},
  "target_kind_default": "constant"
}
```

Legacy bare-state-dict files are still loadable (with empty `metadata`), but the
inference UI cannot auto-range their target sliders. Retrofit them once with:

```bash
source rocm_env.sh && python scripts/retrofit_latch_stats.py
```

This rewrites every `latch_weights/*.pt` into the v2 format, computing
`feature_stats` from the training dataset.

## Inference UI

The Gradio LatCH panel now exposes:

- **Variance ρ** (default 1.0) — TFG variance-guidance step size; the paper-canonical 0.03 is essentially imperceptible at the schedules SAO uses, so the default is bumped to 1.0.
- **Mean μ** (default 1.0) — TFG mean-guidance step size on `z_{0|t}`.
- **Noise γ** (default 0.3) — Gaussian noise std added to clean LatCH evaluations (paper §2.2).
- **Mean iters** (default 4) — `N_iter` in the paper.
- **Log gradient norms** — when enabled, the sampler prints a per-step table of
  `||grad_var||`, `||grad_mean||`, and the relative perturbation `rho_t · ||grad_var|| / ||x||`. Useful for verifying that a guide is actually moving the latents.

Per slot:

- **Model** — dropdown of files in `latch_weights/`.
- **Target kind** — `constant` (uniform value over time), `ramp_up` / `ramp_down` (linear envelope), `beat_grid` (target value is BPM, places 1.0 logits at beat positions).
- **Target value** — slider auto-ranged to the loaded checkpoint's `feature_stats`.
- **Weight** — multiplied into the per-guide loss before differentiation (multi-control balancing).
- **Start % / End %** — selective-TFG window (paper §2.3 — default 0–20%).

## Known limitations vs the paper

This implementation is **LatCH-F only** (forward-simulated noise conditioning). The paper's better-performing LatCH-B variant (trajectory-based noise conditioning) is not implemented. The sampler is deterministic Euler at 250 steps, not the paper's 100-step stochastic DDIM. Variance-guidance gradient is taken w.r.t. `cφ(z_t, t)` rather than `cφ(z_{0|t}, 0) ∘ z_{0|t}(z_t)` (cheaper but a deviation from Eq. 3). See follow-up plans for fixing these.
