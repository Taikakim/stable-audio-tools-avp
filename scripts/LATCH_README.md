# Latent-Control Heads (LatCH) Module

This directory contains the implementation of **LatCH-F** (Forward-Simulated Noise Conditioning) for Stable Audio Open, written by [aavepyora.online](https://aavepyora.online) from Stability AI's research paper *Low-Resource Guidance for Controllable Latent Audio Diffusion* (arXiv:2603.04366).

## File Structure

### 1. `latch_model.py`
Defines the `LatCH` architecture. It is a lightweight (~7M parameter) Bidirectional Transformer that operates directly on the VAE latents. 
*   **Key Features**: Continuous Rotary Position Embeddings (RoPE), timestep sequence concatenation (stripping it before the final projection layer), and input projections adapted for `[B, 64, 256]` latents.

### 2. `latch_dataset.py`
The PyTorch `Dataset` used to feed latent/feature pairs into the training loop.
*   **Key Features**: Maps SAO `.npy` latents (from your `goa-small` drive) to their corresponding `.INFO` or `.json` metadata files. It dynamically handles the necessary padding or trimming to ensure exactly `T=256` frames and expects time-series arrays for features.

### 3. `train_latch.py`
The noise-conditioned pre-training loop.
*   **Key Features**: Forward-noises latents with the schedule of the diffusion model the head will guide — pass `--objective` (the Small base model is `rectified_flow`, i.e. linear `z_t=(1-t)·z0+t·noise`, **not** VP cosine). It switches loss by feature (BCE-logits for beat/onset, cosine for hpcp/chroma, MSE otherwise) and records `noise_schedule` + `loss_type` in the checkpoint so inference matches.
*   **Usage**: `python scripts/train_latch.py --feature rms_energy_bass --objective rectified_flow --epochs 10`
*   **Critical**: a head trained for the wrong objective produces unreliable guidance. Inference warns if a head's `noise_schedule` ≠ the model objective.

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

## Building the training database

LatCH heads are trained on `(latent, feature)` pairs. The feature side is a
SQLite **time-series database** produced by the companion project
[**mir-feature-extraction**](https://github.com/Taikakim/mir-feature-extraction)
(arXiv-feature MIR pipeline). `train_latch.py` reads it via `latch_dataset.py`.

**What the DB looks like** (`data/timeseries.db`):
- SQLite in WAL mode (safe for concurrent worker writes). One row per crop:
  `ts(key TEXT PRIMARY KEY, data BLOB)`.
- `key` = the crop's filename stem (e.g. `"Artist - Title_0"`) — the **same stem
  as the pre-encoded latent `.npy`**, which is how the two are paired.
- `data` = `gzip(msgpack({field: {s: shape, b: float32 bytes}}))` (~13–25 KB/crop).
- Every field is resampled to **256 frames (~21.5 Hz)** to match the SAO latent
  grid, so target↔latent alignment is exact. Fields are the `*_ts` features
  listed under *Keys* above.

**How to build it** (config-driven):
```bash
# in the mir-feature-extraction repo
python -m venv mir && source mir/bin/activate
pip install -r requirements.txt
python scripts/download_essentia_models.py          # HPCP / AI models
python src/master_pipeline.py --config config/master_pipeline.yaml
```
`config/master_pipeline.yaml` sets the audio paths and the stages
(`organize → track_analysis → cropping → crop_analysis`). The DB is populated in
the **`crop_analysis`** stage: `src/spectral/timeseries_features.py::extract_timeseries`
computes the 256-frame `_ts` arrays (band RMS in dB, spectral moments, HPCP,
beat/onset activations, tonic) and writes them with `db.put(crop_stem, {...})`.

- To (re)populate **only** the time-series on an existing crop set, run with
  `stages: { crop_analysis: true }` and the others `false`.
- To bootstrap from pre-existing `.INFO` time-series, use
  `scripts/migrate_timeseries_to_db.py`.
- Check coverage: `scripts/verify_features.py`, or `TimeseriesDB.open().count()`.

**Point the trainer at it** — default path is `/home/kim/Projects/mir/data/timeseries.db`;
override with `--db-path` or `db_path:` in `latch_train.yaml`. The latent `.npy`
files (same stems) go in `latent_dir`.

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
python scripts/retrofit_latch_stats.py
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
- **Start % / End %** — selective-TFG window as a fraction of the step schedule. NOTE: for the rectified-flow Small model, σ stays near 1 (α≈0) for the early steps, and guidance strength scales with α, so the **first ~20% is a dead zone** — put the window in the **back half (≈0.5–1.0)** where guidance actually has power. (Window is step-index based, so the right values differ between RF and v models.)

## Tuning the controls per feature

The strength knobs are universal; what changes per feature is the **target value's meaning**, the sensible **kind**, and the **loss** (auto-selected from the checkpoint).

### Universal strength knobs (start here)
- **Window** `[0.5, 1.0]` — guidance lives in the low-σ / high-α back half (see Start/End note above).
- **Gain ρ = μ ≈ 3–10** — the paper's 0.03 is inaudible on this setup. ρ drives variance guidance (gradient on the noisy latent); μ drives mean guidance (gradient on the clean estimate `z_{0|t}`) — μ usually does the meaningful work; keep them equal unless experimenting.
- **Weight 1.0** for a single head; only use it to *balance* when stacking multiple heads (e.g. bass 1.0 + beat 0.5).
- **γ 0.3**, **iters 4** — leave as-is; γ smooths the guidance (raise to soften), iters strengthen mean guidance per step (slower).
- **Calibration reality:** relative tracking is reliable (more target → more feature, in order); absolute level is *offset* — if you want to truly hit a value, push it high and/or raise gain. Verify a head with the closed-loop pattern in `latch_verify_rms.py`.

### Per-feature target value & kind

| Feature(s) | Target value means | Range / units | Loss (auto) | Kind | Control |
|---|---|---|---|---|---|
| `rms_energy_{bass,body,mid,air}` | loudness of that band | dB, ~[−97, −4] (higher = louder) | MSE | `constant` (steady), `ramp_up/down` (swell/fade) | **strong, local** — verified bass corr 1.000 |
| `spectral_flatness` | tonal↔noisy texture | ~[0, 1] (higher = noisier/airier) | MSE | `constant` / `ramp` | strong-ish, local |
| `spectral_{flux,skewness,kurtosis}` | spectral motion / shape | feature-specific | MSE | `constant` / `ramp` | moderate |
| `beat_activations`, `downbeat_activations`, `onsets_activations` | **BPM** (grid spacing) | beats/min | BCE-logits | **`beat_grid`** (places activations at beat positions) | rhythmic; needs `beat_grid` kind |
| `hpcp` | 12-d chroma direction | unit-ish vector | cosine | `constant` | steers harmonic content (direction, not magnitude); experimental |
| `tonic` | pitch class | 0–11, **circular** | MSE | `constant` | **weak/ill-posed** — MSE on a circular quantity; prefer `hpcp` for harmonic steering |
| `tonic_strength` | how key-defined | ~[0, 1] | MSE | `constant` | moderate |

Notes:
- The **Target value slider auto-ranges** to the head's `feature_stats`, so it's already in the right units — pick within that range.
- **dB features** are negative; "more bass" means a *less negative* value (toward −5), "less" means toward −60.
- **Beat/onset**: use `beat_grid` and set the target value to the desired **BPM**; these are logit targets (BCE), so guidance shapes *where* energy lands in time, not a level.
- **hpcp/tonic** are the weakest controls here — global/harmonic and (for tonic) circular; don't expect dB-style precision.

## Known limitations vs the paper

This implementation is **LatCH-F only** (forward-simulated noise conditioning). The paper's better-performing LatCH-B variant (trajectory-based noise conditioning) is not implemented. The sampler is deterministic Euler at 250 steps, not the paper's 100-step stochastic DDIM. Variance-guidance gradient is taken w.r.t. `cφ(z_t, t)` rather than `cφ(z_{0|t}, 0) ∘ z_{0|t}(z_t)` (cheaper but a deviation from Eq. 3). See follow-up plans for fixing these.
