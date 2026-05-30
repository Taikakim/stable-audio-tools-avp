# LatCH Tuning Notes — what we've learned

Hard-won, practical lessons from training and tuning LatCH control heads on the
Small (rectified-flow) model. Read this before training a new feature or chasing
"the guidance does nothing." Companion docs: `LATCH_README.md` (reference),
`LATCH_FEATURE_TRAINING_PLAN.md` (per-feature ranges), `MODELS_ROADMAP.md`.

## Training a head

- **Forward-noising schedule MUST match the diffusion model.** Train with the
  model's objective (`rectified_flow` linear for Small, *not* VP cosine). A
  schedule mismatch ⇒ the head sees out-of-distribution latents at sampling ⇒
  useless gradients. (`--objective`, stored as `noise_schedule`.)
- **Loss must match the feature** and be applied identically at inference:
  `bce_logits` (beat/onset), `cosine` (hpcp), `smooth_l1` (everything else).
- **Use a robust loss + clamp, not raw MSE.** dB features have garbage outliers
  on the *quiet* end (log of near-silence → −800 dB); clamp rms to `[−60,0]` and
  use SmoothL1 (Huber) so outliers don't dominate.
- **Watch `val_median`, not `val_mean` or the epoch average** — the mean is
  outlier-skewed. Keep the **best-by-val-median** checkpoint.
- **Heads converge fast (~10 epochs) then overfit.** 30 epochs is ample;
  best-by-val selects the right one regardless. No need for 40 on these.
- **Standardize regression targets — CONFIRMED fix, recommend by default.**
  A small-scale feature's head outputs at the feature's tiny scale (flatness ~0.1)
  → its guidance gradient is negligible → "dead." Training/guiding in zero-mean/
  unit-std space (`--standardize`; un-standardized at inference via metadata) gives
  a usable gradient scale. **Verified: flatness sensitivity 0.10 → 0.42 (WEAK→OK).**
  Bonus: standardizing *all* regression heads makes the guidance scale **uniform
  across features**, so the same ρ/μ/weight work everywhere (no per-feature
  retuning). Recommendation: standardize by default for regression heads.

## Inference / guidance tuning

- **Window must END at 1.0 (for RF).** Guidance strength scales with α, and
  `s_t = α/Σα` makes the *early* steps a dead zone (σ≈1, α≈0). Power lives in the
  high-α tail (~80–98% of steps). Use `[0.4, 1.0]`; the old `[0.2, 0.8]` default
  discards the strongest region.
- **The paper's ρ=μ=0.03 is inaudible here** — use ~3 (subtle) → 8 (clear) → higher.
- **Weight is the real loudness knob.** It multiplies the guidance gradient, so
  `weight · ρ` is the effective strength. The ρ/μ sliders cap (now 30); **raise
  Weight (up to 50) to push harder** — this is what made `air` audible.
- **To HEAR control, A/B at opposite targets with the same seed.** Compare the
  feature's two extremes — far more obvious than one clip vs. none.
- **Per-band perceptibility differs:** `rms_energy_bass`/`body` are punchy; `air`
  (2.5–22 kHz) and other high-freq features are inherently subtler → push harder.
- **Calibration:** relative tracking is reliable (more target → more feature, in
  order) but the **absolute level is offset** — push high, or standardize, to hit
  a value. Verify with the closed-loop check, not by ear alone.
- **Target slider bounds:** use robust p1/p99 (raw min/max are outlier-trashed);
  for clamped features use the **full clamp range** so you can over-drive.

## Controllability — the meta-lesson

- **A head only guides if its output responds to the latent.** Measure
  `‖∂pred/∂z‖` and prediction variance across clips (`latch_head_sensitivity.py`).
  Tiny values ⇒ dead-for-guidance regardless of gain/weight.
- **BUT a collapsed head ≠ an unencodable feature.** The cheap linear
  encodability probe (`latch_probe_encodability.py`) showed the latent encodes
  almost *everything* well — e.g. `spectral_flatness` R²=0.87 — even where the
  trained head was input-insensitive. So the collapse is a **training** problem
  (see standardization), not missing signal.
- **Probe encodability BEFORE concluding a feature is uncontrollable**, and before
  spending GPU. The latent decides what's possible, not feature intuition.
- **Verify with waveform-diff + ears, not a feature extractor's headline number.**
  For `beat_activations`, librosa reported big "BPM shifts" on audio that the
  relative-L2 waveform diff showed barely moved (and the same seed read the same
  BPM across all targets) — the beat tracker octave-guesses on near-identical
  clips. Measure dL2 vs the *seed-scale* (different seed ≈ 1.48) to see how much
  the guidance actually changed, and compare opposite targets (g_lo↔g_hi) — if
  they don't diverge more than each does from baseline, the target is being ignored.

## Empirical results (2026-05-26)

| feature(s) | status |
|---|---|
| rms_energy_{bass,body,mid,air} | **controllable** (trained, pushed) |
| spectral_flux (sens ~7.8), spectral_kurtosis | **controllable** (trained, pushed) |
| beat_activations | **NOT usable** — closed-loop (2026-05-26) shows only a small *target-agnostic* perturbation (dL2 0.06–0.19 vs seed-scale 1.48); g90↔g150 don't diverge; "BPM shifts" were tracker noise. Drop it; do tempo via a `bpm` conditioner |
| spectral_flatness | **recovered via `--standardize`** (sens 0.10→0.42) — was a scale/collapse issue, not missing signal |
| tonic_strength | collapsed at raw scale; feature only moderately encoded (probe R²~0.34) — retry standardized |
| spectral_skewness | weak (sens ~0.19) |
| tonic, atonality, reverberation | only moderately encoded (probe R²~0.3) |
| (probe) lufs/warmth/sharpness/brightness, hpcp_*, tiv_*, bpm/syncopation | strongly encoded — good future candidates |

## Tools
- `latch_head_sensitivity.py` — screen a head (does it respond to the latent?).
- `latch_probe_encodability.py` — does the latent encode a feature at all (cheap, pre-training)?
- `latch_verify_rms.py` — closed-loop: generate → decode → re-extract → does it track the target?
- `latch_trajectory_probe.py` — σ/α/s_t per sampler (where guidance has power).
- `latch_sweep.py` — gain/window bracket.
