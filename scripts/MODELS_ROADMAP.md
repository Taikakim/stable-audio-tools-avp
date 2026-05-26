# Models we could train with the data we have

Beyond the per-feature LatCH control heads, here are candidate models, ranked by
value × feasibility × **whether the latent actually encodes the signal** (the last
factor is decisive — the sensitivity screener already showed several "obvious"
features are dead-for-guidance).

## What we have
- **202k pre-encoded latents** (`/run/media/kim/Lehto/latents`, `[64,256]`), same VAE as the base model.
- **MIR timeseries DB** (~209k crops): 14 per-frame `_ts` features.
- **`.INFO` / companion JSON**: 97+ **scalar** MIR features (lufs, bpm, key, roughness, production_complexity, danceability, …) + `music_flamingo` genre/mood/text tags.
- **Stem latents** (`/run/media/kim/Lehto/goa-stems`).
- Base Stable Audio Open (Small, RF) + its VAE.

## Empirical controllability map (from the sensitivity screener)
- **Strong (keep):** rms_energy_{bass,body,mid,air}, spectral_flux, spectral_kurtosis.
- **Weak:** spectral_skewness.
- **Dead (skip):** spectral_flatness, tonic_strength, tonic — the latent barely encodes them.
- → The screener (`latch_head_sensitivity.py`) is the gate for everything below.

## Candidates (ranked)

### 1. Controllability probe across the scalar `.INFO` features — *cheap, do first*
We've mapped the 14 `_ts` features; the ~97 **scalar** INFO fields (lufs, bpm, key,
roughness, production_complexity, danceability, brightness, mood scores, …) are
unmapped. Train tiny LatCH heads (broadcast scalar → constant target) or just probe
`‖∂pred/∂z‖` per feature → a ranked atlas of *what the latent encodes*. Decides where
all further control/conditioning effort should go. Low cost (heads are ~5M params,
~45 min each, and the probe can be much shorter). **Highest information per GPU-hour.**

### 2. Finish the controllable LatCH heads
- **Rhythm:** `beat/downbeat/onsets_activations` via `beat_grid`/BPM (BCE-logits). Screen
  first — rhythm is salient but may or may not be latent-encoded. If they pass, they're
  high-value (tempo/groove control).
- **`hpcp`** (harmony, cosine, 12-vector target) — needs the vector-target UI (key
  selector). The right tool for harmonic steering (tonic is dead).
- Skip flatness/tonic_strength/tonic.

### 3. Multi-feature LatCH head — *efficiency + joint control*
One bidirectional-transformer backbone with **multiple output heads** for all *strong*
features (the 6 keepers). Trains once, enables combined control (e.g. loud bass + high
flux) from a single model, and shares representation. Modest lift over the single-feature
trainer (multi-task loss, per-feature output proj).

### 4. Native numeric-conditioner fine-tune — *most powerful, biggest lift*
The original project goal: fine-tune the **DiT itself** to take the controllable MIR
features as **conditioning** (cross-attn / global cond, `base_model_finetune.json`),
rather than training-free guidance. Cleaner control (no ρ/μ/window/weight tuning, no
sampler override), composes naturally with the prompt. Cost: full-model fine-tune on the
pre-encoded dataset (hours–days), and only worth conditioning on features the probe shows
are learnable. LatCH heads are the cheap reversible alternative; this is the production one.

### 5. High-level style/mood control
LatCH-style **classification heads** (BCE) predicting genre/mood probability from latents
(labels from `music_flamingo` tags / `genre_labels.txt`) → guidance toward "more goa",
"more aggressive", etc. Or fold these into #4 as categorical conditioners. Value depends
on how separable the classes are in latent space — screen with a quick linear probe.

### 6. Utility predictors (not control)
Latent→{BPM, key, genre} estimators for retrieval, auto-tagging, dataset QC, or the
latent player. Cheap; reuses the LatCH architecture; useful tooling even if not for
generation control.

### 7. Stem-aware experiments — *exploratory*
Stem latents enable per-stem control (guide just the drums' energy) or latent-domain
stem manipulation, building on the existing `latent_crossfader`. Higher uncertainty.

## Recommended order
1. **Controllability probe of scalar INFO features** (#1) — cheap, decides the rest.
2. **Rhythm heads** (#2) — screen, then train the ones that pass.
3. **Multi-feature head** (#3) — consolidate the keepers for combined control.
4. Then choose the big bet: **native conditioner fine-tune** (#4) for production-grade
   control, and/or **style/mood control** (#5).

Always screen with `latch_head_sensitivity.py` before investing — the latent decides
what's possible, not our intuition about the feature.
