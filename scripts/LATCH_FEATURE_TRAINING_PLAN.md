# LatCH Feature Training Plan

Roadmap for training LatCH heads on the remaining MIR features, with **measured**
target ranges (sampled from `timeseries.db`, 300 crops) and per-feature UI-slider
guidance (range + linear vs logarithmic). All heads train with the current
pipeline: `objective rectified_flow`, robust Huber loss, held-out val split,
W&B/test-inference via `latch_train.yaml`.

## Status
- ✅ `rms_energy_bass` — trained (`v2`, `best.pt` val_median ≈ 1.82).
- ⬜ 13 features remaining (this plan).

## Measured distributions (per-frame, 300 crops)

| feature | min | p5 | median | p95 | max | shape |
|---|---|---|---|---|---|---|
| rms_energy_bass | −299* | −60 | −21 | −10 | −5 | dB |
| rms_energy_body | −1049* | −39 | −22 | −13 | −7 | dB |
| rms_energy_mid | −3079* | −41 | −25 | −15 | −8 | dB |
| rms_energy_air | −837* | −50 | −31 | −19 | −10 | dB |
| spectral_flatness | 0.001 | 0.004 | 0.075 | 0.39 | 0.67 | **~100× spread** |
| spectral_flux | 0 | 8 | 53 | 182 | 294 | positive, mild skew |
| spectral_skewness | −0.94 | 0.72 | 1.9 | 4.7 | 16 | signed |
| spectral_kurtosis | 1.4 | 2.5 | 6.5 | 35 | 385 | **heavy right tail** |
| beat_activations | 0 | 0 | 0.17 | 0.98 | 1 | 0–1 prob |
| downbeat_activations | 0 | 0 | 0 | 0.90 | 1 | 0–1 prob (sparse) |
| onsets_activations | 0 | 0.05 | 0.19 | 0.64 | 1 | 0–1 prob |
| tonic | 0 | 0 | 4 | 9 | 11 | pitch class (circular) |
| tonic_strength | 0 | 0.38 | 0.65 | 0.83 | 0.97 | 0–1 |
| hpcp | — | — | — | — | — | 256×12 chroma (vector) |

\* The extreme `rms_*` mins are near-silent-frame artifacts of `20·log10`; the
`[−60, 0]` dB clamp (auto for `rms_energy_*`) removes them.

## Per-feature spec

| feature | loss (auto) | default kind | clamp | UI slider range | **slider scale** | priority | notes |
|---|---|---|---|---|---|---|---|
| rms_energy_body | smooth_l1 | ramp_up | [−60,0] | [−55, −7] | **linear** (dB is already log) | 1 | same recipe as bass |
| rms_energy_mid | smooth_l1 | ramp_up | [−60,0] | [−55, −8] | linear | 1 | |
| rms_energy_air | smooth_l1 | ramp_up | [−60,0] | [−60, −10] | linear | 1 | |
| tonic_strength | smooth_l1 | constant | [0,1] | [0.3, 1.0] | linear | 1 | how key-defined |
| spectral_flatness | smooth_l1 | constant | floor 1e-3 | [0.001, 0.7] | **LOG** | 2 | tonal↔noisy, ~100× spread |
| spectral_flux | smooth_l1 | ramp_up | clip≈290 | [0, 200] | linear | 2 | transient/spectral motion |
| spectral_skewness | smooth_l1 | constant | none | [−1, 8] | linear (signed → can't log) | 2 | |
| spectral_kurtosis | smooth_l1 | constant | clip≈50 | [1, 50] | **LOG** | 2 | heavy tail (max 385) |
| beat_activations | bce_logits | **beat_grid** | — | **BPM [60, 200]** | linear | 3 | target value = BPM, not activation |
| downbeat_activations | bce_logits | beat_grid | — | BPM [60, 200] | linear | 3 | sparser than beats |
| onsets_activations | bce_logits | beat_grid | — | BPM [60, 200] | linear | 3 | |
| hpcp | cosine | — | — | per-pitch-class / key selector | n/a (vector) | 4 | **needs vector-target UI** |
| tonic | smooth_l1 | constant | — | [0, 11] | linear (circular!) | 4 | ill-posed under MSE — weak |

## Cross-cutting changes needed (before/with the harder features)

1. **Robust slider bounds + scale in metadata.** The UI auto-ranges sliders from
   `feature_stats.min/max`, but raw min/max are outlier-contaminated (e.g.
   kurtosis max 385, rms min −3079). Add to the checkpoint metadata:
   `slider_min`/`slider_max` from **p1/p99** (robust) and `slider_scale`
   (`linear`|`log`), computed at train time. Then the gradio slider honors them
   (a log slider for `spectral_flatness`/`spectral_kurtosis`). This is a small
   `train_latch.py` (stats) + `diffusion_cond.py` (slider) change.
2. **beat_grid = BPM.** Beat/downbeat/onset heads train with BCE-logits; at
   inference their `target_value` is **BPM** (the builder places activations on a
   grid). Verify `build_target("beat_grid", bpm, …)` matches the training target
   density; expose a BPM slider [60, 200] for `beat_grid` kind.
3. **hpcp vector target.** Cosine loss needs a 12-d chroma direction, not a
   scalar. Needs a UI key/pitch-class selector that builds the target vector.
   Defer until the scalar features are done.
4. **tonic is circular.** MSE/Huber on pitch class (0↔11 distance should be 1,
   not 11) is ill-posed → expect weak control. Either accept it as weak, or add a
   circular loss (sin/cos embedding) later. Prefer `hpcp` for harmonic steering.
5. **LR decay for longer runs.** v2 was still improving at 20 epochs; add an
   optional cosine/step scheduler and train the regression heads ~30 epochs.

## Suggested order

- **Batch 1 (easy wins, today):** `rms_energy_body/mid/air`, `tonic_strength`.
  Identical recipe to bass; linear dB sliders; verify closed-loop like bass.
- **Batch 2 (spectral):** `spectral_flatness` (LOG), `spectral_flux`,
  `spectral_skewness`, `spectral_kurtosis` (LOG). Do change #1 (log-slider
  metadata) first so the sliders are usable.
- **Batch 3 (rhythm):** `beat_activations`, `downbeat_activations`,
  `onsets_activations` — needs change #2 (BPM/beat_grid verified).
- **Batch 4 (special):** `hpcp` (change #3), `tonic` (change #4) — most UI work,
  weakest payoff; do last.

## Training command (per feature)

```bash
# regression / spectral features (clamp auto-applies for rms_energy_*):
python scripts/train_latch.py --config latch_train.yaml \
    --feature rms_energy_body --tag v2 --epochs 30

# rhythm (loss auto-selects bce_logits; set test kind=beat_grid, target=BPM):
python scripts/train_latch.py --config latch_train.yaml \
    --feature beat_activations --tag v2 --epochs 30
```
`train_all_latch.sh` loops all features but uses the old per-feature CLI; update
it to `--config latch_train.yaml --feature $f --tag v2` to get logging + test
renders, and skip `hpcp`/`tonic` until their UI changes land.
