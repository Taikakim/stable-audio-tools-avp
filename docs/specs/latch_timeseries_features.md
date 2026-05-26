# Spec: Per-frame time-series features for LatCH control heads

**Status:** draft · **Date:** 2026-05-26 · **Owner:** Kim
**Consumers:** `stable-audio-tools` LatCH training (`scripts/train_latch.py`,
`scripts/latch_dataset.py`) · **Producer:** the MIR project (`/home/kim/Projects/mir`)

---

## 1. Goal

Extend the MIR feature pipeline to emit **per-frame time-series for *all* features**
(rhythmic first), for **both the full mix and each stem**, computed over **whole
tracks** so we can take **continuous, arbitrary-offset crops**. This unlocks a
family of LatCH control heads (rhythmic density, per-stem activity, spectral
envelopes, harmony-over-time, …) instead of only the handful of full-mix series
that exist today.

## 2. Why (motivation from the LatCH side)

LatCH heads guide Stable Audio Open generation by predicting a per-frame MIR
feature from the noisy latent and nudging the latent toward a target. We verified:

- **The latent encodes far more than we control** — a linear ridge probe decodes
  most features at R²>0.5 (e.g. `spectral_flatness` 0.87). Almost every feature is
  a viable control candidate; we're target-data-limited, not signal-limited.
- **Continuous, dense targets work; sparse binary ones don't.** Raw binary beat
  markers are nearly undecodable (probe R²≈0.03) and gave a dead control. Smoothing
  the marker train into a continuous envelope tripled–8×'d decodability
  (`onsets` raw 0.10 → σ=3 0.34). See `scripts/latch_prototype_beat_density.py`.
- **Per-stem matters.** Full-mix onset density conflates drums + bass + synth +
  vocals. A *drum* groove control needs a *drum-stem* onset envelope — a distinct
  feature from bass/other. These do not exist yet (see §3).

## 3. Current state (audit, 2026-05-26)

What exists in `/home/kim/Projects/mir/data/timeseries.db` (SQLite, 2.7 GB, 209,235
entries — **full-mix crops only**, key = npy stem e.g. `"Artist - Title_0"`):

Per-entry time-series fields (each ~256 frames, full-mix):
`rms_energy_{bass,body,mid,air}_ts`, `spectral_{flatness,flux,skewness,kurtosis}_ts`,
`beat_activations_ts`, `downbeat_activations_ts`, `onsets_activations_ts`, `hpcp_ts`,
`tonic_ts`, `tonic_strength_ts`.

Gaps:
- **No stem time-series.** All DB keys are full-mix; there is no `*_drums_ts`.
- **Stem JSONs are not stem-specific.** `latents_stems/…_drums.json` is a *copy of
  the full-track metadata* (`source: full_mix.mp3`) plus a few per-stem **scalars**
  (`lufs_drums`, `harmonic_movement_bass`). No per-frame data.
- **Per-stem rhythm is scalar-only.** `rhythm/per_stem_rhythm.py` emits
  `onset_density_average_{stem}`, `syncopation_{stem}`, etc. — aggregates, not
  envelopes — and they aren't even present in the current crops' JSON.
- **`onsets_activations_ts` is computed on the full mix** (`spectral/
  timeseries_features.py:427`, `_compute_onset_ts(audio,…)` on the mono-summed crop).
- Features are extracted **per fixed crop**, not over the whole track, so we can't
  resample continuous windows at arbitrary offsets.

What we *do* have for stems: **VAE latents** at `/run/media/kim/Lehto/latents_stems/
<track>/<track>_<crop>_{bass,drums,other,vocals}.npy` (shape `[64, 256]`). Stem
*audio* lived on the `Mantu` drive (`Goa_Separated_crops`), currently unmounted.

## 4. Frame-alignment contract (REQUIRED)

LatCH targets must align frame-for-frame with the VAE latent the head reads.

- Stable Audio Open Small: `sample_rate=44100`, pretransform `downsampling_ratio=2048`.
- ⇒ latent frame rate = 44100 / 2048 = **21.533 Hz** (**46.44 ms/frame**).
- A training crop = `sample_size=524288` samples = 11.888 s = **256 latent frames**.
- Time-series must be sampled at **21.533 Hz** (or at a higher rate that resamples
  cleanly to it). For whole-track storage, store the full track at this frame rate;
  a crop is then a `[offset : offset+256]` window. (Verified previously: the current
  256-frame crops are correctly aligned to the latent — no time-warp.)

## 5. Requirements for the MIR producer

1. **Whole-track time-series.** Extract each feature across the entire track at the
   latent frame rate (§4), stored once per track, so any continuous window can be
   cropped at train time (enables random-offset augmentation).
2. **Per-stem AND full-mix.** For every feature, produce a full-mix series and one
   series per available stem. Drums first; bass/other next; vocals lowest priority.
3. **Rhythmic features first.** Priority order: onset-strength envelope (per stem),
   beat/downbeat activations, then a per-frame syncopation / on-beat-ratio /
   rhythmic-density envelope (the scalar `per_stem_rhythm` measures, but windowed).
4. **Store raw/continuous envelopes, not pre-smoothed.** Smoothing kind/width is a
   *training-time* knob (already implemented consumer-side — gaussian / linear /
   lowpass / beat_weighted, see §7). Storing raw lets us sweep smoothing without
   re-extraction. (Onset *strength* is naturally continuous; beat/downbeat markers
   are sparse — keep them raw, smoothing happens downstream.)
5. **Naming convention:** `<feature>_<stem>_ts` for stems (e.g.
   `onsets_activations_drums_ts`), bare `<feature>_ts` for full mix. Stem ∈
   `{drums, bass, other, vocals}`.
6. **Keying:** keep the full-mix crop stem as the join key (`"Artist - Title_<n>"`),
   OR move to whole-track keys + a (track, offset) crop index — decide in §8.
7. **Sourcing for stems:** prefer the original separated-stem audio if remountable;
   otherwise decode the existing stem VAE latents → audio → extract (lossy but
   transient-preserving; validate onset detection on a few clips first).
8. **Don't break the existing DB.** Either add new fields/keys to `timeseries.db`
   or write a sidecar store; the 2.7 GB DB is shared with the MIR project.

## 6. What LatCH needs per entry (consumer contract)

For a head on feature `F`, the dataset yields `(latent[64,256], target[C,256])`:
- `target` = `F`'s per-frame series for the crop window, channel-first.
- 1-D features → `C=1`; `hpcp` → `C=12`.
- The **input latent is always the FULL-MIX latent** (that's what's denoised at
  inference). A *drum* head therefore maps full-mix latent → drum-stem target. The
  stem only defines the *target*, never the input.
- Consumer applies clamp / smoothing / standardization; producer just supplies the
  raw aligned series.

## 7. Consumer-side support already built (2026-05-26)

`scripts/latch_dataset.py` + `scripts/train_latch.py` already handle:
- `smooth_kind ∈ {none, gaussian, linear, lowpass, beat_weighted}`, `smooth_width`
  (frames) — smooths a marker series into a continuous density envelope; routes the
  feature to a regression loss (SmoothL1) and `constant` target kind.
  `beat_weighted` scales onsets by beat-grid proximity (on-beat emphasis).
- `subset_frac` / `subset_seed` — deterministic fraction for quick bake-offs.
- Checkpoint metadata stores `smooth_kind/width`, `standardized`, `std_mean/std`,
  slider bounds, `noise_schedule`, `loss_type`.
- Bake-off config: `latch_train_density.yaml` (4 ep / 30% / `--standardize` so val
  losses are comparable across smoothing kinds → pick lowest val_median).

So once `onsets_activations_drums_ts` (etc.) lands, training is just:
`--feature onsets_activations_drums --smooth-kind <winner> --smooth-width 3`.

## 8. Open decisions

- **Storage:** sidecar store vs new fields in shared `timeseries.db`.
- **Keying:** keep per-crop keys vs whole-track + offset index (the latter is the
  natural fit for "continuous samples from full tracks").
- **Stem audio source:** remount original stems vs decode stem latents.
- **Stem coverage:** drums-only first vs drums+bass+other vs all four.
- **Frame rate storage:** store exactly at 21.533 Hz vs a higher rate + resample.

## 9. What we can train on *today* (no MIR changes)

The existing full-mix series are usable now — see the project memory and
`scripts/LATCH_TUNING_NOTES.md`. Notably the **full-mix `onsets_activations_ts`**
supports the smoothing bake-off immediately (it's "overall rhythmic activity," not
drum-specific, but validates the continuous-density approach end-to-end while the
per-stem pipeline is built).
