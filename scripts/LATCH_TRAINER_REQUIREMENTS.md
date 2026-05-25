# LatCH Trainer — Requirements (EARS)

Initial requirements for the **LatCH head training pipeline** (`train_latch.py`,
`latch_dataset.py`, `latch_model.py`). Scope is training and checkpoint
production only; the inference/TFG guidance path is specified separately.

Requirements use the EARS keywords: **ubiquitous** ("shall"), **event-driven**
("When"), **state-driven** ("While"), **unwanted-behavior** ("If … then"), and
**optional** ("Where"). IDs are stable handles for traceability.

---

## 1. Invocation & configuration

- **TR-1** The trainer shall accept a target feature name, number of epochs,
  batch size, learning rate, latent directory, timeseries DB path, save
  directory, and forward-noising objective as command-line arguments.
- **TR-2** Where an argument is omitted, the trainer shall apply its documented
  default (feature `rms_energy_bass`, 10 epochs, batch size 8, lr 1e-4,
  objective `rectified_flow`).
- **TR-3** The trainer shall accept the target feature name in its bare form
  (without the `_ts` suffix) and shall append `_ts` internally for TimeseriesDB
  lookups.
- **TR-4** If the supplied objective is not one of `rectified_flow`,
  `rf_denoiser`, or `v`, then the trainer shall reject the run with an error
  before training begins.

## 2. Environment setup

- **TR-5** The trainer shall apply the ROCm "training" profile from
  `rocm_env.yaml` before the first `import torch`.
- **TR-6** The trainer shall load the ROCm environment module standalone (via
  importlib) so that the `stable_audio_tools` package `__init__` (which imports
  torch and applies the inference profile) does not run first.
- **TR-7** Where a CUDA/ROCm device is available, the trainer shall place the
  model and batches on that device; otherwise it shall fall back to CPU.

## 3. Dataset construction & target resolution

- **TR-8** The dataset shall discover training items by scanning the latent
  directory recursively for per-track `.npy` files.
- **TR-9** The dataset shall exclude stem latents (filenames ending in `_bass`,
  `_drums`, `_other`, `_vocals`) from the training set.
- **TR-10** When the target feature is a time-series feature, the dataset shall
  read the target array from the TimeseriesDB using the latent filename stem as
  the lookup key.
- **TR-11** Where the target feature is a scalar feature, the dataset shall read
  it from the companion `.INFO`/`.json` file and broadcast it across all 256
  frames.
- **TR-12** The dataset shall normalise every target array to shape
  `(channels, max_frames)`, treating the smaller dimension as channels (so
  `hpcp` resolves to 12 channels).
- **TR-13** The dataset shall pad or trim each latent and each target along the
  time axis to exactly `max_frames` (default 256) frames.
- **TR-14** If a requested time-series feature is needed but the TimeseriesDB
  module cannot be imported, then the dataset shall raise an error at
  construction time rather than at first batch.
- **TR-15** When an individual item fails to load, the dataset shall skip to the
  next item; if every item fails, then it shall raise an error reporting that no
  valid items exist.
- **TR-16** When constructed, the dataset shall emit a diagnostic warning if the
  first item's time-series feature is absent from the DB while its scalar
  counterpart exists in `.INFO` (signalling that time-series extraction has not
  been run).

## 4. Model construction

- **TR-17** The trainer shall infer the model's output-channel count from the
  channel dimension of a sampled target (1 for scalar/1-D series, 12 for
  `hpcp`).
- **TR-18** The trainer shall instantiate the LatCH head with 64 input channels
  matching the VAE latent dimensionality.
- **TR-19** The head shall accept a noisy latent `[B, 64, T]` and a per-sample
  timestep `[B]` and shall return a control prediction `[B, out_channels, T]`.
- **TR-20** The head shall concatenate the timestep embedding as a prepended
  sequence token and shall strip that token before the final output projection.

## 5. Forward-noising schedule (LatCH-F)

- **TR-21** The trainer shall forward-noise each clean latent using the schedule
  selected by the `--objective` argument.
- **TR-22** Where the objective is `rectified_flow`/`rf_denoiser`, the trainer
  shall noise linearly as `z_t = (1−t)·z0 + t·noise`.
- **TR-23** Where the objective is `v`, the trainer shall noise with the VP
  schedule `z_t = cos(π/2·t)·z0 + sin(π/2·t)·noise`.
- **TR-24** The trainer shall sample timesteps `t` uniformly on `[0, 1]` per
  sample per step, where `t=0` is clean and `t=1` is pure noise.
- **TR-25** The trainer shall record the chosen objective in the checkpoint as
  the head's `noise_schedule`, because a head must be guided only by a diffusion
  model whose objective matches its forward-noising schedule.

## 6. Loss selection

- **TR-26** When the target feature is `hpcp`/`chroma`, the trainer shall use a
  cosine loss and shall record `loss_type = "cosine"`.
- **TR-27** When the target feature is a beat/onset activation, the trainer shall
  use a sparse-weighted BCE-with-logits loss and shall record
  `loss_type = "bce_logits"`.
- **TR-28** Where no feature-specific rule applies, the trainer shall use MSE and
  shall record `loss_type = "mse"`.
- **TR-29** The recorded `loss_type` shall match the criterion actually used at
  training time so that inference-time guidance can apply the same loss.

## 7. Training loop

- **TR-30** While training, the trainer shall iterate the dataset for the
  configured number of epochs in shuffled, drop-last batches.
- **TR-31** While training, the trainer shall use mixed-precision autocast with
  gradient scaling.
- **TR-32** While training, the trainer shall display per-batch loss and report
  the average loss at the end of each epoch.

## 8. Checkpointing

- **TR-33** When an epoch completes, the trainer shall save a v2 checkpoint
  containing `state_dict`, `feature_name`, `feature_stats`,
  `target_kind_default`, `noise_schedule`, and `loss_type`.
- **TR-34** Before training, the trainer shall estimate `feature_stats` (mean,
  std, min, max, sample count) from a bounded random sample of the dataset so the
  inference UI can auto-range its target sliders.
- **TR-35** The trainer shall derive a default target kind from the feature name
  (`beat_grid` for beat/onset/downbeat, `ramp_up` for rms/energy, otherwise
  `constant`).
- **TR-36** The trainer shall write each checkpoint to the save directory named
  by feature and epoch (`latch_<feature>_ep<N>.pt`), creating the directory if
  absent.

## 9. Cross-version consistency (framework note)

- **TR-37** Where the same trainer or shared framework code targets more than one
  Stable Audio model generation, the forward-noising schedule, latent
  dimensionality, and frame count shall be resolved per target model rather than
  assumed, because sampler and hyperparameter settings differ across versions.

---

## Open questions / assumptions to confirm

1. **Latent/target frame rate** is fixed at 256 frames (~21.53 Hz) for the Small
   model. Should this be derived from the target model's config rather than
   hard-coded?
2. **Validation/early-stopping** — there is currently no held-out split or
   stopping criterion; every epoch is saved. Is that intended, or should a
   best-checkpoint policy be a requirement?
3. **Reproducibility** — no seed control is currently specified. Should runs be
   seedable?
4. **Legacy checkpoints** — `LATCH_README.md` mentions a `retrofit_latch_stats.py`
   migration; should backward-compatible loading of bare-state-dict files be an
   explicit trainer/inference requirement?
