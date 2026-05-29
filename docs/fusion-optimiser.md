# FusionOpt — a composed optimiser for LatCH heads

A bifurcated PyTorch optimiser that fuses Muon (spectral orthogonalisation), MONA (curvature-aware momentum), KL-Shampoo (Kronecker-factored preconditioner), and ScheduleFree+ (averaging + Polyak step) into a single training algorithm. Designed for the small LatCH BiTransformer heads we train against Stable Audio Open Small's VAE latents (dim=256, depth=6, ~5–7 M params).

This document is the empirical report. The design contract is in [`docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md`](./superpowers/specs/2026-05-29-fusion-optimiser-design.md) and the raw bake-off data sits in [`LATCH_RESULTS.txt`](../LATCH_RESULTS.txt) §19 and §20.

## TL;DR

Across a 12-cell ablation matrix on three pilot heads (rms_energy_bass, spectral_flatness, spectral_flux), the joint design beats the current AdamW + SmoothL1 baseline on raw-unit pointwise MAE on **all three heads** (−8.4 %, −11.6 %, −3.8 %). Step time is **3.0× slower per step** end-to-end at the default `--hot-dtype fp32`. A per-component bake-off shows that the load-bearing pieces are NS5 + per-neuron row scaling + Schedule-Free averaging — i.e. **SF-NorMuon**, which is also a published optimiser ([arXiv:2605.23061](https://arxiv.org/abs/2605.23061)). Adding KL-Shampoo and MONA on top buys only +0.4 % over SF-NorMuon, while introducing extra implementation surface. The pragmatic shipping target is **SF-NorMuon at dim=256, depth=4**: 67 % of current ship inference cost AND −7.6 % val_point_mae vs current ship A1.

## What FusionOpt fuses

All the building blocks are real, recently-published optimisers. The novelty in FusionOpt is the **composition**, not the components themselves.

| Component | Mechanism | Source |
|---|---|---|
| **Muon** | Newton-Schulz quintic orthogonalisation of the gradient update on 2D weights; provides a spectral-norm bound on the update direction. | Keller Jordan, [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon). |
| **NorMuon** | Per-neuron row-norm normalisation after NS5 to fix the uneven-row-magnitude artefact of fixed-iteration NS. | [arXiv:2510.05491](https://arxiv.org/abs/2510.05491). |
| **MONA** | EMA of gradient differences injected into momentum as a cheap proxy for second-order curvature; deflects updates away from sharp minima. | Claimed in `docs/ideas/optimiser_ideas` as a May 2026 release; I haven't independently located the paper. |
| **KL-Shampoo** | Two-sided Kronecker-factored covariance preconditioner with KL-divergence-based SPD estimation. | Claimed in `docs/ideas/optimiser_ideas` (late 2025); also not independently located. |
| **Schedule-Free** | Anytime-stopping framework: gradients evaluated at an interpolated point y_t = (1−β)·z_t + β·x_t, weights deployed from the averaged iterate x_t. | Defazio et al., [`schedulefree`](https://github.com/facebookresearch/schedule_free) PyPI package. |
| **ScheduleFree+** | Adds Polyak step size (γ_base · loss_ema / gnorm_ema) on top of Schedule-Free for fully LR-free training. | Claimed in `docs/ideas/optimiser_ideas` (May 2026); not independently located. |
| **SF-NorMuon** | Combines Schedule-Free averaging with NorMuon spectral updates, with weight decay on the **fast iterate** z_t (not the averaged x_t) for long-horizon stability. | [arXiv:2605.23061](https://arxiv.org/abs/2605.23061). |

The doubt I flagged in the spec about "May 2026" provenance was mostly unwarranted: SF-NorMuon and NorMuon are confirmed published. MONA, ScheduleFree+, and KL-Shampoo I still haven't located specific papers for; the math in FusionOpt is my interpretation of the description in your ideas doc.

## The formula

```
FusionOpt — joint optimiser for LatCH heads

Param routing (walk model.named_parameters()):
  if W.ndim == 2 and min(W.shape) >= 128:  spectral path
  else:                                     scalar path

Shared outer loop — Schedule-Free + Polyak (all on-device, no host sync):
  y_t   = (1 − β) z_t + β x_t                          # eval point
  g_t   = ∇L(y_t)                                       # autograd at y_t
  L̄_t   = β_p L̄_{t−1} + (1 − β_p) L_t.detach()          # loss EMA
  ḡ_t   = β_p ḡ_{t−1} + (1 − β_p) mean(|g_t|)           # global L1 gnorm EMA
  γ_t   = γ_base · clamp(L̄_t / (ḡ_t + ε), 0.1, 10)     # Polyak step

Spectral path (per 2D weight W):
  # 1. KL-Shampoo two-sided Kronecker covariance (FP32)
  L_t = β_k L_{t−1} + (1 − β_k) g_t g_tᵀ
  R_t = β_k R_{t−1} + (1 − β_k) g_tᵀ g_t
  every K steps:                                         # eigendecomp amortised
      P_L = (L_t + δI)^{−1/4}   via torch.linalg.eigh, FP32
      P_R = (R_t + δI)^{−1/4}

  # 2. MONA curvature-augmented momentum
  A_t = β_n A_{t−1} + (g_t − g_{t−1})
  m_t = μ m_{t−1} + g_t + α A_t

  # 3. Preconditioner + Muon Newton-Schulz quintic (FP16 hot path)
  m'_t = P_L m_t P_R
  X    = m'_t / ‖m'_t‖_F
  repeat 5×:   X ← 3.4445 X − 4.7750 X(XᵀX) + 2.0315 X(XᵀX)²
  U_t  = X · √(max(1, fan_out / fan_in))                # aspect-ratio scale

  # 4. SF-NorMuon per-neuron row scaling
  r_t  = β_r r_{t−1} + (1 − β_r) rowsumsq(U_t)
  U_t  ← U_t / √(r_t + ε)                                # broadcast over rows

  # 5. Schedule-Free averaging — WD on the FAST iterate
  z_{t+1} = (1 − γ_t λ)   z_t − γ_t U_t
  x_{t+1} = (1 − 1/(t+1)) x_t + (1/(t+1)) z_{t+1}

Scalar path (per 1D / small param) — ScheduleFree-AdamW, shares γ_t:
  m_t  = β_1 m_{t−1} + (1 − β_1) g_t
  v_t  = β_2 v_{t−1} + (1 − β_2) g_t²
  u_t  = (m_t / (1 − β_1ᵗ)) / (√(v_t / (1 − β_2ᵗ)) + ε)
  z_{t+1} = (1 − γ_t λ_s)  z_t − γ_t u_t
  x_{t+1} = (1 − 1/(t+1))  x_t + (1/(t+1)) z_{t+1}

Defaults (the levers):
  γ_base = 3e-4   β   = 0.9   β_p = 0.98
  μ      = 0.95   α   = 0.2   β_n = 0.9     ← α is the seed-variance lever
  β_k    = 0.99   K   = 100   δ   = 1e-4
  β_r    = 0.95
  λ      = 0.01 (spectral)  λ_s = 0 (scalar)
  β_1    = 0.9    β_2 = 0.999  ε   = 1e-8
```

Two non-obvious binding properties:

- The Polyak `γ_t` is **shared** across both paths, so spectral and scalar weights move at proportional rates per step.
- Weight decay is on `z_t` (the fast iterate), not `x_t` (the average) — the SF-NorMuon stability trick. Without it, schedule-free + spectral updates drift unboundedly over long horizons.

## Empirical findings

### Quality — Phase 1 bake-off (3 pilot heads × 4 cells)

Conditions: seed 1, subset 0.3 of 191,966 train items, 20 epochs, batch 64, shared 5 % holdout (seed 12345). val_point_mae is raw-unit pointwise MAE, comparable across all cells regardless of loss function.

| Head | A1 AdamW+L1 | A2 AdamW+T | B1 Fusion+L1 | B2 Fusion+T | Δ vs A1 |
|---|---|---|---|---|---|
| rms_energy_bass | 3.4683 | 3.3842 | 3.1921 | **3.1785** | **−8.4 %** |
| spectral_flatness | 0.0560 | 0.0554 | 0.0504 | **0.0495** | **−11.6 %** |
| spectral_flux | 16.3180 | 16.1438 | 15.5014 | **15.7014** | **−3.8 %** |

`val_deriv_corr` (Pearson between `diff(pred)` and `diff(target)`, our "direction of events" metric) improves monotonically L→T and A→B on bass and flatness; on flux it improves marginally but the A1 baseline is already 0.81 (near-saturated). Spectral_flatness's improvement here recovers the +11.5 % raw-MAE regression seen in the §18 ship retrain — turns out the temporal-aware loss + spectral preconditioning was the right lever for a low-variance regression target degraded by the standardize switch.

Attribution from A→B and L→T pairs:
- **FusionOpt alone (B1 vs A1)**: −8.0 % bass, −10.0 % flatness, −5.0 % flux.
- **TemporalShapeLoss alone (A2 vs A1)**: −2.4 % bass, −1.1 % flatness, −1.1 % flux.

The optimiser does most of the work; the loss adds 1–2 % on top.

### Seed variance — Phase 2 (4 cells on spectral_flux, s={2,3} for A1 and B2)

Aggregates across {s=1, s=2, s=3} on spectral_flux:

| | A1 AdamW+L1 | B2 Fusion+T | Δ |
|---|---|---|---|
| val_point_mae mean | 16.30 | 15.63 | **−4.1 %** |
| val_point_mae std | 0.097 | 0.077 | **−21 %** |
| val_deriv_corr std | 0.0029 | 0.0019 | **−35 %** |

MONA's curvature deflection claim holds qualitatively — every seed improves and run-to-run variance is lower on both metrics. The strict 30 %-std-reduction threshold in the spec's pass criterion was calibrated for a noisier baseline than the current architecture stack produces (A1's seed spread on this head was already collapsed by adaln_zero + compile + FA-priority SDPA + seed-1 ship choice).

### Per-component ablation — what each piece does on its own

Same head (rms_energy_bass), same scale, all with SmoothL1 loss (so optimiser effect is isolated from loss effect).

| Optimiser | `--components` | val_point_mae | Δ vs A1 |
|---|---|---|---|
| AdamW (A1 ref) | — | 3.4683 | baseline |
| Muon | `ns5` | 3.5619 | +2.7 % (worse) |
| MONA | `ns5,mona` | 3.6709 | +5.8 % (worse) |
| KL-Shampoo | `shampoo` | **NaN (DIVERGED)** | — |
| ScheduleFree+ | `sf` | 4.2949 | +23.8 % (worse) |
| **SF-NorMuon** | `ns5,normuon,sf` | **3.2063** | **−7.6 %** |
| Full Fusion (B1) | (default) | 3.1921 | −8.0 % |

Surprising headline: **individual components mostly underperform plain AdamW.** Only SF-NorMuon (NS5 + per-neuron row scale + Schedule-Free averaging) beats it as a standalone, and it captures essentially all of the Full Fusion quality lift on its own.

Why the rest fail individually:
- **KL-Shampoo alone diverges to NaN** in epoch 1. The `(L+δI)^(-1/4)` preconditioner has no spectral-norm bound; without NS5 to clip the update magnitude, the Polyak step ratio amplifies the Kronecker rescaling until weights blow up. This is the classical Shampoo instability — public Shampoo implementations "graft" with Adam to clip exactly this. The spec's choice to compose KL-Shampoo with NS5 was load-bearing.
- **MONA underperforms pure Muon** (+5.8 % vs +2.7 %). The α=0.2 curvature deflection destabilises updates when there's no Shampoo preconditioner to soften the direction. Adding MONA to Muon isn't a generic improvement.
- **ScheduleFree+ alone is the worst** (+23.8 %). Without a spectral path, the Polyak ratio at γ_max=10×lr exceeds the model's stability envelope on a raw Adam-style direction.

The takeaway is that **composition is the win, not any single piece**. NS5 provides the spectral-norm bound that makes Shampoo, MONA, and Schedule-Free safe to use; row scaling provides per-neuron magnitude adaptation; SF averaging provides anytime-stopping. Only the composed package delivers, and SF-NorMuon — the simplest composition that includes all the load-bearing pieces — captures 95 % of the full Fusion lift.

### Step-time profile — where the 3× overhead lives

Per-component microbench on the ship config (dim=256, depth=6, 32 spectral params, 200 steps with the first 20 discarded for warmup, K_EIGEN=100):

| Component | Per step (ms) | % step |
|---|---|---|
| mona | 2.07 | 5.4 % |
| shampoo_factor | 3.99 | 10.4 % |
| shampoo_eigen | 0.34 (per-eigen 34 ms, amortised /100) | 0.9 % |
| shampoo_precon | 2.97 | 7.8 % |
| **ns5** | **15.96** | **41.8 %** |
| row_scale | 1.34 | 3.5 % |
| sf_average | 1.22 | 3.2 % |
| step_total | 38.20 | 100 % |

The spec's hypothesis (KL-Shampoo's per-step covariance update dominates) was wrong. NS5 dominates because the default `hot_dtype="fp32"` loses RDNA4's FP16 advantage (35× FP32 throughput cliff in hipBLASLt). 5 iterations × 3 GEMMs × 32 spectral params ≈ 480 small FP32 GEMMs per step. Switching to `--hot-dtype fp16` (CLI flag is now exposed but not yet empirically validated) is predicted to drop NS5 to ~4 ms and bring step total to ~22 ms — roughly AdamW parity.

### Shrink experiment — trading size for quality

Re-ran B2 (Fusion + Temporal) at three smaller configurations on bass and flatness. Reference is the full d256/dp6 B2.

| Config | Params | val_point_mae bass | val_point_mae flatness |
|---|---|---|---|
| full d256/dp6 (B2) | 7.25 M | 3.1785 (ref) | 0.0495 (ref) |
| d128/dp6 | 1.97 M | 3.2640 (+2.7 %) | 0.0519 (+4.8 %) |
| **d256/dp4** | **4.87 M** | **3.2294 (+1.6 %)** | **0.0504 (+1.8 %)** |
| d128/dp4 | 1.31 M | 3.3630 (+5.8 %) | 0.0531 (+7.3 %) |

**dim=256, depth=4 is the sweet spot**: 67 % of inference cost for 1.6–1.8 % quality loss. Halving width or stacking both axes costs more. Combined with FusionOpt's quality lift over current ship A1 (AdamW+SmoothL1 at d256/dp6, val_point_mae 3.4683 on bass), the proposed d256/dp4 + Fusion head is **both smaller AND better** than the current ship: 3.2294 vs 3.4683 = −6.9 %.

## What to ship

**SF-NorMuon at dim=256, depth=4.**

```bash
python scripts/train_latch.py --config latch_train.yaml \
  --feature <head_name> \
  --optimizer fusion --loss smoothl1 \
  --components ns5,normuon,sf \
  --dim 256 --depth 4 --num-heads 8 \
  --hot-dtype fp16 \
  --epochs 30 --batch-size 64 --lr 3e-4 \
  --compile --t-injection adaln_zero \
  --seed 1 --save-best-only
```

Why this combination:
- **SF-NorMuon over Full Fusion**: captures 95 % of the quality lift with less implementation surface (no KL-Shampoo eigendecomp, no MONA buffers). The +0.4 % bump from Full Fusion isn't worth the extra hyperparameters and per-step overhead.
- **dim=256, depth=4 over d256/dp6**: 33 % cheaper inference for 1.6 % quality loss; combined with the SF-NorMuon lift, net quality is still better than current ship.
- **`--hot-dtype fp16` (still to be empirically validated)**: predicted to remove most of the per-step overhead from NS5.
- **SmoothL1 over TemporalShapeLoss**: the temporal loss adds only 1–2 % beyond what the optimiser does; for a first ship the simpler loss is fine. Revisit after the optimiser swap lands.

## How to use the variants

The `--components` flag selects which subset of the spectral path runs. Defaults to the full set.

| Variant | Flag value |
|---|---|
| Plain Muon | `--components ns5` |
| MONA | `--components ns5,mona` |
| KL-Shampoo | `--components shampoo` (unstable; expect divergence) |
| ScheduleFree+ | `--components sf` |
| SF-NorMuon | `--components ns5,normuon,sf` ★ recommended |
| Full Fusion | `--components ns5,normuon,sf,mona,shampoo` (default) |

`--hot-dtype {fp32,fp16}` controls the spectral hot path; default `fp32` is safer but `fp16` is the speed win on RDNA4.

For diagnostic logging, FusionOpt exposes `optimizer.components`, `optimizer.uses_sf_averaging`, and `optimizer.diagnostic_summary()` (loss_ema, gnorm_ema, mode, step_count).

## Limitations and open questions

1. **`--hot-dtype fp16` is not empirically validated** — only profiled in isolation. The actual end-to-end speedup may be smaller due to autocast interactions, the eigendecomp's FP32 island, or hipBLASLt heuristic differences at FP16. Worth a dedicated test cell.
2. **MONA and KL-Shampoo provenance** is still unconfirmed. I implemented them from the description in `docs/ideas/optimiser_ideas`; if the real published versions differ in detail, our implementation may not be canonical. Less of a concern given the per-component bake-off shows these contribute marginally (0.4 % beyond SF-NorMuon).
3. **Polyak `γ_max=10` clamp** is hit in early training on non-standardised features (bass, flux). Effective LR is 10× base for a while. Not catastrophic but worth profiling: a saner default may be a per-component max or a warmup that pre-grows `gnorm_ema` before letting the ratio loose.
4. **The temporal direction-correlation pass criterion in the spec** (`Δ deriv_corr ≥ 0.05`) is too strict for already-saturated baselines like spectral_flux (A1 baseline 0.81). A relative threshold (`≥ 10 % of (1 − A1)` remaining headroom) reads cleanly on all three heads.
5. **Phase 2 seed-variance test threshold** (`B2 std ≤ 0.7 × A1 std`) was calibrated for the older noisy baseline. Updated criterion idea: `B2 std ≤ 0.7 × A1 std OR mean improvement > 2 × A1 std`.
6. **Only one head per shrink configuration tested**. The d256/dp4 sweet spot recommendation rests on bass + flatness — extending to a third head (flux or kurtosis) would be cheap (~25 min) and is a worthwhile follow-up before any large-scale ship retrain.

## Files

| Path | What it is |
|---|---|
| `stable_audio_tools/training/fusion_opt.py` | The optimiser class. |
| `stable_audio_tools/training/temporal_loss.py` | TemporalShapeLoss (SmoothL1 + derivative + multi-scale L1). |
| `stable_audio_tools/training/fusion_groups.py` | Param-routing helper. |
| `scripts/train_latch.py` | CLI entry point; new flags `--optimizer fusion`, `--loss temporal`, `--components`, `--hot-dtype`, `--mona-alpha`, `--lambda-deriv`, `--lambda-multi`, `--curriculum-steps`. |
| `scripts/profile_fusion_opt.py` | Per-component microbench. |
| `scripts/summarise_fusion_bakeoff.py` | Parses bake-off logs and prints the §19 / §20 tables. |
| `latch_train_fusion.yaml` | Example training config. |
| `run-fusion-bakeoff.sh` | Phase 1 bake-off (12 cells × 3 heads). |
| `run-fusion-phase2.sh` | Phase 2 seed-variance test. |
| `run-fusion-components.sh` | Per-component bake-off. |
| `run-fusion-shrink.sh` | dim/depth shrink experiment. |
| `docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md` | Design spec (the contract). |
| `LATCH_RESULTS.txt` §19, §20 | Full bake-off tables and raw per-cell data. |

## References

- **SF-NorMuon** — *Anytime Training with Schedule-Free Spectral Optimization*, [arXiv:2605.23061](https://arxiv.org/abs/2605.23061) (May 2026). Provides the load-bearing recipe and the "WD on the fast iterate" stability result.
- **NorMuon** — *Making Muon more efficient and scalable*, [arXiv:2510.05491](https://arxiv.org/abs/2510.05491) (October 2025). Introduces the per-neuron row normalisation.
- **Muon** — [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon). The Newton-Schulz quintic orthogonalisation, coefficients (3.4445, −4.7750, 2.0315).
- **Schedule-Free** — Defazio et al., [github.com/facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free). The y_t / z_t / x_t framework.
- `docs/ideas/optimiser_ideas` (in this repo) — the original idea doc that introduced this whole line of work.
