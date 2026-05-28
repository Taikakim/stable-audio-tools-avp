# FusionOpt — joint optimiser + temporal-aware loss for LatCH heads

**Status:** Approved (2026-05-29). User delegated approval and autonomous implementation.
**Author:** Kim (architect/artist), drafted with Claude Code.
**Branch:** `latch-rms-control`
**Target model:** LatCH BiTransformer heads on Stable Audio Open Small (`dim=256`, `depth=6`, `heads=8`, `t_injection=adaln_zero`, 4.9 M params)
**Target hardware:** AMD RX 9070 XT (RDNA4, gfx1201, 16 GiB), ROCm 7.2.x, hipBLASLt build `dabb6df2b98`, PyTorch 2.10 (SA3 cache).

## Summary

LatCH heads currently train with AdamW + SmoothL1. AdamW is an element-wise optimiser that ignores the 2D structure of weight matrices; SmoothL1 is a point-wise loss that ignores the temporal structure of feature curves. Both miss large optimisation signals for the small (~5 M-param) heads we're training.

This spec introduces:

1. **FusionOpt** — a single PyTorch `Optimizer` that fuses the best features of Muon (spectral orthogonalisation), MONA (curvature-aware Nesterov injection), KL-Shampoo (Kronecker-factored covariance preconditioner) and ScheduleFree+ (Polyak step size + iterant averaging + WD on the fast iterate). Bifurcated routing: 2D matrices aligned to LatCH's 256-grid get the spectral path; small/odd matrices and 1D params get a ScheduleFree-AdamW scalar path. Both paths share one Schedule-Free outer loop, so the model can be stopped at any epoch and the averaged iterate is a valid deployable model.

2. **TemporalShapeLoss** — a three-component loss that augments the existing point-wise SmoothL1 with a derivative term (penalises wrong direction of events) and a multi-scale L1 term over octave-spaced temporal downsamples (penalises wrong shape of superstructures: phrases, bars, sections).

Goals: better quality at the same compute budget, reduced seed variance on small subsets, anytime stopping flexibility, all within the RDNA4 fast-tile envelope. Cost tolerance is ~4× AdamW state and ~10–20 % wall-clock slower per step (still net-faster to convergence in expectation).

## Goals

1. Beat AdamW + SmoothL1 on `val_point_mae` (raw, unit-comparable across loss functions) on all three pilot heads (`rms_energy_bass`, `spectral_flux`, `spectral_flatness`) at fixed seed.
2. Reduce seed-spread (σ across 3 seeds) on `spectral_flux` by ≥ 30 % vs. the current AdamW baseline.
3. Capture temporal direction: `val_deriv_corr` (Pearson between `diff(pred)` and `diff(target)`) ≥ 0.05 higher than AdamW + SmoothL1 on rms_energy_bass and spectral_flux.
4. Stay on the RX 9070 XT FP16 fast paths: every spectral-path matmul on the 256-grid; no host syncs in the optimiser step.
5. Backwards-compatible: existing trained heads in `latch_weights/` and existing YAML configs train unchanged unless `--optimizer fusion` is set.

## Non-goals

- Designing a genuinely novel optimisation algorithm (we explicitly stand on Muon + Schedule-Free; KL-Shampoo and MONA implementations follow the math claimed in `docs/ideas/optimiser_ideas`).
- Generalising to non-LatCH workloads (Stable Audio diffusion backbone, autoencoder, ARC). FusionOpt is scoped to the BiTransformer heads.
- Touching inference code (Gradio UI, `run_gradio.py`, `unwrap_model.py`) beyond an auto-detect on the checkpoint format.
- Validating on the 12-channel `hpcp` feature in Phase 1. Cosine-loss vs. per-channel diff attribution is its own ablation; deferred to a follow-up.
- Replacing the depth=6 / dim=256 architecture choice. That decision is settled.

## 1. Architecture overview

Three new modules and one CLI hook:

```
stable_audio_tools/training/
├── fusion_opt.py            # FusionOpt(torch.optim.Optimizer)
├── temporal_loss.py         # TemporalShapeLoss(nn.Module)
└── fusion_groups.py         # build_fusion_param_groups(model)
scripts/train_latch.py       # +CLI: --optimizer fusion --loss temporal --mona-alpha ...
```

Per-step data flow (single mini-batch):

1. **Eval point**: Schedule-Free interpolates `y_t = (1-β)·z_t + β·x_t`. Forward runs through the model's parameters held in `z_t` (the SF wrapper swaps tensors via `optimizer.train()` / `optimizer.eval()`).
2. **Loss**: `TemporalShapeLoss(pred, target) = L_point + λ_d·L_deriv + λ_m·L_multi` — all three components on the same `(B, F, T)` tensor.
3. **Backward**: autograd produces gradients on `z_t`.
4. **`FusionOpt.step()`** routes per param group:
   - **Spectral group** (256-grid 2D matrices): MONA-augmented momentum → KL-Shampoo preconditioner transform → Muon NS5 → per-neuron row-norm scale → WD on `z_t` → update `z_t` → update averaged `x_t`.
   - **Scalar group** (latent_proj, out_proj, 1D params): ScheduleFree-AdamW on `z_t` → update averaged `x_t`.
5. Polyak step size `γ_t` is computed from on-device EMAs every step — no `.item()` calls in the hot loop.

Train/eval semantics inherited from Schedule-Free, applied across both groups:

- `optimizer.train()` — live `z_t` weights, model trains on them.
- `optimizer.eval()` — averaged `x_t` weights, used for validation and checkpoint serialisation.

`train_latch.py` already calls `.train()` / `.eval()` around the val loop for the `schedulefree` optimiser path; we reuse that contract.

Checkpoint format: saves both `z_t` (live state) and `x_t` (averaged state) state dicts; `--save-best-only` selects the averaged `x_t`, which is the deployable model. This is the SF/SF-NorMuon contract: anytime stopping yields a valid model from `x_t`, not `z_t`.

## 2. Parameter-group routing

A small helper walks the LatCH model once and assigns every trainable parameter to **spectral** or **scalar**. Threshold: matrix-shaped with both dimensions ≥ 128.

```python
def build_fusion_param_groups(model, force_scalar=()):
    spectral, scalar = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(re.search(pat, name) for pat in force_scalar):
            scalar.append(p); continue
        if p.ndim == 2 and min(p.shape) >= 128:
            spectral.append(p)
        else:
            scalar.append(p)
    return [
        {"params": spectral, "group_type": "spectral"},
        {"params": scalar,   "group_type": "scalar"},
    ]
```

For LatCH's current ship config (`t_injection=adaln_zero`, depth=6, dim=256, heads=8):

| Group | Layers routed there | ~Params | Fast on RDNA4? |
|---|---|---|---|
| **Spectral** | `qkv` (768×256), `proj` (256×256), `mlp.fc1` (1024×256), `mlp.fc2` (256×1024), `adaLN_mod.1` (1536×256), `t_film` (512×256 if FiLM mode), time-embedder MLP layers | ~4.5 M (~91 %) | ✓ all 256-grid aligned |
| **Scalar** | `latent_proj` (256×64), `out_proj` (F×256, F ∈ {1, 12}), all biases, all LayerNorm γ/β | ~0.4 M (~9 %) | n/a — small/odd shapes, AdamW handles fine |

Threshold rationale: `min(shape) ≥ 128` is deliberately loose so a future half-dim projection (e.g. 256×128) would still route to the spectral path. Layers smaller than that — `latent_proj` (K=64) and `out_proj` (M=F=1–12) — are correctly excluded; they're off the 256-grid fast path *and* rank-deficient in one dimension, which is exactly where Muon explicitly says to skip NS5.

The `adaLN_mod.1` projection (256 → 6×256 = 1536) is a particularly nice case: 1536 is the SA3 d_model magic number, hitting the 135 TFLOPS fast tile on RDNA4.

## 3. Optimiser update rule

### Shared: Schedule-Free outer loop + Polyak step

```
y_t  = (1-β)·z_t + β·x_t                       # eval point (model forwards here)
g_t  = ∇L(y_t)                                 # gradients, from autograd

loss_ema_t = β_p·loss_ema_{t-1} + (1-β_p)·loss.detach()
gnorm_ema  = β_p·gnorm_ema      + (1-β_p)·mean(|g_t|)
γ_t        = γ_base · clamp(loss_ema_t / (gnorm_ema + ε), 0.1, 10)
```

`γ_t` is clamped to `[0.1·γ_base, 10·γ_base]` so the Polyak ratio can't blow up early when `loss_ema` is large but `gnorm` is tiny. All reductions are on-device; never call `.item()` in the hot loop.

### Spectral path (per 2D matrix W with parameters)

```
# 1. KL-Shampoo — two-sided Kronecker covariance from RAW gradient
#    (matches traditional Shampoo convention; covariance estimates the natural
#    gradient geometry, then is applied to the augmented momentum below)
L_t = β_k·L_{t-1} + (1-β_k)·(g_t @ g_t.T)      # output-side, (out, out)
R_t = β_k·R_{t-1} + (1-β_k)·(g_t.T @ g_t)      # input-side,  (in, in)
if t % K == 0:                                  # K=100 default
    P_L = (L_t + δ·I)^(-1/4)                    # FP32 eigendecomp
    P_R = (R_t + δ·I)^(-1/4)                    # FP32 eigendecomp

# 2. MONA — curvature-aware augmented momentum
A_t = β_n·A_{t-1} + (g_t - g_{t-1})            # EMA of gradient differences
m_t = μ·m_{t-1} + g_t + α·A_t                  # Nesterov-like injection

# 3. Apply KL-Shampoo preconditioner to augmented momentum
m'_t = P_L @ m_t @ P_R                          # preconditioned momentum (FP16)

# 3. Muon — spectral normalisation via Newton-Schulz quintic
M̄_t = m'_t / ‖m'_t‖_F                          # Frobenius-normalise → SVs in [0,1]
for _ in range(5):
    A   = M̄_t @ M̄_t.T
    M̄_t = 3.4445·M̄_t - 4.7750·A·M̄_t + 2.0315·A·A·M̄_t
U_t = M̄_t * sqrt(max(1, fan_out/fan_in))       # aspect-ratio scale

# 4. SF-NorMuon — per-neuron (row) step adaptation
r_t = β_r·r_{t-1} + (1-β_r)·rowsumsq(U_t)
U_t = U_t / sqrt(r_t.unsqueeze(-1) + ε)

# 5. Schedule-Free averaging (WD on the FAST iterate)
z_{t+1} = (1 - γ_t·λ)·z_t - γ_t·U_t
x_{t+1} = (1 - 1/(t+1))·x_t + (1/(t+1))·z_{t+1}
```

### Scalar path (per 1D / small parameter)

Standard ScheduleFree-AdamW with the same Polyak `γ_t`:

```
m_t = β_1·m_{t-1} + (1-β_1)·g_t
v_t = β_2·v_{t-1} + (1-β_2)·g_t²
u_t = m̂_t / (sqrt(v̂_t) + ε)                   # bias-corrected Adam direction
z_{t+1} = (1 - γ_t·λ_s)·z_t - γ_t·u_t
x_{t+1} = (1 - 1/(t+1))·x_t + (1/(t+1))·z_{t+1}
```

### NS5 quintic on RDNA4

5 iterations of `X ← 3.4445·X - 4.7750·X(XᵀX) + 2.0315·X(XᵀX)²`. Each iteration is 3 matmuls. For LatCH's largest spectral matrix (1024×256), that's three GEMMs of shape `(1024,256)·(256,1024)·(1024,256)` — all on the 256-grid. Whole NS5 call: ~15 GEMMs, all sub-millisecond at FP16 with FP32 accumulate. Trivial against the model forward pass.

### KL-Shampoo numerical stability

- **SPD ridge**: add `δ·I` to `L_t` and `R_t` before eigendecomposition (δ = 1e-4 default). Prevents zero or negative eigenvalues from small EMAs early in training.
- **FP32 island**: the eigendecomposition and `^(-1/4)` step run in FP32; only the resulting `P_L`, `P_R` matrices are downcast to FP16 for the preconditioner matmul. This avoids the 35× FP32 cliff in the hot path while keeping the eigen step well-conditioned.
- **Warm start**: `P_L`, `P_R` initialise as identity. They blend toward the true `^(-1/4)` over the first ~K·3 = 300 steps. Prevents early-training divergence from noisy covariance estimates.

### Hyperparameter defaults

| Name | Default | Sensitivity | Description |
|---|---:|---:|---|
| `γ_base` | `3e-4` | medium | Polyak base step (matches current AdamW LR) |
| `β` (SF interp) | `0.9` | low | how much eval point leans toward averaged x_t |
| `μ` (spectral momentum) | `0.95` | low | momentum for the spectral path |
| **`α` (MONA)** | **`0.2`** | **high** | curvature injection strength — the seed-variance lever |
| `β_n` | `0.9` | low | EMA of gradient differences for MONA |
| `β_k` (KL-Shampoo factor EMA) | `0.99` | medium | slower = more stable, less adaptive |
| `K` (eigen cadence) | `100` | low | steps between L, R eigendecompositions |
| `δ` (SPD ridge) | `1e-4` | low | numerical stability for eigendecomp |
| `β_r` (row-norm EMA) | `0.95` | low | SF-NorMuon per-neuron scaling |
| `λ` (spectral WD) | `0.01` | medium | weight decay on z_t for spectral params |
| `λ_s` (scalar WD) | `0.0` | low | weight decay on z_t for LayerNorm/biases |
| `β_1, β_2` (scalar) | `0.9, 0.999` | low | Adam moments for scalar group |
| `β_p` (Polyak EMAs) | `0.98` | low | loss and gnorm smoothing |

`α` is the one to tune in the bake-off. Higher `α` (~0.3) → more aggressive deflection from sharp minima (stronger seed-variance fix, possible stall in smooth valleys). Lower `α` (~0.1) → closer to plain Muon. Default `0.2` is a reasonable midpoint.

### Composition order (why this works)

1. KL-Shampoo estimates covariance from the **raw gradient** (traditional Shampoo convention); the preconditioner captures the natural-gradient geometry of the loss surface, independent of momentum dynamics.
2. MONA adds curvature info to momentum **before** preconditioning, so the update direction has both Nesterov-like deflection and Shampoo-like whitening.
3. KL-Shampoo preconditioner is applied to the augmented momentum, then Muon NS5 runs on the result. NS5 normalises by Frobenius norm of its input, so the preconditioner's magnitude scaling is absorbed cleanly.
4. SF-NorMuon's row-norm scaling runs **after** NS5, so it adapts per-neuron step magnitudes based on the orthogonalised update direction (not the raw momentum).
5. WD on the fast iterate `z_t` (not the averaged `x_t`) is the SF-NorMuon stability trick — required for spectral updates to remain bounded over long horizons.

## 4. TemporalShapeLoss

All three components operate on the same `(B, F, T)` tensor (`F=1` for scalar features, `F=12` for hpcp).

```python
class TemporalShapeLoss(nn.Module):
    def __init__(
        self,
        huber_beta: float = 1.0,
        lambda_deriv: float = 1.0,
        lambda_multi: float = 0.5,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
        point_loss: str = "auto",          # "auto" → cosine for hpcp, smooth_l1 otherwise
        curriculum_steps: int = 0,         # 0 = off; otherwise ramp λ_d, λ_m from 0 to default
    ):
        ...

    def forward(self, pred, target):
        L_point = self._point_loss(pred, target)
        L_deriv = self._derivative_loss(pred, target)
        L_multi = self._multiscale_loss(pred, target)
        scale = self._curriculum_scale()    # 1.0 if curriculum off, else linear ramp
        self.last_components = {"L_point": L_point.detach(),
                                "L_deriv": L_deriv.detach(),
                                "L_multi": L_multi.detach()}
        return L_point + scale * (self.lambda_deriv * L_deriv + self.lambda_multi * L_multi)
```

### Components

**L_point** — inherits the trainer's current behaviour: SmoothL1 for scalar features, cosine for hpcp (12-channel chroma). `huber_beta` is set from `feature_stats` exactly as today.

**L_deriv** — first-order temporal difference, then SmoothL1:

```
diff_pred   = pred[..., 1:]   - pred[..., :-1]      # (B, F, T-1)
diff_target = target[..., 1:] - target[..., :-1]
L_deriv     = SmoothL1(diff_pred, diff_target, beta=huber_beta)
```

For hpcp this is SmoothL1 on the diff vectors element-wise (not cosine on diffs — that would double-penalise across-channel direction shifts already handled by L_point).

**L_multi** — multi-scale L1 via `avg_pool1d`:

```
for s in scales:
    pred_s   = F.avg_pool1d(pred,   kernel_size=s, stride=s)   # (B, F, T/s)
    target_s = F.avg_pool1d(target, kernel_size=s, stride=s)
    L_s      = SmoothL1(pred_s, target_s, beta=huber_beta)
L_multi = mean over s
```

Scale grid `s ∈ {2, 4, 8, 16, 32, 64, 128, 256}` covers ~170 ms (1/16-note range at SAO small's VAE rate) up to the global mean (~22 s). Pure reductions, no matmuls — compute is dust on RDNA4. Equal weight per scale by default.

### Composition

```
L_total = L_point + scale · (λ_d · L_deriv + λ_m · L_multi)
```

Defaults: `λ_d = 1.0`, `λ_m = 0.5`. Rationale:

- `λ_d = 1.0` — derivative term gets equal weight to point loss; "direction of events" is a first-class objective.
- `λ_m = 0.5` — multi-scale supports rather than dominates; with 8 scales averaged it would otherwise overwhelm the other two terms in gradient magnitude.

### Validation impact

`val_total` is no longer comparable to the existing `val_median` numbers in `LATCH_RESULTS.txt` (different loss). We log all three components separately to WandB (`L_point`, `L_deriv`, `L_multi`) plus the raw-unit comparable `val_point_mae` (just SmoothL1 on pred/target, recomputed without scaling). AdamW baseline gets re-evaluated on the new loss before comparison.

## 5. Validation protocol

### Ablation matrix

A 2×2 across the two changes (optimiser + loss) on three pilot heads. The AdamW + SmoothL1 cells are **free** — already exist in `latch_ship_retrain_s1/`. So this is 9 new runs.

|  | **SmoothL1 (old loss)** | **TemporalShapeLoss (new loss)** |
|---|---|---|
| **AdamW (baseline)** | A1 — already have data | A2 — loss-only contribution |
| **FusionOpt** | B1 — optimiser-only contribution | B2 — joint (our pick) |

### Pilot heads (3, chosen to stress different dimensions)

1. **`rms_energy_bass`** — well-trained, low seed variance. Smooth temporal curve with strong directional structure (kicks). Tests: does the temporal loss + better optimiser push a "best case" head further?
2. **`spectral_flux`** — explicitly a derivative-like feature; previously high seed variance on 30 % subset. Tests: does MONA + derivative loss reduce seed variance on a flux-style target?
3. **`spectral_flatness`** — *mildly regressed* on ship retrain (+11.5 % raw-MAE). Tests: can the temporal loss + better geometry recover the regression *without* reverting the standardize change?

Each run: same `--seed 1`, `--epochs 30`, `--save-best-only`. The autonomous-night run uses `--subset-frac 0.3` and `--epochs 20` for tractability within the 6-hour budget; a full-subset rerun on winning conditions is a follow-up.

### Diagnostic metrics (logged per epoch to WandB)

| Metric | Why we log it |
|---|---|
| `val_total` | the composite loss being optimised |
| `val_point_mae` | raw-unit pointwise MAE — comparable across all four cells |
| `val_deriv_corr` | Pearson between `diff(pred)` and `diff(target)` — direct test of "direction of events" preservation |
| `val_multiscale_mae` | raw-unit MAE averaged across the 8 scales — superstructure test |
| `L_point`, `L_deriv`, `L_multi` | component breakdown so we can see which term drives gradient |

### Pass criteria

The joint design (B2) wins if **all three** hold:

1. **Quality**: `val_point_mae` at B2 ≤ `val_point_mae` at A1 (baseline) on all three pilot heads. No regressions on the unit-comparable metric.
2. **Temporal awareness**: `val_deriv_corr` at B2 ≥ `val_deriv_corr` at A1 by ≥ 0.05 on rms_energy_bass and spectral_flux. Confirms the temporal loss term isn't being ignored.
3. **Geometry contribution**: B2 ≤ A2 on `val_point_mae` for at least 2 of 3 heads. Confirms FusionOpt is doing work beyond what the new loss alone does.

### Phase 2 — seed-variance test (conditional)

Only triggered if Phase 1 looks promising. Three extra seeds on `spectral_flux` for A1 and B2 only (4 new runs total). Pass: B2 seed-std ≤ 0.7 × A1 seed-std. This is the MONA-curvature claim we explicitly want to validate.

### Out of scope for Phase 1

- `hpcp` feature (cosine-loss interactions deserve separate ablation).
- Depth > 6 architectures.
- Other `t_injection` modes (settled on `adaln_zero`).
- Generalising the bake-off to all 8 trained heads.

## 6. Integration with `train_latch.py`

### CLI surface

Three new flags, plus four hyperparameter knobs. All defaults keep existing behaviour untouched.

```
--optimizer fusion              # adds "fusion" to existing choices
--loss {smoothl1,temporal}      # new flag; default "smoothl1" for back-compat
--lambda-deriv 1.0              # TemporalShapeLoss derivative weight
--lambda-multi 0.5              # TemporalShapeLoss multi-scale weight
--mona-alpha 0.2                # the seed-variance lever
--curriculum-steps 0            # temporal-term warmup (0 = off)
--reset-optimizer               # drop existing optimizer state on resume
                                # (required when resuming FusionOpt from a legacy AdamW checkpoint)
```

Parse-time validation:

- `--optimizer fusion --scheduler cosine` → reject (Polyak is schedule-free; cosine doubles up).
- `--loss temporal` with `feature` being hpcp → allowed; cosine handles L_point, SmoothL1 handles L_deriv and L_multi.

### YAML config keys (mirror CLI 1:1)

The existing `pick(args.X, "X", default)` pattern handles this for free.

```yaml
optimizer: fusion
loss: temporal
lambda_deriv: 1.0
lambda_multi: 0.5
mona_alpha: 0.2
curriculum_steps: 0
```

Existing configs (`latch_train.yaml`, `latch_train_hpcp.yaml`, etc.) without these keys continue to use `adamw` + `smoothl1`.

### Wiring changes

Two existing blocks get extended; nothing else moves.

**Optimizer creation** (currently `train_latch.py` lines ~303–322):

```python
elif optimizer_name == "fusion":
    from stable_audio_tools.training.fusion_opt import FusionOpt
    from stable_audio_tools.training.fusion_groups import build_fusion_param_groups
    optimizer = FusionOpt(
        build_fusion_param_groups(model),
        lr=lr,
        mona_alpha=mona_alpha,
        # ...other fusion hparams
    )
```

And the `_is_sf` guard becomes `optimizer_name in {"schedulefree", "fusion"}` so the existing `.train()` / `.eval()` toggling around val + checkpoint covers FusionOpt automatically.

**Loss creation** (currently line ~360):

```python
if loss_name == "temporal":
    from stable_audio_tools.training.temporal_loss import TemporalShapeLoss
    criterion = TemporalShapeLoss(
        huber_beta=huber_beta,
        lambda_deriv=lambda_deriv,
        lambda_multi=lambda_multi,
        scales=(2, 4, 8, 16, 32, 64, 128, 256),
        point_loss="auto",
        curriculum_steps=curriculum_steps,
    )
else:
    criterion = nn.SmoothL1Loss(beta=huber_beta)
```

The call site doesn't change — `TemporalShapeLoss.forward(pred, target)` returns a scalar, same as `SmoothL1Loss`. Diagnostic logging reads `criterion.last_components["L_deriv"]` etc.

### Checkpoint format

The existing checkpoint dict gets two new keys:

```python
torch.save({
    "model_state_dict":   raw_model.state_dict(),         # FAST iterate z_t
    "averaged_state_dict": optimizer.average_state_dict(),# SF average x_t (NEW)
    "optimizer_state_dict": optimizer.state_dict(),
    "metadata": {
        ...existing fields...,
        "optimizer": "fusion",
        "loss": "temporal",
        "fusion_config": {"mona_alpha": ..., "lambda_deriv": ..., ...},
    },
    "feature_stats": feature_stats,
}, path)
```

`--save-best-only` saves the averaged `x_t` as the deployable model; the live `z_t` is saved alongside for resume only.

### Loading existing checkpoints

`load_latch_from_checkpoint(path)` (in `stable_audio_tools/models/latch.py`) auto-detects:

1. If `"averaged_state_dict"` present → use it (FusionOpt-trained head).
2. Else fall back to `"model_state_dict"` (legacy AdamW heads).

Inference paths need zero changes.

### Migration story

| What | Action |
|---|---|
| Existing trained heads in `latch_weights/` | Untouched. Continue to work in inference. |
| Existing YAML configs | Untouched. Train with the old optimizer + loss. |
| New runs that want FusionOpt | Opt in via CLI flags or YAML keys. |
| Resume FusionOpt from `_best.pt` | Loads both `z_t` and `x_t`. |
| Resume FusionOpt from a legacy AdamW checkpoint | Requires `--reset-optimizer`. Drops the AdamW state; copies model weights into both `z_t` and `x_t`; averaging restarts from a flat state. |
| `unwrap_model.py` | No changes — uses `averaged_state_dict` when present. |

### Files touched

```
NEW  stable_audio_tools/training/fusion_opt.py        # ~300 LoC
NEW  stable_audio_tools/training/temporal_loss.py     # ~80 LoC
NEW  stable_audio_tools/training/fusion_groups.py     # ~30 LoC
MOD  scripts/train_latch.py                           # ~25 lines added
MOD  stable_audio_tools/models/latch.py               # ~10 lines (averaged_state_dict autodetect)
NEW  latch_train_fusion.yaml                          # example config for FusionOpt runs
```

No changes to: inference paths, Gradio UI, `unwrap_model.py`, `pre_encode.py`, any model architecture.

## Open questions and risks

### Provenance verification

**MONA**, **KL-Shampoo**, and **ScheduleFree+** are described in `docs/ideas/optimiser_ideas` as "May 2026" releases. I have not independently verified these against papers or open-source repositories — the math is plausible and consistent with the direction the field is moving, but the specific formulae used here are my interpretations of the doc's description. If actual published versions diverge, the implementation will need to be reconciled. Muon and Schedule-Free (base) are battle-tested public algorithms; those parts are low-risk.

Mitigation: implement to the math sketched in this spec; document the implementation as "FusionOpt-2026-05" with explicit hyperparameter defaults so future cross-checking against published versions is straightforward.

### KL-Shampoo numerical conditioning under FP16

Eigendecomposition of `L_t` and `R_t` runs in FP32 (the spec calls this out as a "FP32 island"). If FP32 eigendecomp on the 256×256 SPD matrices proves slow or unstable on ROCm/hipBLASLt 7.2.x, we fall back to one of:

1. Cholesky-based preconditioner (cheaper, less accurate but well-conditioned).
2. Power iteration to approximate the top eigenvectors only (cheap, captures the dominant geometry).
3. Identity preconditioner (degrades FusionOpt to Muon-MONA + SF, losing KL-Shampoo's contribution but keeping the rest).

The bake-off will reveal which path is needed; option 3 is the safe rollback.

### Compute overhead estimate

- Per-step optimiser work, spectral path, all matrices: ~15 NS5 GEMMs + 2 EMA updates + 1 preconditioner matmul (every step) + amortised eigendecomp (every K steps, FP32) ≈ 1–2 ms.
- Per-step optimiser work, scalar path: same as ScheduleFree-AdamW ≈ 100 µs.
- Loss computation: 3 components vs. 1 in baseline; multi-scale L1 with 8 scales adds ~8× the avg_pool + SmoothL1 cost on a (B=64, F=1, T=256) tensor ≈ negligible.

Expected step-time impact: 10–20 % slower than AdamW + SmoothL1 baseline (38 it/s → ~30 it/s). Within the 4× cost tolerance.

### Memory overhead estimate

Per spectral matrix W of shape `(M, N)`:
- `m` (momentum), `A` (curvature EMA), `g_prev` (gradient memory): 3 × M·N (FP16) = 6 MN bytes
- `L` (output covariance): M² (FP32) = 4M² bytes
- `R` (input covariance): N² (FP32) = 4N² bytes
- `P_L`, `P_R` (preconditioner inverses): M² + N² (FP16) = 2(M² + N²) bytes
- `r` (row-norm EMA): M (FP16) = 2M bytes
- `x` (SF average): M·N (FP16) = 2 MN bytes

Spectral state ≈ 8 MN + 6(M² + N²) bytes per matrix. Summed across LatCH's ~6 spectral layers (each with a 256-grid matrix up to 1024×256), total spectral state ≈ 10–20 MiB. Negligible against the 16 GiB budget.

### Early-training divergence

The Schedule-Free averaging + spectral updates + Polyak step compose nontrivially. There's a known failure mode in raw schedule-free + spectral methods: if the fast iterate `z_t` diverges, the average `x_t` becomes worse than useless. SF-NorMuon's WD-on-z guards against this.

If we see divergence in the bake-off: (a) increase `λ` (spectral WD on z_t), (b) add a Polyak `γ_t` warmup so the step starts small and ramps up over the first ~500 steps, (c) increase β (SF interp) so the eval point sits closer to the averaged iterate.

### Cross-loss attribution

Since we're changing both the loss and the optimiser, the 2×2 ablation matrix is the discipline that makes this debuggable. If only one cell (B2) is run, no attribution is possible.

## References

- **Muon** — Keller Jordan, "Muon: An Optimizer for the Hidden Layers of Neural Networks", https://github.com/KellerJordan/Muon. Newton-Schulz orthogonalisation, quintic polynomial coefficients (3.4445, -4.7750, 2.0315). Public, mature implementation. *(verified)*
- **Schedule-Free** — Defazio et al., `schedulefree` PyPI package. Iterant averaging without explicit learning-rate schedule. Already used in this codebase via `AdamWScheduleFree`. *(verified)*
- **MONA** — claimed as "Muon Optimizer with Nesterov Acceleration" (May 2026) in `docs/ideas/optimiser_ideas`. Adds an EMA-of-gradient-differences curvature term to Muon's momentum. *(provenance unverified; math implemented as described)*
- **ScheduleFree+** — claimed as a Polyak-step + inner-momentum extension of Schedule-Free (May 2026). *(provenance unverified)*
- **SF-NorMuon** — claimed as the fusion of Schedule-Free averaging with Muon spectral updates, with weight decay on the fast iterate. *(provenance unverified; the WD-on-z trick is the core stability mechanism we rely on)*
- **KL-Shampoo** — claimed as a Kronecker-factored preconditioner with KL-divergence-based SPD estimation, eliminating Adam-grafting overhead. *(provenance unverified; we implement the two-sided Kronecker covariance update with FP32 eigendecomp and treat it as a Shampoo variant)*
- `LATCH_RESULTS.txt` §18 — current head inventory, corrected raw-MAE table, the standardize-regression context for `spectral_flatness` and `spectral_skewness`.
- `Stable Audio 3/tunings_throughput.html` — RX 9070 XT (gfx1201) hipBLASLt GEMM throughput data; source of the 256-grid fast-tile constraints and the "multiple of d_model" rule.
- `docs/specs/` — sibling specs for related LatCH work (sigmas/schedule, paper interpretation, verification protocol).

## Approval and next steps

User (Kim) approved this spec on 2026-05-29 with delegated authority to proceed autonomously through implementation and Phase 1 bake-off within a ~6-hour budget. GPU access waits ~1 hour for a concurrent Claude Code instance to complete. Implementation proceeds via the writing-plans skill; bake-off results to be summarised in `LATCH_RESULTS.txt` §19 for morning review.
