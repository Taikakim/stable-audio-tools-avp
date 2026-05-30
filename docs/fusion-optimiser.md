# FusionOpt — design, recipe, and empirical case study

A composed optimiser for small-to-medium DiT-style models. Bifurcates parameters
into a **spectral path** (Muon + KL-Shampoo + SF-NorMuon for 2D matrices ≥ 128 × 128)
and a **scalar path** (ScheduleFree-AdamW for biases, LayerNorm, small projections).
Shared Schedule-Free outer loop with a Polyak step size — all reductions on-device,
no host syncs in the optimiser hot loop.

This document is project-independent. Concrete numbers come from a case study
deployment (small audio-control heads on RDNA4 / ROCm); they're framed as
evidence rather than as deployment requirements.

If you want a portable single-page handbook, see [`FUSION_SHAREABLE.md`](./FUSION_SHAREABLE.md).
The detailed design spec lives at
[`docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md`](./superpowers/specs/2026-05-29-fusion-optimiser-design.md).

## TL;DR

```bash
--optimizer fusion --components ns5,normuon,sf \
--hot-dtype bf16 \
--compile
```

That's **SF-NorMuon** at bf16 with torch.compile. Equivalent to manually setting:

```python
optimizer = FusionOpt(
    params=build_fusion_param_groups(model),
    lr=3e-4,
    components={"ns5", "normuon", "sf"},   # the load-bearing subset
    hot_dtype="bf16",                       # speed-favoring; "fp16_safe" is quality-favoring
)
```

For **diversity-incentivised** training (penalising similarity to a frozen
reference head), use the full composition instead — warm-started from your
SF-NorMuon ship checkpoint:

```bash
--optimizer fusion --components ns5,normuon,sf,mona,shampoo \
--hot-dtype bf16 --compile \
--warm-start --diversity-ref path/to/sf_normuon_ship_head.pt --lambda-div 0.3
```

The KL-Shampoo + MONA components, dropped from production for cost, turn
out to be the stabilisers under a *negative magnitude-unbounded* loss term.
See the [Diversity-incentivised training](#diversity-incentivised-training--and-why-full-fusion-is-the-recipe-here-not-sf-normuon)
section for the case-study evidence.

In our case study (5 M-param adaLN-zero transformer heads), the production
recipe gave **−7 % to −20 % raw MAE** vs an AdamW baseline at the **same wall-clock**
after architecture co-tuning. Other workloads should test the recipe
applicability section.

## What FusionOpt fuses

All seven building blocks are published optimisers. The novelty is the
**composition**, the **param-group routing**, the **shared Polyak step size**,
and the **inference-side time cache** designed alongside the optimiser.

| Component | Mechanism | Source |
|---|---|---|
| **Muon** | Newton-Schulz quintic orthogonalisation on 2D weights; spectral-norm bound on the update. | Keller Jordan, [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon). |
| **NorMuon** | Per-neuron row-norm normalisation after NS5 (fixes uneven row magnitudes from a fixed NS-iteration count). | [arXiv:2510.05491](https://arxiv.org/abs/2510.05491). |
| **MONA** | EMA of gradient differences injected into momentum as a curvature proxy; deflects from sharp minima. Validated on 1 B–68 B MoE pretraining. | [arXiv:2605.26842](https://arxiv.org/abs/2605.26842). |
| **KL-Shampoo** | Two-sided Kronecker covariance preconditioner via KL divergence; matches Shampoo/SOAP without Adam grafting. | [arXiv:2509.03378](https://arxiv.org/abs/2509.03378). |
| **Schedule-Free** | Anytime-stopping framework: gradients at y_t = (1−β)·z_t + β·x_t, deploy from averaged x_t. | Defazio et al., [`schedulefree`](https://github.com/facebookresearch/schedule_free) PyPI package. |
| **ScheduleFree+** | Polyak step size on top of Schedule-Free; LR-free training at LLM scale. | [arXiv:2605.19095](https://arxiv.org/abs/2605.19095). |
| **SF-NorMuon** | Schedule-Free + NorMuon + WD on the **fast iterate** z_t (not x_t) — load-bearing for long-horizon stability. | [arXiv:2605.23061](https://arxiv.org/abs/2605.23061). |

The composition we recommend shipping (SF-NorMuon) is precisely
arXiv:2605.23061 + arXiv:2510.05491 — we discovered it empirically via a
per-component ablation rather than reading the paper first. Both routes
converge on the same recipe.

## Applicability

Use FusionOpt when:

- You're training a **transformer-style network with 2D weight matrices**
  (qkv projections, MLP fc layers ≥ 128 × 128).
- You're at **5 M – 10 B parameters per model** (where the papers validate).
- Your hardware has **good tuned bf16 matmul kernels** (most modern accelerators
  — H100, B100, MI3xx, RDNA4).
- You'd benefit from a **schedule-free training loop** — no LR schedule
  tuning, anytime-stopping yields a deployable model.

Don't use it for:

- **Pure conv nets** — Muon's matrix-aware updates aren't directly meaningful
  on 4-D conv kernels.
- **Very small networks** (< 1 M params) — overhead dominates.
- **Workloads where AdamW's per-element adaptivity wins** — very sparse
  gradients, weird scale-mismatched parameter groups.

## The formula

```
FusionOpt — bifurcated optimiser

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
  # 1. KL-Shampoo two-sided Kronecker covariance (FP32 always)
  L_t = β_k L_{t−1} + (1 − β_k) g_t g_tᵀ
  R_t = β_k R_{t−1} + (1 − β_k) g_tᵀ g_t
  every K steps:                                         # eigendecomp amortised
      P_L = (L_t + δI)^{−1/4}   via torch.linalg.eigh, FP32
      P_R = (R_t + δI)^{−1/4}

  # 2. MONA curvature-augmented momentum
  A_t = β_n A_{t−1} + (g_t − g_{t−1})
  m_t = μ m_{t−1} + g_t + α A_t

  # 3. Preconditioner + Muon Newton-Schulz quintic (hot dtype, bf16 default)
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
  μ      = 0.95   α   = 0.2   β_n = 0.9
  β_k    = 0.99   K   = 100   δ   = 1e-4
  β_r    = 0.95
  λ      = 0.01 (spectral)  λ_s = 0 (scalar)
  β_1    = 0.9    β_2 = 0.999  ε   = 1e-8
```

Two non-obvious binding properties:

- The Polyak `γ_t` is **shared** across both paths, so spectral and scalar
  weights move at proportional rates per step.
- Weight decay is on `z_t` (the fast iterate), not on `x_t` (the average) —
  the **load-bearing stability mechanism**. Without it, schedule-free +
  spectral updates drift unboundedly over long horizons (proved in the
  SF-NorMuon paper; empirically verified).

## Production hot-dtype: bf16

```bash
--hot-dtype bf16    # production default (speed-favoring)
--hot-dtype fp32    # safer fallback (1.5–2x slower than bf16)
--hot-dtype fp16    # UNSAFE — NS5 quintic overflows. Diverges.
--hot-dtype fp16_safe   # quality-favoring (fp16 matmul + fp32 polynomial accum)
```

**fp16 diverges to NaN after a few optimiser steps.** NS5 normalises by
Frobenius norm so input values are ~ O(1), but `X @ Xᵀ` on a 1024 × 256
matrix can reach values up to ~256, and the iterated polynomial
`X (XᵀX)²` reaches ~ 312 000 — well over fp16's 65 504 max. Tensor cores
accumulate in fp32 internally but the cast back to fp16 between matmuls
loses the headroom.

**bf16 has fp32-equivalent exponent range** (8-bit), so the overflow can't
happen even with the iterated polynomial. The 7-bit mantissa loses ~1 % on
val MAE compared to fp32; well worth the stability.

**`fp16_safe`** (added as an option after the fp16 finding) does fp16 matmul
with fp32 polynomial accumulation and per-tensor rescale-restore around each
matmul. **Quality-favoring**: in our case study it ran at 0.5× bf16
throughput but delivered val MAE slightly *better* than fp32. Useful for
ship-retrain refinement passes where 1 % quality matters more than wall-clock.
On RDNA4 the throughput cost makes it impractical as a default; on NVIDIA
hardware where reduction-vs-matmul ratios differ, it may be more competitive.

## Empirical findings — case study

Our deployment was small audio-control heads (5 M params, dim=256, depth=4-6,
adaLN-zero conditioning, predicting MIR features from noisy VAE latents).
Numbers below are from a 30-epoch, full-data training run on RDNA4 / ROCm 7.2.x.

### Quality — production ship retrain

After settling on the production recipe (SF-NorMuon + bf16 + depth-4 shrink):

| Head | AdamW baseline (current ship) | FusionOpt recipe | Δ |
|---|---|---|---|
| rms_energy_bass | 3.4683 | 3.0339 | **−12.5 %** |
| spectral_flatness | 0.0560 | 0.0450 | **−19.6 %** |
| spectral_flux | 16.3180 | 14.2560 | **−12.6 %** |

At the **same wall-clock per step** as AdamW. The "3 × slower" worry that
appeared in early bake-offs was specific to Full Fusion at the original
depth-6 architecture; SF-NorMuon at depth-4 + bf16 + compile matches
AdamW throughput (~45 it/s in our case study).

### Per-component ablation — what each piece does on its own

We ran each individual inspiration as a standalone cell with SmoothL1 loss
to isolate optimiser effect from loss effect:

| Variant | components | val_MAE | Δ vs AdamW |
|---|---|---|---|
| AdamW (baseline) | — | 3.4683 | baseline |
| Muon | `ns5` | 3.5619 | +2.7 % (worse) |
| MONA | `ns5,mona` | 3.6709 | +5.8 % (worse) |
| KL-Shampoo | `shampoo` | **NaN (DIVERGED)** | — |
| ScheduleFree+ | `sf` | 4.2949 | +23.8 % (worse) |
| **SF-NorMuon** | `ns5,normuon,sf` | **3.2063** | **−7.6 %** ← only standalone winner |
| Full Fusion | (all) | 3.1921 | −8.0 % |

**Individual components mostly underperform plain AdamW** at our model
scale. Only SF-NorMuon — the simplest composition that includes all the
load-bearing pieces — captures the bulk of the Full Fusion lift on its own.
Adding Shampoo and MONA on top adds 0.4 % beyond SF-NorMuon at 50 % more
wall-clock.

Why the rest fail individually:

- **KL-Shampoo alone diverges to NaN** in epoch 1. The
  `(L + δI)^(−1/4)` preconditioner has no spectral-norm bound; the
  Polyak ratio amplifies the Kronecker rescaling. Classical Shampoo
  instability. The spec's choice to compose Shampoo with NS5 turns out
  to be load-bearing.
- **MONA + Muon underperforms pure Muon** in our 5 M-param regime. The
  α=0.2 curvature deflection destabilises updates without Shampoo's
  preconditioner softening it. The MONA paper validates on 1 B–68 B MoE
  scales; the recipe may help at larger sizes than ours.
- **ScheduleFree+ alone is the worst** (+23.8 %). Without a spectral
  path, Polyak's effective LR at `γ_max = 10 × γ_base` exceeds the
  model's stability envelope on raw Adam-style updates.

### Profile — where the cost lives

Per-component microbench at dim=256, depth=6 (32 spectral parameters):

| Component | Per step (ms) | % step |
|---|---|---|
| MONA momentum | 2.07 | 5.4 % |
| Shampoo factor update | 3.99 | 10.4 % |
| Shampoo eigendecomp | 0.34 (amortised) | 0.9 % |
| Shampoo preconditioner apply | 2.97 | 7.8 % |
| **Muon NS5 quintic** | **15.96** | **41.8 %** |
| Per-neuron row scale | 1.34 | 3.5 % |
| Schedule-Free averaging | 1.22 | 3.2 % |
| **step_total** | **38.20** | **100 %** |

NS5 dominates. The original spec hypothesised KL-Shampoo would dominate;
it doesn't.

### Architecture trade-offs

Shrinking the model is cheaper than other optimisations for inference cost.
In our case study:

| Config | params | val MAE (bass) | inference cost |
|---|---|---|---|
| d256 / dp6 (original) | 7.25 M | 3.1785 | baseline |
| d256 / dp4 | 4.87 M | 3.2294 (+1.6 %) | 67 % |
| d128 / dp6 | 1.97 M | 3.2640 (+2.7 %) | 50 % |
| d128 / dp4 | 1.31 M | 3.3630 (+5.8 %) | 33 % |

**Depth shrinks cheaper than width**: 4 → 6 depth cost 1.6 % quality
for 33 % inference savings; halving width cost 2.7 %. Width contributes
squared, depth linear — depth is the cheaper axis.

### Diversity-incentivised training — and why Full Fusion is the recipe here, not SF-NorMuon

Beyond the score-driven ship retrain, the FusionOpt setup makes *diversity
training* tractable. By adding a penalty `−λ · MSE(pred, ref_pred)` that
rewards divergence from a frozen reference model, we get heads that solve
the same task but represent it differently. Useful when artistic diversity
matters more than scalar val loss.

**Important: the production SF-NorMuon recipe is NOT the right choice for
diversity training.** The KL-Shampoo + MONA components — which we drop from
production for cost reasons (50 % more wall-clock for <1 % score quality) —
turn out to be load-bearing stabilisers when the loss function includes a
*negative* magnitude-unbounded term. The case-study result that justifies
this claim:

In our case study (penalty λ=0.3 against the production SF-NorMuon ship head):

| Variant | val MAE | val deriv-corr | notes |
|---|---|---|---|
| Reference (SF-NorMuon production ship) | 3.03 | 0.74 | baseline |
| **Warm-start + Full Fusion + diversity** | **4.19** | **0.33** | **"drifted but structured" — recommended diversity recipe** |
| Fresh init + SF-NorMuon + diversity | 6.04 | 0.01 | "alien coherent" — uncorrelated with target's direction |
| AdamW + diversity (fresh OR warm) | NaN | NaN | broken — AdamW can't handle negative loss components |

Reading the table:

- **Warm-start + Full Fusion is the diversity-training winner.** The head
  keeps recognisable structure (deriv_corr 0.33 — partial correlation with
  the target's direction) while drifting from the reference (val_MAE 4.19
  vs 3.03). Musically: it still listens to the same audio cue but expresses
  its target a few dB sideways. Useful as an aesthetic-palette alternative
  to the production head.
- **Fresh init + SF-NorMuon** finds a coherent but unrelated solution
  (deriv_corr ≈ 0). Predictions are aesthetically uncorrelated with the
  target's actual direction. May produce timbral surprises, but not a
  controllable drift of the production behaviour.
- **AdamW + diversity diverges to NaN** in both fresh-init and warm-start
  variants. Adam's exponential moving averages can't bound the negative
  loss component. Schedule-Free averaging + NS5 row-scaling survives because
  of its internal magnitude clipping.

The practical implication for production: ship with SF-NorMuon for the
score-driven heads; use **Full Fusion (warm-started from the SF-NorMuon
ship head) with a diversity penalty** when you want a parallel "personality
variant". Two recipes for two purposes — the KL-Shampoo + MONA components
aren't vestigial, they're load-bearing under adversarial objectives.

## Code shapes

### FusionOpt with components flag

```python
class FusionOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4,
                 components: set[str] | None = None,
                 hot_dtype: str = "bf16",
                 # ...
                 ):
        all_comps = {"mona", "shampoo", "ns5", "normuon", "sf"}
        if components is None:
            components = all_comps
        self._components = frozenset(components)
        # ... rest of init
```

Conditional execution in the spectral step makes each piece a config flag:

```python
# 1. KL-Shampoo (optional)
if "shampoo" in self._components:
    L_t.mul_(beta_k).add_(g @ g.T, alpha=(1 - beta_k))
    R_t.mul_(beta_k).add_(g.T @ g, alpha=(1 - beta_k))
    if step % eigen_period == 0:
        state["P_L"] = _inv_quarter(L_t)  # FP32
        state["P_R"] = _inv_quarter(R_t)

# 2. MONA augmented momentum (optional)
if "mona" in self._components:
    A_buf.mul_(beta_n).add_(g - g_prev)
    g_prev.copy_(g)
    m.mul_(mu).add_(g + alpha * A_buf)
else:
    m.mul_(mu).add_(g)

# 3. Preconditioner (optional)
m_pre = state["P_L"] @ m @ state["P_R"] if "shampoo" in self._components else m
m_pre = m_pre.to(hot_dtype)

# 4. NS5 (optional, but typically required for stability of shampoo + sf)
if "ns5" in self._components:
    U = newton_schulz_5(m_pre).float()
    U *= max(1.0, fan_out/fan_in) ** 0.5
else:
    U = m_pre.float()

# 5. Row scaling (optional)
if "normuon" in self._components:
    r.mul_(beta_r).add_(rowsumsq(U), alpha=(1-beta_r))
    U /= r.clamp_min(1e-12).sqrt().unsqueeze(-1)

# 6. SF averaging or direct update
if "sf" in self._components:
    z.mul_(1 - gamma_t * wd).add_(U, alpha=-gamma_t)
    x.mul_(1 - 1/t).add_(z, alpha=1/t)
else:
    p.data.mul_(1 - gamma_t * wd).add_(U, alpha=-gamma_t)
```

### TimeConditioningCache for inference (adaLN-zero only)

Precompute the time-conditioning operations that are pure functions of t
and weights, look them up at sampler runtime:

```python
class TimeConditioningCache:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self._by_value: dict[float, dict] = {}

    def warm_for_schedule(self, n_steps, include_zero=True):
        for t in torch.linspace(1.0, 0.0, n_steps + 1)[:-1].tolist():
            self.ensure(t)
        if include_zero:
            self.ensure(0.0)

    def ensure(self, t):
        k = round(float(t), 6)
        if k in self._by_value:
            return
        with torch.no_grad():
            t_tensor = torch.tensor([t], device=self.device)
            t_emb = self.model.t_embedder(t_tensor)
            mods = []
            for block in self.model.blocks:
                if hasattr(block, "adaLN_mod"):
                    full = block.adaLN_mod(t_emb)
                    mods.append(tuple(full.chunk(6, dim=-1)))
                else:
                    mods.append(None)
            self._by_value[k] = {"t_emb": t_emb, "modulators": mods}

    def get(self, t, auto_warm=True):
        k = round(float(t), 6)
        entry = self._by_value.get(k)
        if entry is None and auto_warm:
            self.ensure(t)
            entry = self._by_value.get(k)
        return entry
```

Wire into model forward (only when t is uniform across the batch):

```python
def forward(self, x, t):
    cache_entry = None
    cache = getattr(self, "_time_cache", None)
    if cache is not None and t.numel() > 0:
        t_first = t.flatten()[0]
        if t.numel() == 1 or torch.all(t == t_first):
            cache_entry = cache.get(float(t_first.item()))
    if cache_entry is not None:
        t_emb = cache_entry["t_emb"]
        block_mods = cache_entry["modulators"]
    else:
        t_emb = self.t_embedder(t)
        block_mods = [None] * len(self.blocks)
    for i, block in enumerate(self.blocks):
        x = block(x, t_emb, ..., mods=block_mods[i])
```

A process-level registry keyed by `(model_path, n_steps)` lets caches
survive across multiple inference calls; first inference at any given
`(checkpoint, step-count)` fully populates the cache, subsequent calls
hit 100 %.

## Diagnostic logging

For WandB / TensorBoard observability:

```python
def diagnostic_summary(self):
    return {
        "fusion/step_count": self._step_count,
        "fusion/mode": 0 if self._mode == "train" else 1,
        "fusion/gnorm_ema": float(self._gnorm_ema or 0.0),
        "fusion/loss_ema":  float(self._loss_ema or 0.0),
    }
```

Watch `fusion/gnorm_ema` and `fusion/loss_ema` for stability:
- `gnorm_ema` should grow smoothly during the first few hundred steps then
  plateau. Sudden spikes signal divergence.
- `loss_ema / gnorm_ema` is the Polyak ratio; if it sits at the clamp
  ceiling (10 or 0.1) for many steps, the clamp is fighting Polyak.

## Limitations and open problems

1. **`--hot-dtype fp16_safe` is a quality knob, not a speed knob** on RDNA4.
   The fp16 matmul throughput win is eaten by per-tensor rescale overhead.
   Custom Triton kernel with rescale fused into the matmul epilogue could
   reclaim the theoretical 1.3–1.5×; ~200 LoC of real engineering.

2. **MONA at small scale doesn't help** in our case study. The mechanism is
   real (paper validated 1 B–68 B) but the curvature deflection seems
   scale-dependent. Test at your scale before including.

3. **AdamW + diversity penalty is broken** — Adam's exponential moving
   averages can't handle the negative loss component. Spectral optimisers
   work because their internal normalisations bound update magnitudes.

4. **Polyak γ clamp default `[0.1, 10]` × γ_base may need tuning** on
   workloads with non-standardised regression targets where loss magnitude
   is far from gradient magnitude.

5. **Eigendecomp must run in fp32** even when the rest of Shampoo runs in
   bf16. Tested fp16 eigendecomp briefly — produced garbage. Cast L, R to
   fp32, decompose, cast back. This is the "fp32 island" pattern.

## References

- **SF-NorMuon** — *Anytime Training with Schedule-Free Spectral Optimization*, [arXiv:2605.23061](https://arxiv.org/abs/2605.23061). Load-bearing recipe + WD-on-fast-iterate proof.
- **NorMuon** — *Making Muon more efficient and scalable*, [arXiv:2510.05491](https://arxiv.org/abs/2510.05491). Per-neuron row normalisation.
- **MONA** — *Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training*, [arXiv:2605.26842](https://arxiv.org/abs/2605.26842).
- **KL-Shampoo** — *Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization*, [arXiv:2509.03378](https://arxiv.org/abs/2509.03378).
- **ScheduleFree+** — *Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models*, [arXiv:2605.19095](https://arxiv.org/abs/2605.19095).
- **Muon** — Keller Jordan, [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon). The Newton-Schulz quintic with coefficients (3.4445, −4.7750, 2.0315).
- **Schedule-Free** — Defazio et al., [`schedulefree`](https://github.com/facebookresearch/schedule_free). The y_t / z_t / x_t framework.
