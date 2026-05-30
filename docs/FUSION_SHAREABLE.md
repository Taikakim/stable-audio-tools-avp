# FusionOpt — portable handbook

A composed PyTorch optimiser that fuses **Muon** (spectral orthogonalisation),
**MONA** (curvature-aware momentum), **KL-Shampoo** (Kronecker preconditioner),
and **ScheduleFree+** (averaging + Polyak step) — and a `TimeConditioningCache`
for diffusion-model inference. Designed for small-to-medium DiT-style networks
(~5 M – ~10 B parameters, adaLN-zero conditioning, transformer-style 2D weight
matrices).

This document is project-independent. Concrete numbers below come from a
case-study deployment (small audio-control heads on RDNA4), but every
recommendation transfers to any architecture that meets the assumptions
listed under "Applicability".

## TL;DR — production recipe

For most small-to-medium DiT setups with adaLN-zero conditioning and FP16/BF16
training:

```bash
python train.py \
  --optimizer fusion --components ns5,normuon,sf \
  --hot-dtype bf16 \
  --epochs 30 --batch-size 64 --lr 3e-4 \
  --compile --t-injection adaln_zero \
  --save-best-only
```

That is **SF-NorMuon** (NS5 + per-neuron row scaling + Schedule-Free averaging)
plus bf16 matmul precision plus `torch.compile`. Empirically this is the
load-bearing subset of the full Fusion composition; adding MONA and KL-Shampoo
on top buys ≤ 1 % quality for ~50 % more wall-clock time. **Skip them by
default.**

In our case-study (5 M-param control heads), this recipe delivered:
- **−7 % to −20 % raw-MAE** vs an AdamW baseline at the same wall-clock.
- **Same it/s** as AdamW after architecture co-tuning (depth shrink).

## Applicability — when does FusionOpt help?

✓ **Yes**:
- Transformer-style 2D weight matrices (qkv, projections, MLPs ≥ 128 × 128).
- adaLN-zero conditioning (gives the time cache its leverage; not required).
- Mid-range scales (5 M – 10 B params per the paper validations).
- BF16- or FP16-friendly hardware with tuned matmul kernels.

✗ **No / unclear**:
- Pure conv nets — spectral methods like Muon are matrix-aware, less
  appropriate for 4-D conv kernels.
- Very small networks (< 1 M params) — overhead may dominate.
- Workloads where AdamW's per-element adaptivity matters more than spectral
  geometry (e.g. very sparse gradients).

## What FusionOpt fuses

All seven building blocks are real, recently-published optimisers. The novelty
is the **composition**, not the components.

| Component | Mechanism | Source |
|---|---|---|
| **Muon** | Newton-Schulz quintic orthogonalisation on 2D weights; spectral-norm bound on the update. | Keller Jordan, [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon). |
| **NorMuon** | Per-neuron row-norm normalisation after NS5 (fixes uneven row magnitudes from fixed NS-iteration count). | [arXiv:2510.05491](https://arxiv.org/abs/2510.05491). |
| **MONA** | EMA of gradient differences injected into momentum as a curvature proxy; deflects from sharp minima. | [arXiv:2605.26842](https://arxiv.org/abs/2605.26842). Validated on 1 B–68 B MoE pretraining. |
| **KL-Shampoo** | Two-sided Kronecker covariance preconditioner via KL divergence; matches Shampoo without Adam grafting. | [arXiv:2509.03378](https://arxiv.org/abs/2509.03378). |
| **Schedule-Free** | Anytime-stopping framework: gradients at y_t = (1−β)·z_t + β·x_t, deploy from averaged x_t. | Defazio et al., [schedulefree](https://github.com/facebookresearch/schedule_free). |
| **ScheduleFree+** | Polyak step size on top of Schedule-Free; LR-free training at LLM scale. | [arXiv:2605.19095](https://arxiv.org/abs/2605.19095). |
| **SF-NorMuon** | Schedule-Free + NorMuon + WD on the **fast** iterate z_t (not x_t) — the load-bearing stability mechanism. | [arXiv:2605.23061](https://arxiv.org/abs/2605.23061). |

## Findings that transfer

Empirically validated and likely to hold on any DiT-style workload:

1. **bf16 is the right hot-path dtype for NS5; fp16 diverges.** The iterated
   quintic `X (XᵀX)²` produces intermediate values that exceed fp16's 65 504
   ceiling on any matrix bigger than ~256 × 256. NaN after a few optimiser
   steps. bf16's fp32-equivalent exponent (8-bit vs fp16's 5-bit) prevents
   overflow. Quality cost vs fp32: ~1 % on raw MAE.

2. **SF-NorMuon (ns5+normuon+sf) captures ~95 % of the quality lift over
   AdamW** in case-study experiments. Adding MONA and KL-Shampoo on top buys
   ~1 % for ~50 % more wall-clock. **Use the smaller composition unless that
   last 1 % matters.**

3. **Schedule-Free averaging + WD on the fast iterate z_t** is the stability
   mechanism that makes spectral updates safe over long horizons. WD on x_t
   (the average) lets z_t drift to infinity over thousands of steps. The
   SF-NorMuon paper's theory checks out empirically.

4. **`torch.compile` is critical.** Without it, the spectral-path overhead
   dominates the per-step cost. With it, the per-step cost matches AdamW on
   the same model size. Don't run optimiser benchmarks without compile.

5. **adaLN-zero block conditioning enables a TimeConditioningCache for
   inference.** At fixed sampler step counts, t_emb + per-block modulators
   are pure functions of t and weights → cacheable. ~5–10 % inference
   latency win at 40-step rendering.

## Findings that may or may not transfer

Scale- and workload-dependent:

1. **MONA at small scale**: in our case-study (5 M params), MONA + Muon
   *underperforms* pure Muon (+5.8 % vs +2.7 % MAE over AdamW). The MONA
   paper validates the recipe at 1 B–68 B MoE scales. The curvature deflection
   may help at LLM scale but doesn't here. Test before shipping.

2. **KL-Shampoo standalone diverges to NaN** in our case-study. The SPD
   covariance preconditioner has no spectral-norm bound; without NS5 to clip
   the update magnitude, the rescaling blows up. This is the classical
   Shampoo instability the paper addresses; **must be composed with NS5 (or
   grafted with Adam)** to be stable. Holds on any size.

3. **The d→ depth shrink vs d→ width shrink tradeoff**: for inference cost,
   cut depth not width. (In our case study, depth 4 → 6 cost 1.6 % quality
   for 33 % inference savings; halving width was much worse.) Width
   contributes squared, depth linear — depth is the cheaper axis.

## Findings that are hardware-specific (RDNA4 case study)

Our measurements were on AMD RX 9070 XT (RDNA4, gfx1201) with ROCm 7.2.x and
hipBLASLt build dabb6df2b98. Other platforms will differ:

1. **TunableOp autotuning matters.** On a fresh cache, the bf16 path may have
   no tuned kernels; first run takes ~30 % longer. Bake in a warmup cell
   before benchmarking. Set `PYTORCH_TUNABLEOP_TUNING="1"` BEFORE
   `import torch` — late application is silently ignored.

2. **fp16-with-fp32-intermediates (rescale-restore inside NS5)** doesn't pay
   off on RDNA4: the rescale-restore overhead (~6 max-reductions × matmuls
   per step) dominates the fp16 throughput win. On NVIDIA H100/Blackwell
   where reduction-vs-matmul throughput ratios differ, this *might* be
   different — testable.

3. **`hot_dtype="fp16_safe"` (= fp16 matmuls + fp32 polynomial accumulation
   + per-tensor rescale)** is **a quality knob, not a speed knob**: in our
   case study it ran 0.5× bf16 throughput but delivered val_MAE slightly
   better than fp32. Useful for ship-retrain refinement passes where 1 %
   quality matters more than wall-clock.

## Code shapes to copy

### FusionOpt component flags

A `components` set selects which spectral-path mechanisms run. The default
is the full set; the production target uses a subset.

```python
class FusionOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4,
                 components=None,           # subset of {"mona","shampoo","ns5","normuon","sf"}
                 hot_dtype="bf16",          # "fp32" | "fp16" | "bf16" | "fp16_safe"
                 # ...):
        all_comps = {"mona", "shampoo", "ns5", "normuon", "sf"}
        if components is None:
            components = all_comps
        self._components = frozenset(components)
```

Conditional execution in the spectral step:

```python
if "shampoo" in components:
    # KL-Shampoo factor update (FP32 — eigendecomp lives here)
    L_t = β_k L + (1−β_k) g g.T
    R_t = β_k R + (1−β_k) g.T g
    if step % K == 0:
        P_L = (L + δI)^(−1/4)
        P_R = (R + δI)^(−1/4)

if "mona" in components:
    A_t = β_n A + (g − g_prev); g_prev = g
    m_t = μ m + g + α A_t
else:
    m_t = μ m + g

m' = P_L @ m_t @ P_R if "shampoo" in components else m_t

# Convert to hot_dtype for the NS5 hot path
m'_hot = m'.to(hot_dtype)

if "ns5" in components:
    X = m'_hot / ||m'_hot||_F
    for _ in range(5):
        A = X @ X.T
        X = 3.4445 X − 4.7750 A X + 2.0315 (A A) X
    U = X * sqrt(max(1, fan_out / fan_in))
else:
    U = m'_hot

U = U.float()

if "normuon" in components:
    r_t = β_r r + (1−β_r) rowsumsq(U)
    U = U / sqrt(r_t.unsqueeze(-1) + ε)

if "sf" in components:
    z = (1 − γ·λ) z − γ U      # WD on FAST iterate
    x = (1 − 1/t) x + (1/t) z
else:
    p = (1 − γ·λ) p − γ U      # direct WD on live weights
```

**Critical detail**: eigendecomp must run in fp32 (cast L, R, run
`torch.linalg.eigh`, cast P_L/P_R back to hot dtype for the matmul). FP16
eigendecomp on an SPD covariance matrix produces garbage.

### Param routing — spectral vs scalar split

The spectral path applies to 2D matrices with both dims ≥ 128; everything
else (biases, LayerNorm, embeddings, output projections) goes to
ScheduleFree-AdamW.

```python
def build_fusion_param_groups(model, force_scalar=()):
    spectral, scalar = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(re.search(pat, name) for pat in force_scalar):
            scalar.append((name, p))
            continue
        if p.ndim == 2 and min(p.shape) >= 128:
            spectral.append((name, p))
        else:
            scalar.append((name, p))
    return [
        {"params": [p for _,p in spectral], "group_type": "spectral", "weight_decay": 0.01},
        {"params": [p for _,p in scalar],   "group_type": "scalar",   "weight_decay": 0.0},
    ]
```

### TimeConditioningCache for inference (adaLN-zero only)

For models with `adaLN_mod` per block and a fixed-step-count sampler:

```python
class TimeConditioningCache:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self._by_value: dict[float, dict] = {}

    def warm_for_schedule(self, n_steps, include_zero=True):
        # For rectified flow: linspace(1, 0, n+1)[:-1]
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

Wire into model forward (only when t is uniform across batch):

```python
def forward(self, x, t):
    cache_entry = None
    cache = getattr(self, "_time_cache", None)
    if cache is not None and t.numel() > 0:
        t_first = t.flatten()[0]
        if t.numel() == 1 or torch.all(t == t_first):
            cache_entry = cache.get(float(t_first.item()))

    if cache_entry is not None:
        t_emb = cache_entry["t_emb"]                   # (1, dim) — broadcasts
        block_mods = cache_entry["modulators"]
    else:
        t_emb = self.t_embedder(t)
        block_mods = [None] * len(self.blocks)

    for i, block in enumerate(self.blocks):
        x = block(x, t_emb, ..., mods=block_mods[i])
```

The block accepts `mods=None` to keep the live path identical to before:

```python
def forward(self, x, t_emb, ..., mods=None):
    if mods is None:
        g1, b1, a1, g2, b2, a2 = self.adaLN_mod(t_emb).chunk(6, dim=-1)
    else:
        g1, b1, a1, g2, b2, a2 = mods
    # ... rest of block forward
```

A persistent module-level registry keyed by `(model_path, n_steps)` makes
caches survive across inference calls; first inference at a given
(checkpoint, step-count) populates the cache fully, subsequent calls
hit 100 %.

## Verification checklist for porting

When applying FusionOpt to a new project:

1. **First do an `--hot-dtype fp16` smoke run** — if val loss is NaN after
   epoch 1, confirm bf16 instead. Universal finding.
2. **MONA value at your scale**: A/B `--components ns5,normuon,sf` vs
   `--components ns5,mona,normuon,sf` on one feature is sufficient.
3. **Depth-shrink tradeoff**: test on one config; depth tends to shrink
   cheapest.
4. **Polyak γ clamp**: default `gamma_max=10` may need tightening on
   workloads with bigger gradient magnitudes. Watch `fusion/gnorm_ema` and
   `fusion/loss_ema` for early-training instability.
5. **TunableOp cache populates**: first training run takes ~30 % longer
   while the matmul autotuner runs. Bake it in as a "warmup" cell before
   benchmarks.
6. **`PYTORCH_TUNABLEOP_TUNING="1"` env var BEFORE `import torch`** — the
   common gotcha. If applied late, the env vars are silently ignored.

## Files to copy

```
fusion_opt.py              # FusionOpt(Optimizer), newton_schulz_5,
                           # newton_schulz_5_fp16_safe, _inv_quarter
fusion_groups.py           # build_fusion_param_groups param router
time_cache.py              # TimeConditioningCache + get_or_build_cache
```

Plus the model-forward and block-forward adjustments shown above.

## Open problems (testable on your project)

- **fp16 with fused rescale-restore inside NS5.** Tested and rejected on
  RDNA4: per-tensor rescale overhead (~6 max-reductions × 480 matmuls/step)
  dominates the fp16 throughput win. A custom Triton kernel with rescale
  fused into the matmul epilogue could reclaim the theoretical 1.3–1.5×,
  but that's ~200 LoC of real engineering. May behave differently on
  NVIDIA hardware.
- **Model soup across heterogeneous heads doesn't work** for "averaging
  the best heads" — heads at similar val_loss live in different basins,
  uniform soup falls into a high-loss ridge. The form of "soup" that
  works for us is Schedule-Free's trajectory averaging (already in SF).
- **Diversity-incentivised training** (penalise being close to a reference
  head): tested in case study. Worked with spectral optimisers
  (SF-NorMuon, Full Fusion), NaN'd with AdamW — AdamW's exponential
  moving averages can't handle the negative loss component. Warm-start
  from the reference + Full Fusion produces "drifted but structured"
  variants, useful for aesthetic-palette deployment.

## Provenance

All seven inspirations are confirmed published optimisers. The novelty
in FusionOpt is the **composition**, the **bifurcated routing** (spectral
vs scalar groups), the **shared Polyak step size** across both groups,
and the **single inference cache abstraction** for adaLN-zero models.

The composition we ended up wanting to ship (SF-NorMuon) is precisely
arXiv:2605.23061 + arXiv:2510.05491 — we just discovered it empirically
via the per-component ablation rather than reading the SF-NorMuon paper
first. Both routes lead to the same recipe.
