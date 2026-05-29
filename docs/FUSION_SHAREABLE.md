# FusionOpt for Stable Audio (shareable digest, 2026-05-29)

A self-contained summary of today's work on FusionOpt + bf16 + d256/dp4 +
TimeConditioningCache for LatCH heads on Stable Audio Open Small.
Intended to be applied to / adapted for **Stable Audio 3**.

If you want depth, the full report is at `docs/fusion-optimiser.md` in the
LatCH repo, and the raw bake-off / ship-retrain data is in
`LATCH_RESULTS.txt` §19–§21.

---

## TL;DR — production recipe that worked

For small DiT-style heads (~5 M params, dim=256, depth=4–6, adaLN-zero
conditioning) trained on rectified-flow noise:

```bash
python scripts/train.py \
  --optimizer fusion --components ns5,normuon,sf \
  --dim 256 --depth 4 --num-heads 8 \
  --hot-dtype bf16 \
  --epochs 30 --batch-size 64 --lr 3e-4 \
  --compile --t-injection adaln_zero \
  --save-best-only
```

Versus AdamW d256/dp6 baseline on three regression heads (raw-MAE):
- `rms_energy_bass`: **−12.5 %**
- `spectral_flatness`: **−19.6 %**
- `spectral_flux`: **−12.6 %**

At **the same wall-clock per step** as AdamW (45 it/s after `--compile`).

Recipe runs on AMD RX 9070 XT (RDNA4, gfx1201), ROCm 7.2.x, PyTorch 2.10,
hipBLASLt 100202 + Triton FA. Should port to ROCm CDNA4 (MI3xx) and NVIDIA
H100 / Blackwell with minor tunings.

---

## What FusionOpt fuses

The "SF-NorMuon" subset is the load-bearing combination. It's also a real
published paper.

| Component | Mechanism | Source |
|---|---|---|
| **Muon** | Newton-Schulz quintic orthogonalisation on 2D weights (spectral-norm bound on the update). | [Keller Jordan repo](https://github.com/KellerJordan/Muon) |
| **NorMuon** | Per-neuron row-norm normalisation after NS5 (fixes uneven row magnitudes). | [arXiv:2510.05491](https://arxiv.org/abs/2510.05491) |
| **Schedule-Free** | Anytime-stopping framework: gradients at y_t = (1−β)z_t + β x_t, deploy from averaged x_t. | [Defazio, schedulefree pkg](https://github.com/facebookresearch/schedule_free) |
| **SF-NorMuon** | Schedule-Free + NorMuon + WD on the fast iterate (z_t, not x_t — load-bearing for long-horizon stability). | [arXiv:2605.23061](https://arxiv.org/abs/2605.23061) |
| **MONA** | EMA-of-gradient-differences curvature term injected into momentum. | [arXiv:2605.26842](https://arxiv.org/abs/2605.26842) |
| **KL-Shampoo** | Two-sided Kronecker covariance preconditioner via KL divergence; no Adam-grafting. | [arXiv:2509.03378](https://arxiv.org/abs/2509.03378) |
| **ScheduleFree+** | Polyak step size on top of Schedule-Free. | [arXiv:2605.19095](https://arxiv.org/abs/2605.19095) |

## Empirical findings — what to copy, what to skip

### Findings that almost certainly transfer to SA3

1. **bf16 is the right hot-path dtype for NS5; fp16 diverges.** The
   iterated quintic `X (XᵀX)²` produces intermediate values that exceed
   fp16's 65 504 ceiling on any matrix bigger than ~256×256. NaN after
   ~2 optimiser steps. bf16's fp32-equivalent exponent (8-bit) makes
   overflow impossible. The 7-bit mantissa costs ~1 % quality on
   val_point_mae — well worth the stability.

2. **SF-NorMuon (ns5+normuon+sf) captures ~95 % of the quality lift
   over AdamW.** Adding MONA and KL-Shampoo on top buys another 0.8 %
   for **50 % more wall-clock** (Shampoo's per-step covariance update
   + periodic eigendecomp). Use the smaller composition unless that
   last 0.8 % matters.

3. **Schedule-Free averaging + WD on the fast iterate z_t** is the
   stability mechanism that makes spectral updates safe over long
   horizons. WD on x_t (the average) lets z_t drift to infinity over
   thousands of steps. Tested both empirically; the SF-NorMuon paper's
   theory checks out.

4. **`--compile` is critical.** Without it, the spectral-path overhead
   dominates the per-step cost. With it, the per-step cost matches AdamW
   on the same model size. Don't run optimiser benchmarks without it.

5. **adaLN-zero block conditioning enables the TimeConditioningCache.**
   For inference at fixed step counts, t_emb + per-block modulators are
   pure functions of t and weights → cacheable. ~5–10 % inference
   latency win at 40-step rendering (see §20-aftermath in the LatCH repo).

### Findings that may or may not transfer (workload-dependent)

1. **MONA + Muon underperformed pure Muon at 5 M params** in our bake-off
   (+5.8 % vs +2.7 % over AdamW). The MONA paper validates the recipe at
   1B–68B MoE scales. At SA3 model size (presumably hundreds of millions)
   the curvature deflection may help. Test before shipping.

2. **KL-Shampoo standalone diverges to NaN** in our bake-off — the SPD
   covariance preconditioner has no spectral-norm bound, so the update
   magnitude blows up without NS5's clipping. This is the classical
   Shampoo instability the paper actually addresses; it should be
   composed with NS5 (or grafted with Adam) on any size.

3. **The d256 → d128 shrink costs more quality than the d256/dp6 → d256/dp4
   shrink.** Cut depth, not width, for inference cost. This generalises
   in spirit (depth is linear in compute, width is square) but the
   specific tradeoff numbers are workload-dependent.

### Findings that are RDNA4-specific

1. **TunableOp autotuning matters.** On our cache, the bf16 path had
   only 17 tuned shapes vs 718 fp16. We re-ran with extended autotuner
   budget (`PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=500`) — found that
   the 3 "Default" bf16 entries were actually within 1 % of any tuned
   kernel, so 26 it/s was the realistic ceiling on RDNA4.

2. **Triton FA2 (Triton AMD backend)** is on the FP16 hot path. If you're
   on NVIDIA, swap to the standard FA2 / FlashInfer kernels.

---

## Code shapes to copy

### FusionOpt component flags

The cleanest API is a `components` set that selects which mechanisms run
in the spectral path. The default is the full set; the production target
uses a subset.

```python
class FusionOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4,
                 components=None,   # subset of {"mona","shampoo","ns5","normuon","sf"}
                 hot_dtype="bf16",  # "fp32" | "fp16" | "bf16"
                 ...):
        all_comps = {"mona", "shampoo", "ns5", "normuon", "sf"}
        if components is None:
            components = all_comps
        self._components = frozenset(components)
        # ... rest of init
```

Then in the spectral step, conditionally execute each piece:

```python
if "shampoo" in self._components:
    # KL-Shampoo factor update (FP32)
    L_t = β_k L + (1-β_k) g g.T
    R_t = β_k R + (1-β_k) g.T g
    if step % K == 0:
        P_L = (L + δI)^(-1/4)
        P_R = (R + δI)^(-1/4)

if "mona" in self._components:
    A_t = β_n A + (g - g_prev); g_prev = g
    m_t = μ m + g + α A_t
else:
    m_t = μ m + g

m' = P_L @ m_t @ P_R if "shampoo" in components else m_t

# Convert to hot_dtype here for the NS5 hot path
m'_hot = m'.to(hot_dtype)

if "ns5" in self._components:
    X = m'_hot / ||m'_hot||_F
    for _ in range(5):
        A = X @ X.T
        X = 3.4445*X - 4.7750*A@X + 2.0315*(A@A)@X
    U = X * sqrt(max(1, fan_out/fan_in))
else:
    U = m'_hot

U = U.float()

if "normuon" in self._components:
    r_t = β_r r + (1-β_r) rowsumsq(U)
    U = U / sqrt(r_t.unsqueeze(-1) + ε)

if "sf" in self._components:
    z = (1 - γ·λ) z - γ U      # WD on FAST iterate
    x = (1 - 1/t) x + (1/t) z
else:
    p = (1 - γ·λ) p - γ U      # direct WD on live weights
```

The eigendecomp **must** stay in FP32 (cast L_t, R_t, do `torch.linalg.eigh`,
cast P_L / P_R back to hot_dtype for the matmul). FP16 eigendecomp on a
SPD covariance matrix produces garbage.

### TimeConditioningCache (for any adaLN-zero DiT)

```python
class TimeConditioningCache:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self._by_value: dict[float, dict] = {}  # quantised-t key -> entry

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
            t_emb = self.model.t_embedder(t_tensor)  # (1, dim)
            mods = []
            for block in self.model.blocks:
                if hasattr(block, "adaLN_mod"):
                    full = block.adaLN_mod(t_emb)
                    mods.append(tuple(full.chunk(6, dim=-1)))  # 6 × (1, dim)
                else:
                    mods.append(None)
            self._by_value[k] = {"t_emb": t_emb, "modulators": mods}

    def get(self, t):
        return self._by_value.get(round(float(t), 6))
```

Then in the model's forward, check for an attached cache:

```python
def forward(self, x, t):
    cache_entry = None
    if getattr(self, "_time_cache", None) is not None and t.numel() > 0:
        t_first = t.flatten()[0]
        if t.numel() == 1 or torch.all(t == t_first):
            cache_entry = self._time_cache.get(float(t_first.item()))

    if cache_entry is not None:
        t_emb = cache_entry["t_emb"]  # (1, dim), broadcasts across batch
        block_mods = cache_entry["modulators"]
    else:
        t_emb = self.t_embedder(t)
        block_mods = [None] * len(self.blocks)

    # ... rest of forward, passing block_mods[i] to each block
```

And in the block:

```python
def forward(self, x, t_emb, ..., mods=None):
    if mods is None:
        g1, b1, a1, g2, b2, a2 = self.adaLN_mod(t_emb).chunk(6, dim=-1)
    else:
        g1, b1, a1, g2, b2, a2 = mods
    # ... rest of block forward
```

In the sampler / inference path, instantiate and warm the cache when the
head loads:

```python
cache = TimeConditioningCache(head, device=device)
cache.warm_for_schedule(n_steps, include_zero=True)
head._time_cache = cache  # attach
```

The cache gracefully falls back to live calc on cache miss — no error path,
no quality regression.

### Param routing for the spectral / scalar split

The spectral path applies to 2D matrices ≥ 128 in both dims; everything
else (biases, LayerNorm, small projections) goes to ScheduleFree-AdamW.

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

For SA3 (assumed dim ~1024–2048, depth ~24–40), this routes basically
everything matrix-shaped through Muon NS5 and only the tiny scalar params
through Adam.

---

## Things to verify on SA3 before shipping

1. **`hot_dtype=fp16` divergence is universal.** Run a 1-cell smoke at
   fp16 — if val_loss is NaN after epoch 1, confirm bf16 instead.

2. **MONA's value at SA3 scale.** The 1B-paper recipe may help at SA3's
   model size where ours doesn't. A simple A/B (`--components ns5,normuon,sf`
   vs `--components ns5,mona,normuon,sf`) at one feature is sufficient.

3. **Depth-shrink tradeoff.** d=24→d=16 on a SA3-class model should
   buy 33 % inference cost. Test on one config to confirm the quality
   penalty is acceptable.

4. **Polyak γ clamp.** Default `gamma_max=10` may need tightening on
   workloads with bigger gradient magnitudes. Watch `fusion/gnorm_ema`
   and `fusion/loss_ema` in WandB for early-training instability.

5. **TunableOp cache populates.** First training run takes ~30 % longer
   while hipBLASLt tunes the new shapes. Bake it in as a "warmup" cell
   before any benchmark.

6. **`PYTORCH_TUNABLEOP_TUNING="1"` env var** before `import torch` — the
   common gotcha. If applied late, the env vars are ignored silently.

---

## Files to copy to SA3

If you just want code drops:

```
stable_audio_tools/training/fusion_opt.py        # FusionOpt(Optimizer)
stable_audio_tools/training/fusion_groups.py     # param routing
stable_audio_tools/inference/time_cache.py       # TimeConditioningCache
```

Plus the model-forward and block-forward adjustments shown above.

---

## Open questions

These didn't get answered in our work and would benefit from collaboration:

- **fp16 with rescale-and-restore inside NS5.** Tested and rejected on
  RDNA4: the rescale-restore overhead (~6 max-reductions × 480 matmuls/step)
  dominates the fp16 throughput win, giving 0.5× bf16 throughput. A custom
  Triton kernel with rescale fused into the matmul epilogue could reclaim
  the theoretical 1.3–1.5×, but that's ~200 LoC of real engineering, not
  a hot-fix. Whether the same conclusion holds on H100 / Blackwell isn't
  obvious — on NVIDIA the fp16 matmul vs reduction ratios differ.
- **TimeConditioningCache for non-rectified-flow schedules.** The schedule
  enumeration in `warm_for_schedule` is RF-specific. Adapt for SA3's
  noise schedule (probably similar but worth checking).
- **Phase-2 audition for SA3 heads.** Our renders/index.html UI is
  pluggable — if you produce the same `<variant>__r<ρ>_m<μ>...` filename
  scheme, the same UI works for SA3 outputs.

Ping back if any of the SA3 ports surface surprises.
