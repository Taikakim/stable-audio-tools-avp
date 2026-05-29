"""FusionOpt — Muon + MONA + KL-Shampoo + ScheduleFree+ optimiser.

Bifurcated optimiser for LatCH heads:

- Spectral group (2D matrices, min(shape) >= 128):
    1. KL-Shampoo two-sided Kronecker covariance from RAW gradient
    2. MONA curvature-augmented momentum
    3. KL-Shampoo preconditioner applied to augmented momentum
    4. Muon Newton-Schulz quintic spectral normalisation
    5. SF-NorMuon per-neuron row-norm scaling
    6. Schedule-Free averaging with weight decay on the FAST iterate z_t

- Scalar group (1D params, biases, LayerNorm, small/odd 2D):
    Standard ScheduleFree-AdamW with shared Polyak step size.

Outer loop:
    Polyak step size gamma_t = gamma_base * clamp(loss_ema / gnorm_ema, 0.1, 10).
    All reductions on-device; no host syncs in the hot loop.

Train / eval semantics:
    - optimizer.train() writes the Schedule-Free eval point y = (1-beta)z + beta*x
      into params for the next forward.
    - optimizer.eval() writes the averaged iterate x into params for validation
      and checkpoint serialisation.

See docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md §3.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim import Optimizer


# ---------- Newton-Schulz quintic (Muon) ----------

_NS5_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-12) -> torch.Tensor:
    """Muon's NS5 orthogonalisation.

    Frobenius-normalises G so its singular values land in [0, 1], then runs
    `steps` iterations of the quintic phi(X) = a*X + b*X(X^T X) + c*X(X^T X)^2
    with coefficients (3.4445, -4.7750, 2.0315). After 5 iterations the SVs
    are pulled into approximately [0.7, 1.3].

    For tall matrices (rows > cols) we transpose to keep the inner X @ X^T
    small (cols x cols rather than rows x rows). Result is transposed back.
    """
    a, b, c = _NS5_COEFFS
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(-1, -2).contiguous()
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X


# ---------- KL-Shampoo SPD inverse quarter ----------

def _inv_quarter(M: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    """Compute (M + delta*I)^(-1/4) via eigendecomposition.

    Runs entirely in FP32 (the "FP32 island" per spec §3). Caller is
    expected to downcast the result for the hot-path matmul if desired.
    M must be square, symmetric, on the same device as the model.
    """
    M = M.float()
    n = M.shape[-1]
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    Mr = 0.5 * (M + M.transpose(-1, -2)) + delta * eye  # symmetrise + ridge
    eigvals, eigvecs = torch.linalg.eigh(Mr)
    inv_q = eigvals.clamp_min(1e-12).pow(-0.25)
    return eigvecs @ torch.diag_embed(inv_q) @ eigvecs.transpose(-1, -2)


# ---------- FusionOpt ----------

class FusionOpt(Optimizer):
    """The fused optimiser. Takes param groups from build_fusion_param_groups."""

    def __init__(
        self,
        params: Iterable[dict],
        lr: float = 3e-4,
        # Schedule-Free / Polyak
        beta: float = 0.9,
        beta_p: float = 0.98,
        gamma_min: float = 0.1,
        gamma_max: float = 10.0,
        # Spectral path
        mu: float = 0.95,
        mona_alpha: float = 0.2,
        beta_n: float = 0.9,
        beta_k: float = 0.99,
        eigen_period: int = 100,
        shampoo_delta: float = 1e-4,
        beta_r: float = 0.95,
        # Scalar path
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        # Weight decay (set per-group via param_groups; these are fallbacks)
        weight_decay: float = 0.0,
        # Hot-path dtype: cast preconditioned momentum + NS5 inputs to this dtype
        # (kept FP32 by default for safety; FP16 on RDNA4 if profiling shows need)
        hot_dtype: str = "fp32",
        warmup_steps: int = 0,
        # Component flags — controls which mechanisms run in the spectral path.
        # Use this for the per-component ablation (one optimiser at a time).
        # Default = full Fusion (everything on). Valid components:
        #   "mona"     - MONA curvature-augmented momentum
        #   "shampoo"  - KL-Shampoo two-sided preconditioner
        #   "ns5"      - Muon Newton-Schulz quintic spectral norm
        #   "normuon"  - SF-NorMuon per-neuron row-norm scaling
        #   "sf"       - Schedule-Free averaging (z_t fast iterate + x_t average,
        #                Polyak step, WD on z_t). When disabled, weight decay
        #                applies to live weights p directly.
        components: "set[str] | None" = None,
    ):
        all_components = {"mona", "shampoo", "ns5", "normuon", "sf"}
        if components is None:
            components = all_components
        components = set(components)
        unknown = components - all_components
        if unknown:
            raise ValueError(f"FusionOpt: unknown components: {sorted(unknown)} "
                             f"(valid: {sorted(all_components)})")
        defaults = dict(
            lr=lr,
            beta=beta, beta_p=beta_p, gamma_min=gamma_min, gamma_max=gamma_max,
            mu=mu, mona_alpha=mona_alpha, beta_n=beta_n, beta_k=beta_k,
            eigen_period=eigen_period, shampoo_delta=shampoo_delta, beta_r=beta_r,
            beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay,
            hot_dtype=hot_dtype, warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

        self._components = frozenset(components)
        self._mode = "train"  # "train" or "eval"
        self._step_count = 0
        # Pending loss for Polyak (set by train loop before step)
        self._current_loss: torch.Tensor | None = None
        # On-device EMAs; lazily created on first step()
        self._loss_ema: torch.Tensor | None = None
        self._gnorm_ema: torch.Tensor | None = None

        # Sanity: groups must declare group_type
        for g in self.param_groups:
            if g.get("group_type") not in ("spectral", "scalar"):
                raise ValueError(
                    "FusionOpt param groups must set group_type=spectral|scalar"
                )

    @property
    def uses_sf_averaging(self) -> bool:
        """Whether this optimiser uses Schedule-Free averaging (the .train()/.eval()
        toggle behaviour). False when 'sf' is excluded from components."""
        return "sf" in self._components

    @property
    def components(self) -> "frozenset[str]":
        """Currently enabled components. Useful for logging / introspection."""
        return self._components

    # ---- public ---------------------------------------------------------

    def set_loss(self, loss: torch.Tensor | None) -> None:
        """Pass the most recent training loss for the Polyak step.

        Call this BEFORE optimizer.step() (or before scaler.step(optimizer) when
        using GradScaler). The loss tensor is detached and kept on-device.
        If never called, Polyak falls back to a constant ratio of 1.0.
        """
        self._current_loss = loss.detach() if loss is not None else None

    def train(self) -> None:
        """Switch params to the eval-point y = (1-beta)*z + beta*x.
        No-op when SF averaging is disabled (live weights are always in p)."""
        if self._mode == "train":
            return
        self._mode = "train"
        if "sf" not in self._components:
            return
        for group in self.param_groups:
            beta = group["beta"]
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is None or "z" not in st:
                    continue
                p.data.copy_((1 - beta) * st["z"] + beta * st["x"])

    def eval(self) -> None:
        """Switch params to the averaged iterate x (deployable model).
        No-op when SF averaging is disabled."""
        if self._mode == "eval":
            return
        self._mode = "eval"
        if "sf" not in self._components:
            return
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, None)
                if st is None or "x" not in st:
                    continue
                p.data.copy_(st["x"])

    def average_state_dict(self) -> dict[str, torch.Tensor]:
        """Return {param_name: x_t} for serialisation as a deployable model.
        When SF averaging is disabled, returns the live weights (which ARE the
        deployable model in that case)."""
        out: dict[str, torch.Tensor] = {}
        for group in self.param_groups:
            names = group.get("param_names", [])
            for p, name in zip(group["params"], names):
                st = self.state.get(p, None)
                if "sf" not in self._components or st is None or "x" not in st:
                    out[name] = p.detach().clone()
                else:
                    out[name] = st["x"].detach().clone()
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if self._current_loss is None and loss is not None:
                self._current_loss = loss.detach()

        # Polyak step size (global, on-device)
        gamma_ratio = self._update_polyak()

        for group in self.param_groups:
            if group["group_type"] == "spectral":
                self._spectral_group_step(group, gamma_ratio)
            else:
                self._scalar_group_step(group, gamma_ratio)

        # After updating z and x, write y back to p.data for next forward.
        # Skipped when SF averaging is disabled: live weights are already in p.
        if self._mode == "train" and "sf" in self._components:
            for group in self.param_groups:
                beta = group["beta"]
                for p in group["params"]:
                    st = self.state.get(p, None)
                    if st is None or "z" not in st:
                        continue
                    p.data.copy_((1 - beta) * st["z"] + beta * st["x"])

        self._step_count += 1
        self._current_loss = None
        return loss

    # ---- internals ------------------------------------------------------

    def _update_polyak(self) -> torch.Tensor:
        """Update loss/gnorm EMAs and return gamma_ratio = clamp(L/G, gamma_min, gamma_max).

        Returned as a scalar tensor on the device of the first available grad,
        so downstream multiplications stay GPU-resident.
        """
        # Find a reference device from gradients
        device = None
        total_abs = None
        total_count = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if device is None:
                    device = p.grad.device
                ga = p.grad.detach().abs().sum()
                total_abs = ga if total_abs is None else total_abs + ga
                total_count += p.grad.numel()

        if device is None or total_count == 0:
            # No grads — return identity ratio
            return torch.tensor(1.0)

        gnorm_now = total_abs / total_count
        beta_p = self.param_groups[0]["beta_p"]

        if self._gnorm_ema is None:
            self._gnorm_ema = gnorm_now.clone().detach()
        else:
            self._gnorm_ema = (
                self._gnorm_ema.to(device) * beta_p + gnorm_now * (1 - beta_p)
            )

        if self._current_loss is not None:
            loss_now = self._current_loss.to(device).float()
            if self._loss_ema is None:
                self._loss_ema = loss_now.clone().detach()
            else:
                self._loss_ema = self._loss_ema.to(device) * beta_p + loss_now * (1 - beta_p)
            ratio = self._loss_ema / (self._gnorm_ema + 1e-12)
        else:
            # No loss provided — degenerate to constant 1.0
            ratio = torch.ones((), device=device)

        gmin = self.param_groups[0]["gamma_min"]
        gmax = self.param_groups[0]["gamma_max"]
        return ratio.clamp(gmin, gmax)

    # ---- spectral path --------------------------------------------------

    def _spectral_group_step(self, group, gamma_ratio):
        lr = group["lr"]
        beta = group["beta"]
        mu = group["mu"]
        alpha = group["mona_alpha"]
        beta_n = group["beta_n"]
        beta_k = group["beta_k"]
        beta_r = group["beta_r"]
        eigen_period = group["eigen_period"]
        delta = group["shampoo_delta"]
        wd = group["weight_decay"]
        hot_dtype_name = group["hot_dtype"]
        warmup = group.get("warmup_steps", 0)

        # Warmup factor (linear ramp 0 -> 1 over `warmup` steps)
        if warmup > 0 and self._step_count < warmup:
            warm = (self._step_count + 1) / warmup
        else:
            warm = 1.0

        gamma_t = lr * gamma_ratio * warm

        hot_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(hot_dtype_name, torch.float32)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.detach().float()
            if grad.ndim != 2:
                # Should not happen — fusion_groups guarantees spectral params are 2D
                continue

            state = self.state[p]
            if "z" not in state:
                # Lazy init: copy current p as both z and x (eval point matches).
                state["z"] = p.detach().clone().float()
                state["x"] = p.detach().clone().float()
                state["m"] = torch.zeros_like(grad)           # momentum
                state["A"] = torch.zeros_like(grad)           # MONA curvature EMA
                state["g_prev"] = torch.zeros_like(grad)      # last gradient
                out_dim, in_dim = grad.shape
                state["L"] = torch.zeros(out_dim, out_dim, device=grad.device, dtype=torch.float32)
                state["R"] = torch.zeros(in_dim, in_dim, device=grad.device, dtype=torch.float32)
                state["P_L"] = torch.eye(out_dim, device=grad.device, dtype=torch.float32)
                state["P_R"] = torch.eye(in_dim, device=grad.device, dtype=torch.float32)
                state["r"] = torch.zeros(out_dim, device=grad.device, dtype=torch.float32)
                state["step"] = 0

            # 1. KL-Shampoo factor update from RAW gradient (every step, FP32)
            if "shampoo" in self._components:
                L = state["L"]
                R = state["R"]
                L.mul_(beta_k).add_(grad @ grad.transpose(-1, -2), alpha=(1 - beta_k))
                R.mul_(beta_k).add_(grad.transpose(-1, -2) @ grad, alpha=(1 - beta_k))

                # 2. Periodic eigendecomp -> P_L, P_R (every K steps, FP32 island)
                if self._step_count % eigen_period == 0:
                    state["P_L"] = _inv_quarter(L, delta=delta)
                    state["P_R"] = _inv_quarter(R, delta=delta)

            # 3. MONA augmented momentum (FP32) — optional
            m = state["m"]
            if "mona" in self._components:
                A_buf = state["A"]
                g_prev = state["g_prev"]
                A_buf.mul_(beta_n).add_(grad - g_prev)
                g_prev.copy_(grad)
                m.mul_(mu).add_(grad + alpha * A_buf)
            else:
                # Plain momentum
                m.mul_(mu).add_(grad)

            # 4. Apply KL-Shampoo preconditioner (optional)
            if "shampoo" in self._components:
                P_L_h = state["P_L"].to(hot_dtype)
                P_R_h = state["P_R"].to(hot_dtype)
                m_h = m.to(hot_dtype)
                m_pre = P_L_h @ m_h @ P_R_h
            else:
                m_pre = m.to(hot_dtype)

            # 5. Muon NS5 spectral normalisation (optional)
            if "ns5" in self._components:
                U = newton_schulz_5(m_pre).float()
                out_dim, in_dim = U.shape
                U.mul_((max(1.0, out_dim / in_dim)) ** 0.5)  # aspect-ratio scale
            else:
                U = m_pre.float()

            # 6. SF-NorMuon per-neuron row scaling (optional)
            if "normuon" in self._components:
                r = state["r"]
                row_ss = (U * U).sum(dim=-1)  # (out_dim,)
                r.mul_(beta_r).add_(row_ss, alpha=(1 - beta_r))
                U = U / (r.clamp_min(1e-12).sqrt().unsqueeze(-1))

            # 7. Update — Schedule-Free averaging (with WD on z_t) OR direct on p
            t = state["step"] + 1
            state["step"] = t
            if "sf" in self._components:
                z = state["z"]
                x = state["x"]
                # z_{t+1} = (1 - gamma*wd) z_t - gamma * U
                z.mul_(1 - gamma_t * wd).add_(U, alpha=-gamma_t)
                # x_{t+1} = (1 - 1/t) x_t + (1/t) z_{t+1}
                x.mul_(1 - 1.0 / t).add_(z, alpha=1.0 / t)
            else:
                # No SF averaging: apply WD + step directly to live weights p
                p.data.mul_(1 - gamma_t * wd).add_(U.to(p.dtype), alpha=-gamma_t)

    # ---- scalar path ----------------------------------------------------

    def _scalar_group_step(self, group, gamma_ratio):
        lr = group["lr"]
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        eps = group["eps"]
        wd = group["weight_decay"]
        warmup = group.get("warmup_steps", 0)

        if warmup > 0 and self._step_count < warmup:
            warm = (self._step_count + 1) / warmup
        else:
            warm = 1.0

        gamma_t = lr * gamma_ratio * warm

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            state = self.state[p]
            if "z" not in state:
                state["z"] = p.detach().clone().float()
                state["x"] = p.detach().clone().float()
                state["m"] = torch.zeros_like(grad)
                state["v"] = torch.zeros_like(grad)
                state["step"] = 0

            t = state["step"] + 1
            state["step"] = t

            m = state["m"]
            v = state["v"]
            m.mul_(beta1).add_(grad, alpha=(1 - beta1))
            v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

            bias1 = 1 - beta1 ** t
            bias2 = 1 - beta2 ** t
            m_hat = m / bias1
            v_hat = v / bias2
            u = m_hat / (v_hat.sqrt() + eps)

            if "sf" in self._components:
                z = state["z"]
                x = state["x"]
                z.mul_(1 - gamma_t * wd).add_(u, alpha=-gamma_t)
                x.mul_(1 - 1.0 / t).add_(z, alpha=1.0 / t)
            else:
                # No SF averaging: plain AdamW step on live weights p
                p.data.mul_(1 - gamma_t * wd).add_(u.to(p.dtype), alpha=-gamma_t)

    # ---- diagnostic -----------------------------------------------------

    def diagnostic_summary(self) -> dict[str, Any]:
        """Return a small dict of optimizer-level metrics for WandB logging."""
        out = {
            "fusion/step_count": self._step_count,
            "fusion/mode": 0 if self._mode == "train" else 1,
        }
        if self._gnorm_ema is not None:
            out["fusion/gnorm_ema"] = float(self._gnorm_ema.detach().cpu())
        if self._loss_ema is not None:
            out["fusion/loss_ema"] = float(self._loss_ema.detach().cpu())
        return out
