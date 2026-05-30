"""Microbench: per-component timing for FusionOpt on a real LatCH model.

Builds a depth=6 / dim=256 LatCH head, runs N=200 steps with manual
component timing via torch.cuda.Event, and reports a breakdown.

Run:  sat-venv/bin/python scripts/profile_fusion_opt.py
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from stable_audio_tools.models.latch import LatCH
from stable_audio_tools.training.fusion_opt import newton_schulz_5, _inv_quarter
from stable_audio_tools.training.fusion_groups import build_fusion_param_groups
from stable_audio_tools.training.temporal_loss import TemporalShapeLoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 200
WARMUP = 20
K_EIGEN = 100   # eigendecomp cadence


def cuda_event_timer():
    """Returns a (start, stop, sync) trio of cuda.Event-style timers."""
    if DEVICE == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        return None, None


def sync_ms(start, stop) -> float:
    """Return elapsed time in milliseconds, syncing the device first."""
    if start is not None:
        stop.synchronize()
        return start.elapsed_time(stop)
    else:
        return 0.0


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")
    print(f"N_STEPS: {N_STEPS}  WARMUP: {WARMUP}  K_EIGEN: {K_EIGEN}")
    print()

    # Build full-size LatCH (ship config)
    model = LatCH(in_channels=64, out_channels=1, dim=256, depth=6, num_heads=8,
                  t_injection="adaln_zero").to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LatCH: dim=256, depth=6, heads=8, adaln_zero  ({n_params/1e6:.2f}M params)")

    groups = build_fusion_param_groups(model, spectral_wd=0.01, scalar_wd=0.0)
    spectral_params = [(g, p) for g in groups if g["group_type"] == "spectral"
                                              for p in g["params"]]
    print(f"Spectral params (2D, min(shape)>=128): {len(spectral_params)} tensors")
    print()

    # Per-parameter state buffers (mimic FusionOpt's internal state)
    state = {}
    for _, p in spectral_params:
        out_dim, in_dim = p.shape
        state[id(p)] = {
            "m": torch.zeros_like(p, dtype=torch.float32),
            "A": torch.zeros_like(p, dtype=torch.float32),
            "g_prev": torch.zeros_like(p, dtype=torch.float32),
            "L": torch.zeros(out_dim, out_dim, device=DEVICE, dtype=torch.float32),
            "R": torch.zeros(in_dim, in_dim, device=DEVICE, dtype=torch.float32),
            "P_L": torch.eye(out_dim, device=DEVICE, dtype=torch.float32),
            "P_R": torch.eye(in_dim, device=DEVICE, dtype=torch.float32),
            "r": torch.zeros(out_dim, device=DEVICE, dtype=torch.float32),
            "z": p.detach().clone().float(),
            "x": p.detach().clone().float(),
        }

    # Dummy gradients of realistic magnitude
    def fresh_grads():
        for _, p in spectral_params:
            p.grad = torch.randn_like(p, dtype=torch.float32) * 1e-2

    # Component timers (accumulators in ms)
    totals = {
        "mona":           0.0,  # gradient diff EMA + augmented momentum
        "shampoo_factor": 0.0,  # L_t, R_t covariance updates
        "shampoo_eigen":  0.0,  # eigendecomp (every K_EIGEN steps)
        "shampoo_precon": 0.0,  # P_L @ m @ P_R
        "ns5":            0.0,  # Muon Newton-Schulz quintic
        "row_scale":      0.0,  # SF-NorMuon per-neuron scaling
        "sf_average":     0.0,  # Schedule-Free z and x updates
        "step_total":     0.0,  # full optimizer step
    }
    eigen_calls = 0

    # MONA / SF constants from FusionOpt defaults
    beta_n = 0.9
    beta_k = 0.99
    beta_r = 0.95
    mu = 0.95
    alpha = 0.2
    delta = 1e-4
    gamma_t = 3e-4
    wd = 0.01

    for step in range(N_STEPS):
        fresh_grads()

        # ---- full step timer ----
        step_start, step_stop = cuda_event_timer()
        if step >= WARMUP and step_start is not None:
            step_start.record()

        for _, p in spectral_params:
            st = state[id(p)]
            grad = p.grad

            # 1. KL-Shampoo factor update (raw gradient covariance) -------------
            s1, e1 = cuda_event_timer()
            if step >= WARMUP and s1 is not None:
                s1.record()
            L = st["L"]
            R = st["R"]
            L.mul_(beta_k).add_(grad @ grad.transpose(-1, -2), alpha=(1 - beta_k))
            R.mul_(beta_k).add_(grad.transpose(-1, -2) @ grad, alpha=(1 - beta_k))
            if step >= WARMUP and s1 is not None:
                e1.record()
                totals["shampoo_factor"] += sync_ms(s1, e1)

            # 2. KL-Shampoo eigendecomp (periodic) ------------------------------
            if step % K_EIGEN == 0 and step >= WARMUP:
                s2, e2 = cuda_event_timer()
                if s2 is not None:
                    s2.record()
                st["P_L"] = _inv_quarter(L, delta=delta)
                st["P_R"] = _inv_quarter(R, delta=delta)
                if s2 is not None:
                    e2.record()
                    totals["shampoo_eigen"] += sync_ms(s2, e2)
                    eigen_calls += 1
            elif step % K_EIGEN == 0:
                # do the eigendecomp without timing so subsequent steps have valid P
                st["P_L"] = _inv_quarter(L, delta=delta)
                st["P_R"] = _inv_quarter(R, delta=delta)

            # 3. MONA augmented momentum ----------------------------------------
            s3, e3 = cuda_event_timer()
            if step >= WARMUP and s3 is not None:
                s3.record()
            A_buf = st["A"]
            g_prev = st["g_prev"]
            A_buf.mul_(beta_n).add_(grad - g_prev)
            g_prev.copy_(grad)
            m = st["m"]
            m.mul_(mu).add_(grad + alpha * A_buf)
            if step >= WARMUP and s3 is not None:
                e3.record()
                totals["mona"] += sync_ms(s3, e3)

            # 4. KL-Shampoo preconditioner apply --------------------------------
            s4, e4 = cuda_event_timer()
            if step >= WARMUP and s4 is not None:
                s4.record()
            m_pre = st["P_L"] @ m @ st["P_R"]
            if step >= WARMUP and s4 is not None:
                e4.record()
                totals["shampoo_precon"] += sync_ms(s4, e4)

            # 5. Muon NS5 -------------------------------------------------------
            s5, e5 = cuda_event_timer()
            if step >= WARMUP and s5 is not None:
                s5.record()
            U = newton_schulz_5(m_pre)
            out_dim, in_dim = U.shape
            U = U * ((max(1.0, out_dim / in_dim)) ** 0.5)
            if step >= WARMUP and s5 is not None:
                e5.record()
                totals["ns5"] += sync_ms(s5, e5)

            # 6. SF-NorMuon row scaling -----------------------------------------
            s6, e6 = cuda_event_timer()
            if step >= WARMUP and s6 is not None:
                s6.record()
            r = st["r"]
            row_ss = (U * U).sum(dim=-1)
            r.mul_(beta_r).add_(row_ss, alpha=(1 - beta_r))
            U = U / (r.clamp_min(1e-12).sqrt().unsqueeze(-1))
            if step >= WARMUP and s6 is not None:
                e6.record()
                totals["row_scale"] += sync_ms(s6, e6)

            # 7. SF averaging ---------------------------------------------------
            s7, e7 = cuda_event_timer()
            if step >= WARMUP and s7 is not None:
                s7.record()
            z = st["z"]
            x = st["x"]
            t = step + 1
            z.mul_(1 - gamma_t * wd).add_(U, alpha=-gamma_t)
            x.mul_(1 - 1.0 / t).add_(z, alpha=1.0 / t)
            if step >= WARMUP and s7 is not None:
                e7.record()
                totals["sf_average"] += sync_ms(s7, e7)

        if step >= WARMUP and step_start is not None:
            step_stop.record()
            totals["step_total"] += sync_ms(step_start, step_stop)

    measured = N_STEPS - WARMUP
    print(f"Measured over {measured} steps (after {WARMUP}-step warmup):")
    print()
    print(f"  {'Component':<22} {'Total (ms)':>11} {'Per step (ms)':>14} {'% step':>8}")
    print(f"  {'-'*22} {'-'*11} {'-'*14} {'-'*8}")
    per_step_total = totals["step_total"] / measured if measured else 1.0
    # We want one-step totals: divide by measured steps for everything EXCEPT
    # shampoo_eigen, which only ran on K_EIGEN-th steps; report its per-eigen-call cost.
    components_per_step = {
        "mona":           totals["mona"]           / measured,
        "shampoo_factor": totals["shampoo_factor"] / measured,
        "shampoo_eigen":  (totals["shampoo_eigen"] / eigen_calls) if eigen_calls else 0,
        "shampoo_precon": totals["shampoo_precon"] / measured,
        "ns5":            totals["ns5"]            / measured,
        "row_scale":      totals["row_scale"]      / measured,
        "sf_average":     totals["sf_average"]     / measured,
    }
    # Eigen amortised over K_EIGEN steps
    amortised_eigen = components_per_step["shampoo_eigen"] / K_EIGEN
    components_amortised = dict(components_per_step)
    components_amortised["shampoo_eigen"] = amortised_eigen

    for name, val in components_amortised.items():
        pct = 100 * val / per_step_total if per_step_total > 0 else 0
        suffix = f"(per-eigen={components_per_step['shampoo_eigen']:.2f} ms, amort. /{K_EIGEN})" \
                 if name == "shampoo_eigen" else ""
        print(f"  {name:<22} {val * measured:>11.2f} {val:>14.4f} {pct:>7.1f}% {suffix}")
    print(f"  {'-'*22} {'-'*11} {'-'*14} {'-'*8}")
    print(f"  {'step_total':<22} {totals['step_total']:>11.2f} {per_step_total:>14.4f}   100.0%")
    print()

    # Sanity: per_step_total should ≈ sum of components (within timer overhead)
    components_sum = sum(components_amortised.values())
    print(f"Sum of components:  {components_sum:.4f} ms / step")
    print(f"Measured step total: {per_step_total:.4f} ms / step")
    print(f"Timer overhead / unaccounted: {per_step_total - components_sum:+.4f} ms / step")
    print()

    # Per-layer breakdown: which spectral params are most expensive?
    # (separate sweep — time the most expensive components per layer shape)
    print()
    print("Per-shape breakdown (single-step cost for L_t update + eigen + precon):")
    print(f"  {'Shape':>14}  {'L_t update (ms)':>16}  {'eigen P_L (ms)':>16}  {'precon (ms)':>13}")
    shape_costs = {}
    for _, p in spectral_params:
        out_dim, in_dim = p.shape
        key = (out_dim, in_dim)
        if key in shape_costs:
            continue
        # Time L_t update on this shape (single call, no warmup needed beyond above)
        g = torch.randn(out_dim, in_dim, device=DEVICE, dtype=torch.float32) * 1e-2
        L = torch.zeros(out_dim, out_dim, device=DEVICE, dtype=torch.float32)
        s, e = cuda_event_timer()
        if s is not None: s.record()
        for _ in range(10):
            L.mul_(beta_k).add_(g @ g.transpose(-1, -2), alpha=(1 - beta_k))
        if s is not None:
            e.record(); l_ms = sync_ms(s, e) / 10
        else: l_ms = 0

        s, e = cuda_event_timer()
        if s is not None: s.record()
        for _ in range(5):
            _inv_quarter(L, delta=delta)
        if s is not None:
            e.record(); eig_ms = sync_ms(s, e) / 5
        else: eig_ms = 0

        m = g
        PL = torch.eye(out_dim, device=DEVICE, dtype=torch.float32)
        PR = torch.eye(in_dim, device=DEVICE, dtype=torch.float32)
        s, e = cuda_event_timer()
        if s is not None: s.record()
        for _ in range(10):
            _ = PL @ m @ PR
        if s is not None:
            e.record(); pre_ms = sync_ms(s, e) / 10
        else: pre_ms = 0

        shape_costs[key] = (l_ms, eig_ms, pre_ms)
        print(f"  {str(key):>14}  {l_ms:>16.4f}  {eig_ms:>16.4f}  {pre_ms:>13.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
