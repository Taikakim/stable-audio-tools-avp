"""TemporalShapeLoss — three-component loss for LatCH heads.

Augments the existing point-wise SmoothL1 (or cosine for multi-channel hpcp)
with a derivative term (penalises wrong direction of events) and a multi-scale
L1 term over octave-spaced temporal downsamples (penalises wrong shape of
superstructures: phrases, bars, sections).

See docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md §4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SCALES = (2, 4, 8, 16, 32, 64, 128, 256)


class TemporalShapeLoss(nn.Module):
    """L_total = L_point + scale * (lambda_d * L_deriv + lambda_m * L_multi).

    All components operate on (B, F, T) tensors.

    Args:
        huber_beta: SmoothL1 knee for scalar features. For hpcp (or any
            point_loss="cosine") this is ignored on L_point but still used
            for L_deriv and L_multi (SmoothL1 on diff/pooled vectors).
        lambda_deriv: weight on L_deriv. Default 1.0 — direction is first-class.
        lambda_multi: weight on L_multi. Default 0.5 — supports rather than
            dominates (8 scales averaged would otherwise be too strong).
        scales: temporal scales for L_multi. Default octave-spaced 2..256.
        point_loss: "auto" (cosine for F>1, SmoothL1 for F=1),
                    "smooth_l1" (always SmoothL1),
                    "cosine" (always 1-cosine over channels).
        curriculum_steps: linear warmup of (lambda_d, lambda_m) from 0 to
            their defaults over this many steps. 0 = off (constant from step 1).
    """

    def __init__(
        self,
        huber_beta: float = 1.0,
        lambda_deriv: float = 1.0,
        lambda_multi: float = 0.5,
        scales: tuple[int, ...] = DEFAULT_SCALES,
        point_loss: str = "auto",
        curriculum_steps: int = 0,
    ):
        super().__init__()
        self.huber_beta = float(huber_beta)
        self.lambda_deriv = float(lambda_deriv)
        self.lambda_multi = float(lambda_multi)
        self.scales = tuple(int(s) for s in scales)
        self.point_loss = str(point_loss)
        self.curriculum_steps = int(curriculum_steps)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.last_components: dict[str, torch.Tensor] = {}

    def _curriculum_scale(self) -> float:
        if self.curriculum_steps <= 0:
            return 1.0
        s = int(self._step.item()) if self._step.numel() else 0
        return min(1.0, s / float(self.curriculum_steps))

    def _point_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, F, T)
        mode = self.point_loss
        if mode == "auto":
            mode = "cosine" if pred.shape[1] > 1 else "smooth_l1"
        if mode == "cosine":
            # 1 - mean_t mean_b cos(pred[t], target[t]) over channels
            cos = F.cosine_similarity(pred, target, dim=1)  # (B, T)
            return (1.0 - cos).mean()
        return F.smooth_l1_loss(pred, target, beta=self.huber_beta)

    def _derivative_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_pred = pred[..., 1:] - pred[..., :-1]
        diff_target = target[..., 1:] - target[..., :-1]
        return F.smooth_l1_loss(diff_pred, diff_target, beta=self.huber_beta)

    def _multiscale_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # avg_pool1d needs (B*F, 1, T) or (B, F, T) — F.avg_pool1d takes (N, C, L).
        # Our tensor is already (B, F, T) which matches.
        T = pred.shape[-1]
        losses = []
        for s in self.scales:
            if s <= 1 or s > T:
                continue
            p_s = F.avg_pool1d(pred, kernel_size=s, stride=s)
            t_s = F.avg_pool1d(target, kernel_size=s, stride=s)
            losses.append(F.smooth_l1_loss(p_s, t_s, beta=self.huber_beta))
        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        L_point = self._point_loss(pred, target)
        L_deriv = self._derivative_loss(pred, target)
        L_multi = self._multiscale_loss(pred, target)
        scale = self._curriculum_scale()

        self.last_components = {
            "L_point": L_point.detach(),
            "L_deriv": L_deriv.detach(),
            "L_multi": L_multi.detach(),
            "curriculum_scale": torch.tensor(scale, device=pred.device),
        }

        if self.training:
            self._step += 1

        return L_point + scale * (self.lambda_deriv * L_deriv + self.lambda_multi * L_multi)


def val_diagnostic_metrics(
    pred: torch.Tensor, target: torch.Tensor,
    scales: tuple[int, ...] = DEFAULT_SCALES,
) -> dict[str, torch.Tensor]:
    """Compute the raw-unit diagnostic metrics for logging.

    Returns a dict with:
      - val_point_mae:      mean abs error in raw feature units
      - val_deriv_corr:     Pearson correlation between diff(pred) and diff(target)
      - val_multiscale_mae: mean abs error averaged across temporal scales

    All comparable across loss functions; safe to log even when training
    with SmoothL1 baseline so we have a single yardstick across the bake-off.
    """
    pred = pred.detach().float()
    target = target.detach().float()

    val_point_mae = (pred - target).abs().mean()

    diff_p = pred[..., 1:] - pred[..., :-1]
    diff_t = target[..., 1:] - target[..., :-1]
    # Pearson over flattened (B, F, T-1) — captures overall direction agreement
    dp = (diff_p - diff_p.mean()).flatten()
    dt = (diff_t - diff_t.mean()).flatten()
    denom = (dp.norm() * dt.norm()).clamp_min(1e-12)
    val_deriv_corr = (dp @ dt) / denom

    T = pred.shape[-1]
    multi_losses = []
    for s in scales:
        if s <= 1 or s > T:
            continue
        p_s = F.avg_pool1d(pred, kernel_size=s, stride=s)
        t_s = F.avg_pool1d(target, kernel_size=s, stride=s)
        multi_losses.append((p_s - t_s).abs().mean())
    val_multiscale_mae = (
        torch.stack(multi_losses).mean() if multi_losses else pred.new_zeros(())
    )

    return {
        "val_point_mae": val_point_mae,
        "val_deriv_corr": val_deriv_corr,
        "val_multiscale_mae": val_multiscale_mae,
    }
