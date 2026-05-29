"""DiversityLoss — train a head to be different from a frozen reference.

Wraps any base criterion (SmoothL1, TemporalShapeLoss, ...) and adds a penalty
that REWARDS divergence from a reference head's predictions on the same input.

  L_total = L_task(pred, target)  -  λ_div(t) · min(MSE(pred, ref_pred), bound)

The negative sign means optimizing minimises L_total -> drives MSE between
new-head and reference-head predictions UP. The min() bound prevents the
divergence term from running away to infinity.

Optional linear warmup: λ_div starts at 0 and ramps to its target value over
the first `warmup_steps` training steps. Useful when the new head starts
from random init — let the task loss find SOMETHING coherent first, then
push it away from the reference.

For warm-start (init the new head from the reference checkpoint), warmup
isn't needed: the initial MSE is ~0, the divergence penalty is ~0, and the
task loss + drift naturally moves the new head into a different basin.

The reference head is loaded once, frozen, and kept in eval mode. Its
parameters never receive gradients. Memory cost: one extra LatCH head
on-device (~5-7 M params).

Spec considerations:
- Sets `self.wants_context = True` so the training loop knows to call
  `criterion.set_context(z, t)` before forward(pred, target). The reference
  head needs (z, t) to compute ref_pred.
- For composite losses (e.g. TemporalShapeLoss), L_task includes whatever
  components were configured; diversity is added on top.
- Diagnostic in last_components: includes "L_task", "L_div", "lambda_curr".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(nn.Module):
    """L_task + diversity penalty against a frozen reference head."""

    wants_context = True  # signals to the training loop to call set_context(z, t)

    def __init__(
        self,
        base_criterion: nn.Module,
        ref_head: nn.Module,
        lambda_div: float = 0.3,
        warmup_steps: int = 0,
        mse_bound: float = 100.0,
        use_standardized: bool = False,
        std_mean: float = 0.0,
        std_std: float = 1.0,
    ):
        """
        Args:
            base_criterion: any pred-vs-target loss (SmoothL1, TemporalShapeLoss, …).
            ref_head: a frozen LatCH module. Set to eval(); params .requires_grad=False
                here so the caller doesn't have to.
            lambda_div: target weight on the diversity penalty.
            warmup_steps: linear ramp from 0 -> lambda_div over this many train steps.
                0 = no warmup (immediate full strength). Use >0 for fresh-init runs;
                use 0 for warm-start-from-ref runs.
            mse_bound: cap on MSE(pred, ref_pred) before negating. Prevents the
                divergence term from running away to infinity. Pick relative to
                the feature's natural variance.
            use_standardized: if True, ref_pred is standardized like the new head's
                pred. (Match the training-time standardization of the new head.)
            std_mean, std_std: the standardization params used by the new head.
        """
        super().__init__()
        self.base = base_criterion
        self.ref = ref_head.eval()
        for p in self.ref.parameters():
            p.requires_grad = False
        self.target_lambda = float(lambda_div)
        self.warmup_steps = int(warmup_steps)
        self.mse_bound = float(mse_bound)
        self.use_standardized = bool(use_standardized)
        self.std_mean = float(std_mean)
        self.std_std = float(std_std)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.last_components: dict[str, torch.Tensor] = {}
        # Set via set_context() each batch from the training loop
        self._ctx_z = None
        self._ctx_t = None

    def set_context(self, z: torch.Tensor, t: torch.Tensor) -> None:
        """Stash the batch's noisy latent + timestep so forward() can run the ref."""
        self._ctx_z = z
        self._ctx_t = t

    def lambda_curr(self) -> float:
        if self.warmup_steps <= 0:
            return self.target_lambda
        step = int(self._step.item()) if self._step.numel() else 0
        if step >= self.warmup_steps:
            return self.target_lambda
        return self.target_lambda * (step / self.warmup_steps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        L_task = self.base(pred, target)

        # If no context (e.g. validation), report only the task loss; diversity
        # is a training-time pressure, not a measurement objective.
        if self._ctx_z is None or self._ctx_t is None:
            self.last_components = {"L_task": L_task.detach()}
            return L_task

        # Reference forward (frozen, no grad). The ref runs in whatever dtype
        # autocast is currently using; values are fine in fp16 since this is
        # just a target for MSE.
        with torch.no_grad():
            ref_pred = self.ref(self._ctx_z, self._ctx_t)
        if self.use_standardized:
            # Match the standardization of the new head's pred
            ref_pred = (ref_pred - self.std_mean) / self.std_std

        # Bounded divergence: model is encouraged to disagree up to mse_bound.
        mse = F.mse_loss(pred, ref_pred.detach())
        L_div_unbounded = -mse                 # negative => big MSE shrinks total loss
        L_div_bounded = torch.clamp(L_div_unbounded, min=-self.mse_bound)
        lam = self.lambda_curr()

        self.last_components = {
            "L_task": L_task.detach(),
            "L_div_raw_mse": mse.detach(),
            "L_div_contrib": (lam * L_div_bounded).detach(),
            "lambda_curr": torch.tensor(lam, device=pred.device),
        }

        if self.training:
            self._step += 1

        return L_task + lam * L_div_bounded


def load_reference_head(ckpt_path: str, device: str | torch.device = "cuda") -> nn.Module:
    """Load a LatCH checkpoint as a frozen reference, in eval mode on `device`."""
    from ..models.latch import load_latch_from_checkpoint
    ref = load_latch_from_checkpoint(ckpt_path, device=str(device))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref
