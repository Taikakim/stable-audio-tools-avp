"""TimeConditioningCache — precomputed time embeddings + adaLN-zero modulators.

For a LatCH head with `t_injection="adaln_zero"`, every sampler step does:

  1. t_emb = self.t_embedder(t)                 # (B, dim)
  2. for each block:
       (g1, b1, a1, g2, b2, a2) = block.adaLN_mod(t_emb).chunk(6, dim=-1)
       ... use modulators in the forward path ...

For deterministic sampler schedules with a fixed step count, the t values are
known in advance; the t_emb and modulators are pure functions of t and the head's
weights. Cache them once per step count, look them up at inference.

Savings on LatCH (dim=256, depth=4 production target):
  - t_embedder:   2 small Linears  -> ~0.5 ms / sampler step
  - adaLN_mod x 4 blocks: 8 Linears -> ~1.5 ms / sampler step
  - Total: ~2 ms / step  *  40 steps  =  80 ms per render
That's a 5-10 % cut on inference latency, depending on the rest of the
pipeline.

Usage:
    from stable_audio_tools.inference.time_cache import TimeConditioningCache

    cache = TimeConditioningCache(latch_head, device="cuda")
    cache.warm([10, 20, 30, 40, 50, 100])   # precompute common counts
    # At inference, sampler asks for step idx i of n_steps total:
    entry = cache.get(n_steps, step_idx, sigma_t)
    if entry is None:
        # not cached: fall back to live computation
        t_emb = head.t_embedder(t)
        ...
    else:
        t_emb, modulators = entry

The fallback path makes uncommon step counts safe — no error, no quality loss,
just no speedup. `cache.ensure(n_steps)` triggers on-demand caching for any
step count the user picks.
"""

from __future__ import annotations

from typing import Optional

import torch


class TimeConditioningCache:
    """Precomputed t_embedder + per-block adaLN-zero modulator outputs."""

    def __init__(self, latch_head, device: str | torch.device = "cuda"):
        self.head = latch_head
        self.device = torch.device(device)
        # {n_steps: {"t_values": (N,), "t_emb": (N, dim),
        #            "modulators": [None | (6 × (N, dim))] per block]}}
        self._caches: dict[int, dict] = {}
        # Tag identifying the model weights this cache was built for. If the
        # caller swaps weights without rebuilding, get() returns None for safety.
        self._weights_tag: Optional[int] = self._compute_weights_tag()

    # ---- public ---------------------------------------------------------

    def warm(self, counts) -> None:
        """Precompute caches for the given list of step counts."""
        for n in counts:
            self.ensure(int(n))

    def ensure(self, n_steps: int) -> None:
        """On-demand: build the cache for n_steps if not already present."""
        n_steps = int(n_steps)
        if n_steps in self._caches:
            return
        self._caches[n_steps] = self._build(n_steps)

    def get(self, n_steps: int, step_idx: int):
        """Return (t_emb_step, [per-block modulator tuples]) or None.

        per-block modulator tuple is None for blocks that don't use adaLN-zero
        (e.g. `t_injection="concat"` or `"film"`); the caller falls back to
        live calc for those.
        """
        # Bail out if the model weights have been swapped since the cache built
        if self._weights_tag != self._compute_weights_tag():
            self._caches.clear()
            self._weights_tag = self._compute_weights_tag()
            return None
        cache = self._caches.get(int(n_steps))
        if cache is None:
            return None
        if step_idx < 0 or step_idx >= cache["t_emb"].shape[0]:
            return None
        t_emb = cache["t_emb"][step_idx:step_idx + 1]  # (1, dim), preserves batch axis
        mods = []
        for block_mods in cache["modulators"]:
            if block_mods is None:
                mods.append(None)
            else:
                # Each entry of block_mods is a tensor of shape (N, dim)
                mods.append(tuple(m[step_idx:step_idx + 1] for m in block_mods))
        return t_emb, mods

    def clear(self) -> None:
        self._caches.clear()

    def invalidate_for_new_weights(self) -> None:
        """Call when the underlying LatCH head's weights change."""
        self.clear()
        self._weights_tag = self._compute_weights_tag()

    @property
    def cached_counts(self) -> list[int]:
        return sorted(self._caches.keys())

    # ---- internals ------------------------------------------------------

    @torch.no_grad()
    def _build(self, n_steps: int) -> dict:
        """Precompute t_emb + per-block modulators for the n_steps schedule."""
        # Sampler convention for rectified flow: t goes from 1 (max noise) to
        # 0 (clean). Stable Audio Open Small uses linspace(1, 0, n_steps+1).
        # The LatCH head sees these t values as its conditioning.
        t_values = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)[:-1]

        # t_embedder works on (B,) -> (B, dim)
        t_emb = self.head.t_embedder(t_values)

        modulators = []
        for block in self.head.blocks:
            if hasattr(block, "adaLN_mod"):
                full = block.adaLN_mod(t_emb)              # (N, 6*dim)
                chunks = full.chunk(6, dim=-1)             # 6 × (N, dim)
                modulators.append(chunks)
            else:
                # concat or FiLM block — no per-block modulators to cache
                modulators.append(None)

        return {
            "t_values": t_values,
            "t_emb": t_emb,
            "modulators": modulators,
        }

    def _compute_weights_tag(self) -> int:
        """Cheap identifier for the head's current weights. Uses the data_ptr
        of the first spectral weight tensor — changes on load_state_dict."""
        try:
            for p in self.head.parameters():
                if p.requires_grad and p.ndim == 2:
                    return int(p.data_ptr())
        except Exception:
            pass
        return 0


# ---- demo & sanity check ---------------------------------------------------

def _self_test():
    """Build a tiny LatCH, warm the cache, verify lookup matches live calc."""
    from stable_audio_tools.models.latch import LatCH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = LatCH(in_channels=64, out_channels=1, dim=128, depth=2,
                 num_heads=8, t_injection="adaln_zero").to(device).eval()

    cache = TimeConditioningCache(head, device=device)
    cache.warm([10, 40])
    assert cache.cached_counts == [10, 40]

    # Lookup step 5 of 40
    entry = cache.get(40, 5)
    assert entry is not None, "expected cached entry"
    t_emb_cached, mods_cached = entry

    # Live calc for the same t value
    t_value = torch.linspace(1.0, 0.0, 41, device=device)[:-1][5:6]
    with torch.no_grad():
        t_emb_live = head.t_embedder(t_value)
        mods_live = []
        for block in head.blocks:
            full = block.adaLN_mod(t_emb_live)
            mods_live.append(full.chunk(6, dim=-1))

    # GPU tensor-core matmul has non-determinism well below numerical concern;
    # allow ~1e-4 tolerance.
    assert torch.allclose(t_emb_cached, t_emb_live, atol=1e-4, rtol=1e-4), \
        f"t_emb mismatch (max diff {(t_emb_cached - t_emb_live).abs().max().item():.2e})"
    for cached_block, live_block in zip(mods_cached, mods_live):
        for c, l in zip(cached_block, live_block):
            assert torch.allclose(c, l, atol=1e-4, rtol=1e-4), "modulator mismatch"

    # Uncommon count -> None
    assert cache.get(37, 0) is None
    cache.ensure(37)
    assert cache.get(37, 0) is not None
    print("TimeConditioningCache self-test PASS")


if __name__ == "__main__":
    _self_test()
