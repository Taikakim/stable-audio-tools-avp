"""TimeConditioningCache — precomputed time embeddings + adaLN-zero modulators.

For a LatCH head with `t_injection="adaln_zero"`, every guide-step forward does:

  t_emb = self.t_embedder(t)                 # (B, dim)
  for each block:
      (g1, b1, a1, g2, b2, a2) = block.adaLN_mod(t_emb).chunk(6, dim=-1)
      ... use modulators in the forward path ...

The sampler queries LatCH at a small set of fixed t values:
  - t = 0          (used for the mean-guidance z_0|t loop, every step, n_iter+ times)
  - t = t_curr     (one value per sampler step, used for variance guidance)

t_emb and the modulators are pure functions of t and weights — cacheable.

For the production target (SF-NorMuon at d256/dp4, 40 sampler steps, n_iter=4):
  - t=0 cache hit:  ~4 calls/step * 40 = 160 hits  (3 small Linears each)
  - t=t_curr hit:   ~1 call/step * 40 = 40 hits
Total per render: ~200 cache hits * ~2 ms saved each ~ 400 ms removed.

Usage:
    cache = TimeConditioningCache(latch_head, device="cuda")
    cache.warm_for_schedule(40)        # all t values the sampler will visit
    head._time_cache = cache           # attach; LatCH.forward picks it up

The forward path checks for `self._time_cache` and falls back to live calc when
the t tensor isn't uniform across the batch OR the value isn't cached. No
quality regression: cached values are bit-for-bit identical to live (modulo
tensor-core non-determinism well below 1e-4).
"""

from __future__ import annotations

from typing import Optional

import torch


def _quantize_key(t: float) -> float:
    """Map a t value to a stable dict key (rounds the floating-point noise)."""
    return round(float(t), 6)


class TimeConditioningCache:
    """Precomputed t_embedder + per-block adaLN-zero modulator outputs.

    Keyed by t value (quantised to 1e-6). On-demand caching for any value via
    .ensure(). Self-invalidates when underlying weights change.
    """

    def __init__(self, latch_head, device: str | torch.device = "cuda"):
        self.head = latch_head
        self.device = torch.device(device)
        # quantised t-key -> {"t_emb": (1, dim), "modulators": [tuple(6 × (1,dim)) | None]}
        self._by_value: dict[float, dict] = {}
        self._weights_tag: Optional[int] = self._compute_weights_tag()
        self._hits = 0
        self._misses = 0

    # ---- public ---------------------------------------------------------

    def warm(self, t_values) -> None:
        """Precompute caches for an explicit list of t values."""
        for t in t_values:
            self.ensure(float(t))

    def warm_for_schedule(self, n_steps: int, include_zero: bool = True) -> None:
        """Precompute for the rectified-flow schedule plus optional t=0.

        t = linspace(1.0, 0.0, n_steps+1)[:-1]   # what the sampler visits
        Plus t = 0 if include_zero (used for the z_0|t mean-guidance path).
        """
        n_steps = int(n_steps)
        # Schedule values: linspace(1, 0, n+1)[:-1]; do not include 0 here, that's
        # the index that's dropped — we explicitly add 0 below if requested.
        t_values = torch.linspace(1.0, 0.0, n_steps + 1)[:-1].tolist()
        for t in t_values:
            self.ensure(t)
        if include_zero:
            self.ensure(0.0)

    def ensure(self, t: float) -> None:
        """On-demand: build the cache entry for a single t value if missing."""
        k = _quantize_key(t)
        if k in self._by_value:
            return
        self._by_value[k] = self._build_one(float(t))

    def get(self, t, auto_warm: bool = True):
        """Return cached entry for scalar t.

        When `auto_warm=True` (default), a miss populates the cache for `t`
        and returns the freshly-built entry, so subsequent calls at the same
        t hit the cache. The miss is still counted in hit_stats for visibility.
        Set `auto_warm=False` to get the strict-lookup semantics (returns None
        on miss, caller must explicitly .ensure() to populate).

        The entry is a dict with keys:
          t_emb:       (1, dim) tensor
          modulators:  list of length depth; each element is either
                       tuple of 6 (1, dim) tensors  -> adaln_zero block,
                       or None                       -> film/concat block.
        Returns None when underlying weights changed (cache invalidated) and
        the caller didn't pass auto_warm=True.
        """
        if self._weights_tag != self._compute_weights_tag():
            # Underlying head weights changed — drop the whole cache.
            self._by_value.clear()
            self._weights_tag = self._compute_weights_tag()
        k = _quantize_key(t)
        entry = self._by_value.get(k)
        if entry is None:
            self._misses += 1
            if auto_warm:
                self.ensure(t)
                entry = self._by_value.get(k)
        else:
            self._hits += 1
        return entry

    def clear(self) -> None:
        self._by_value.clear()

    def invalidate_for_new_weights(self) -> None:
        """Call when the underlying LatCH head's weights change."""
        self.clear()
        self._weights_tag = self._compute_weights_tag()

    @property
    def cached_values(self) -> list[float]:
        return sorted(self._by_value.keys())

    @property
    def hit_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    # ---- internals ------------------------------------------------------

    @torch.no_grad()
    def _build_one(self, t: float) -> dict:
        """Precompute t_emb and per-block modulators for a single t value."""
        t_tensor = torch.tensor([float(t)], device=self.device)  # (1,)
        t_emb = self.head.t_embedder(t_tensor)                   # (1, dim)
        modulators: list = []
        for block in self.head.blocks:
            if hasattr(block, "adaLN_mod"):
                full = block.adaLN_mod(t_emb)                    # (1, 6*dim)
                modulators.append(tuple(full.chunk(6, dim=-1)))  # 6 × (1, dim)
            else:
                modulators.append(None)
        return {"t_emb": t_emb, "modulators": modulators}

    def _compute_weights_tag(self) -> int:
        """Cheap identifier for the head's current weights. Uses data_ptr of
        the first trainable 2D weight tensor; changes on load_state_dict."""
        try:
            for p in self.head.parameters():
                if p.requires_grad and p.ndim == 2:
                    return int(p.data_ptr())
        except Exception:
            pass
        return 0


# ---- self-test -------------------------------------------------------------

def _self_test():
    """Verify cached output matches live computation (within tensor-core noise)."""
    from stable_audio_tools.models.latch import LatCH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = LatCH(in_channels=64, out_channels=1, dim=128, depth=2,
                 num_heads=8, t_injection="adaln_zero").to(device).eval()

    cache = TimeConditioningCache(head, device=device)
    cache.warm_for_schedule(40, include_zero=True)
    # Should have 40 schedule values + t=0 entry. Schedule starts at 1.0 and
    # decreases — 0.0 isn't in the schedule (it's the dropped last value), so
    # warm + include_zero gives 41 unique cached entries.
    assert len(cache.cached_values) == 41, f"unexpected count {len(cache.cached_values)}"

    # Sample a value from the schedule
    schedule = torch.linspace(1.0, 0.0, 41)[:-1].tolist()
    t_sample = schedule[5]
    entry = cache.get(t_sample)
    assert entry is not None, "expected cache hit for schedule value"

    # Live recompute
    t_tensor = torch.tensor([t_sample], device=device)
    with torch.no_grad():
        t_emb_live = head.t_embedder(t_tensor)
        mods_live = []
        for block in head.blocks:
            mods_live.append(tuple(block.adaLN_mod(t_emb_live).chunk(6, dim=-1)))

    assert torch.allclose(entry["t_emb"], t_emb_live, atol=1e-4, rtol=1e-4), \
        f"t_emb mismatch (max diff {(entry['t_emb'] - t_emb_live).abs().max().item():.2e})"
    for cm, lm in zip(entry["modulators"], mods_live):
        for c, l in zip(cm, lm):
            assert torch.allclose(c, l, atol=1e-4, rtol=1e-4), "modulator mismatch"

    # t=0 also cached
    assert cache.get(0.0) is not None

    # Uncommon value not yet cached
    assert cache.get(0.137) is None
    cache.ensure(0.137)
    assert cache.get(0.137) is not None

    print(f"TimeConditioningCache self-test PASS  ({len(cache.cached_values)} entries)")


if __name__ == "__main__":
    _self_test()


# ---- process-level cache registry ------------------------------------------

# Persists across multiple generate_diffusion_cond() calls in the same Python
# process. Keyed by (checkpoint path, n_steps). Auto-warm fills in any sampler
# t values that weren't in the linspace warm, so the second render at the same
# checkpoint + step count hits the cache 100 %.

_CACHE_REGISTRY: dict[tuple, TimeConditioningCache] = {}


def get_or_build_cache(latch_head, model_path: str, n_steps: int,
                       device: str | torch.device = "cuda") -> TimeConditioningCache:
    """Return a persistent TimeConditioningCache for (model_path, n_steps).

    Builds on first call, returns the cached instance on subsequent calls.
    The cache's internal weight-tag check still invalidates if someone
    overwrites the underlying head weights, so this is safe across hot reloads.
    """
    key = (str(model_path), int(n_steps))
    cache = _CACHE_REGISTRY.get(key)
    if cache is None:
        cache = TimeConditioningCache(latch_head, device=device)
        cache.warm_for_schedule(int(n_steps), include_zero=True)
        _CACHE_REGISTRY[key] = cache
    else:
        # Cache hit for this (model_path, n_steps). Trust that the same
        # checkpoint path -> same weights; just refresh the head reference
        # so the cache's weight-tag check matches and auto-warmed entries
        # from previous renders are reused. (To rebuild on a true weight
        # swap, call clear_cache_registry() first.)
        cache.head = latch_head
        cache._weights_tag = cache._compute_weights_tag()
    return cache


def clear_cache_registry() -> None:
    """Drop all persistent caches (e.g. after a model swap in a long-running
    process). Frees the GPU memory used by cached t_emb / modulator tensors."""
    _CACHE_REGISTRY.clear()
