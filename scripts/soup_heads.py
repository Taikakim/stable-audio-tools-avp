"""Model soup — average the weights of multiple LatCH heads trained on the same feature.

Pioneered in Wortsman et al. 2022 (arXiv:2203.05482): when you have N independently-
trained models for the same task, a uniform average of their weights (or a greedy
"selected soup") often outperforms any individual model AND any logit ensemble.
Cost: zero extra inference compute, single state dict to ship.

Usage:
  sat-venv/bin/python scripts/soup_heads.py \\
      --feature rms_energy_bass \\
      --out latch_weights/latch_rms_energy_bass_soup_best.pt \\
      --inputs \\
        latch_weights/latch_rms_energy_bass_ship_rms_energy_bass_sfn_s1_best.pt \\
        latch_weights/latch_rms_energy_bass_ship_rms_energy_bass_fullfusion_s1_best.pt

The output checkpoint shares the metadata of the first input (feature_stats,
target_kind_default, std_mean/std_std, t_injection, etc) — that's the same
deployment contract as a normal LatCH checkpoint.

Greedy variant: scripts/soup_heads.py --greedy ... selects a subset that
maximises val performance (requires a validation function; not implemented
in this minimal version).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Paths to two or more LatCH checkpoints to soup-average.")
    p.add_argument("--out", required=True, help="Output checkpoint path.")
    p.add_argument("--feature", default=None,
                   help="Optional sanity check: bail if any input has a different feature_name.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-input weights for a weighted average (defaults to uniform).")
    p.add_argument("--prefer-averaged", action="store_true", default=True,
                   help="When an input has averaged_state_dict (FusionOpt SF average), "
                        "use it; otherwise fall back to state_dict. Default: True.")
    return p.parse_args()


def load_checkpoint(path: str, prefer_averaged: bool = True) -> tuple[dict, dict]:
    """Return (state_dict, metadata). For FusionOpt heads, the averaged_state_dict
    only contains the SF-averaged PARAMETERS, not buffers like rotary_emb.inv_freq,
    so we merge averaged params over the regular state_dict to keep buffers intact.
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: bare state_dict, missing metadata")
    state = dict(raw["state_dict"])
    if prefer_averaged and "averaged_state_dict" in raw and isinstance(raw["averaged_state_dict"], dict):
        state.update(raw["averaged_state_dict"])
    metadata = {k: v for k, v in raw.items() if k not in ("state_dict", "averaged_state_dict")}
    return state, metadata


def soup(states: list[dict], weights: list[float] | None = None) -> dict:
    """Uniform (or weighted) average of N state dicts. All inputs must share keys + shapes."""
    if not states:
        raise ValueError("no states to soup")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states):
        raise ValueError(f"len(weights) {len(weights)} != len(states) {len(states)}")
    norm = sum(weights)
    weights = [w / norm for w in weights]

    # Use the first state's keys + shapes as the contract; all others must match.
    keys = list(states[0].keys())
    out = {}
    for k in keys:
        ref = states[0][k]
        if not torch.is_tensor(ref):
            out[k] = ref
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for w, s in zip(weights, states):
            if k not in s:
                raise KeyError(f"state[{k}] missing in one of the inputs")
            t = s[k]
            if t.shape != ref.shape:
                raise ValueError(f"shape mismatch on {k}: {t.shape} vs {ref.shape}")
            # Cast to fp32 for averaging to avoid lossy fp16/bf16 sums, restore dtype at end
            acc += float(w) * t.to(torch.float32)
        out[k] = acc.to(ref.dtype)
    return out


def main():
    args = parse_args()

    if len(args.inputs) < 2:
        print("Need at least 2 inputs to soup; one is just a copy.", file=sys.stderr)
        sys.exit(1)

    states = []
    metas = []
    for path in args.inputs:
        if not Path(path).exists():
            print(f"missing: {path}", file=sys.stderr)
            sys.exit(1)
        st, md = load_checkpoint(path, prefer_averaged=args.prefer_averaged)
        states.append(st)
        metas.append(md)
        feat = md.get("feature_name", "?")
        n_params = sum(t.numel() for t in st.values() if torch.is_tensor(t))
        print(f"  loaded {path}")
        print(f"    feature={feat}  params={n_params/1e6:.2f}M  metadata-tag={md.get('tag', '-')}")

    # Sanity: same feature across all inputs
    first_feat = metas[0].get("feature_name")
    if args.feature is not None:
        first_feat = args.feature
    for m, path in zip(metas, args.inputs):
        feat = m.get("feature_name")
        if first_feat is not None and feat is not None and feat != first_feat:
            print(f"FATAL: {path} has feature={feat!r} but expected {first_feat!r}", file=sys.stderr)
            sys.exit(1)

    print(f"\nAveraging {len(states)} state dicts (uniform weights={args.weights or 'uniform'})...")
    averaged = soup(states, args.weights)

    # Pack into a LatCH-compatible checkpoint: metadata from the first input,
    # plus a soup_ingredients field for traceability.
    out_pkg = dict(metas[0])
    out_pkg["state_dict"] = averaged
    out_pkg["soup_ingredients"] = [
        {"path": str(Path(p).name), "weight": float(w) if args.weights else None}
        for p, w in zip(args.inputs, (args.weights or [None] * len(args.inputs)))
    ]
    out_pkg["tag"] = (out_pkg.get("tag") or "") + "_soup"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_pkg, str(out_path))
    n_params = sum(t.numel() for t in averaged.values() if torch.is_tensor(t))
    print(f"\nWrote {out_path}  ({n_params/1e6:.2f}M params)")
    print(f"Soup ingredients: {[str(Path(p).name) for p in args.inputs]}")


if __name__ == "__main__":
    main()
