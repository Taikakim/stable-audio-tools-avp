"""Retrofit existing latch_weights/*.pt files into the new dict format.

Old format: bare state_dict (a flat dict of param-name -> tensor).
New format: {"state_dict": ..., "feature_name": ..., "feature_stats": ..., "target_kind_default": ...}.

Usage:
    python scripts/retrofit_latch_stats.py [--dir latch_weights] [--dry-run]

Each .pt file in --dir is inspected:
  - if already new format -> skipped
  - else -> filename is parsed to recover the feature name, the LatCHDataset is
    opened for that feature, target stats are sampled, and the file is rewritten.

Requires that /run/media/kim/{Lehto,Mantu} drives are mounted (LatCHDataset needs them).
"""
import argparse
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latch_dataset import LatCHDataset
from train_latch import _sample_target_stats, _default_kind_for  # type: ignore


_FILENAME_RE = re.compile(r"^latch_(?P<feature>.+)_ep(?P<epoch>\d+)\.pt$")


def parse_feature_name(filename: str) -> str | None:
    m = _FILENAME_RE.match(filename)
    if not m:
        return None
    return m.group("feature")


def is_new_format(obj) -> bool:
    return isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict)


def retrofit_one(path: Path, latent_dir: str, info_dir: str, dry_run: bool) -> bool:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if is_new_format(raw):
        print(f"  [skip] {path.name}: already new format")
        return False

    feature = parse_feature_name(path.name)
    if feature is None:
        print(f"  [skip] {path.name}: could not parse feature from filename")
        return False

    print(f"  [retrofit] {path.name}: feature='{feature}'")
    bare = feature.removesuffix("_ts")
    dataset = LatCHDataset(latent_dir, info_dir, target_feature=bare)
    if len(dataset) == 0:
        print(f"    -> dataset is empty, cannot compute stats; skipping")
        return False

    stats = _sample_target_stats(dataset, n=200)
    if not stats:
        print(f"    -> stats sampling returned empty (drives stale?); skipping")
        return False
    kind = _default_kind_for(bare)
    print(f"    stats={stats}  kind={kind}")

    if dry_run:
        print(f"    (dry-run; not writing)")
        return False

    payload = {
        "state_dict": raw,
        "feature_name": feature,
        "feature_stats": stats,
        "target_kind_default": kind,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="latch_weights",
                    help="Directory containing latch_*.pt files")
    ap.add_argument("--latent-dir", default="/run/media/kim/Lehto/latents")
    ap.add_argument("--info-dir",   default="/run/media/kim/Mantu/ai-music/Goa_Separated_crops")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    weights_dir = Path(args.dir)
    if not weights_dir.exists():
        sys.exit(f"directory does not exist: {weights_dir}")

    files = sorted(weights_dir.glob("*.pt"))
    print(f"Scanning {weights_dir} ({len(files)} .pt files)")
    n_changed = 0
    for f in files:
        if retrofit_one(f, args.latent_dir, args.info_dir, args.dry_run):
            n_changed += 1
    print(f"Done. Files updated: {n_changed}/{len(files)}")


if __name__ == "__main__":
    main()
