"""Parse per-cell logs from latch_fusion_bakeoff_s*/ and emit a summary table.

Usage:  sat-venv/bin/python scripts/summarise_fusion_bakeoff.py latch_fusion_bakeoff_s1/

Reads each <feature>_<cell>.log and pulls:
  - best val_median + the epoch it was set on
  - per-epoch val_point_mae / val_deriv_corr / val_multiscale_mae
  - last-epoch values for all metrics
  - any errors / crashes

Prints a markdown-style table suitable for pasting into LATCH_RESULTS.txt §19.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict


PATTERNS = {
    "epoch": re.compile(
        r"Epoch (\d+): train_avg=([0-9.eE+-]+)\s+val_mean=([0-9.eE+-]+)\s+val_median=([0-9.eE+-]+)"
        r"(?:\s+val_point_mae=([0-9.eE+-]+))?(?:\s+val_deriv_corr=([0-9.eE+-]+))?"
    ),
    "best": re.compile(r"new best \(val_median=([0-9.eE+-]+)\)"),
    "train_items": re.compile(r"\((\d+) train items\)"),
    "loss_type": re.compile(r"Robust loss: (\w+)|Temporal-shape loss:|Optimizer: (\w+)"),
}


def parse_log(path: Path) -> dict:
    """Pull metrics from a single cell log."""
    text = path.read_text(errors="replace")
    out: dict = {
        "path": str(path),
        "epochs_seen": 0,
        "best_val_median": None,
        "best_epoch": None,
        "final": {},
        "epochs": [],
        "error": None,
    }
    if "Traceback" in text:
        # Capture the last traceback's first line for the report
        tb_start = text.rfind("Traceback")
        out["error"] = text[tb_start:tb_start + 300].splitlines()[0]
    for m in PATTERNS["epoch"].finditer(text):
        ep, train_avg, val_mean, val_median = m.group(1), m.group(2), m.group(3), m.group(4)
        val_point_mae = m.group(5)
        val_deriv_corr = m.group(6)
        row = {
            "epoch": int(ep),
            "train_avg": float(train_avg),
            "val_mean": float(val_mean),
            "val_median": float(val_median),
            "val_point_mae": float(val_point_mae) if val_point_mae else None,
            "val_deriv_corr": float(val_deriv_corr) if val_deriv_corr else None,
        }
        out["epochs"].append(row)
    if out["epochs"]:
        out["epochs_seen"] = out["epochs"][-1]["epoch"]
        out["final"] = out["epochs"][-1]
        best = min(out["epochs"], key=lambda r: r["val_median"])
        out["best_val_median"] = best["val_median"]
        out["best_epoch"] = best["epoch"]
        # Also pull the val_point_mae of the BEST-by-val_median epoch
        out["best_val_point_mae"] = best.get("val_point_mae")
        out["best_val_deriv_corr"] = best.get("val_deriv_corr")
    return out


def main(root: Path):
    by_feat: dict[str, dict[str, dict]] = defaultdict(dict)
    for log in sorted(root.glob("*_*.log")):
        name = log.stem
        # name like "rms_energy_bass_A1" — split on the last _ which is the cell tag
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("A1", "A2", "B1", "B2"):
            continue
        feat, cell = parts
        by_feat[feat][cell] = parse_log(log)

    print("# FusionOpt + TemporalShapeLoss bake-off — Phase 1 results")
    print()
    print("Source dir:", root)
    print()

    # Headline table: best val_point_mae per cell, per head
    print("## val_point_mae at best epoch (raw feature units; lower = better)")
    print()
    print("| Head                | A1 (AdamW+L1) | A2 (AdamW+T) | B1 (Fusion+L1) | B2 (Fusion+T) |")
    print("|---------------------|---------------|--------------|----------------|---------------|")
    for feat in sorted(by_feat):
        row = [f"| {feat:<19}"]
        for cell in ("A1", "A2", "B1", "B2"):
            c = by_feat[feat].get(cell)
            if c is None:
                row.append("|     missing   ")
            elif c.get("error"):
                row.append("|     ERROR     ")
            elif c.get("best_val_point_mae") is not None:
                row.append(f"| {c['best_val_point_mae']:>11.4f}   ")
            else:
                row.append("|       —       ")
        row.append("|")
        print("".join(row))
    print()

    # val_deriv_corr table (temporal awareness)
    print("## val_deriv_corr at best epoch (higher = better; direction match)")
    print()
    print("| Head                | A1 | A2 | B1 | B2 |")
    print("|---------------------|----|----|----|----|")
    for feat in sorted(by_feat):
        row = [f"| {feat:<19}"]
        for cell in ("A1", "A2", "B1", "B2"):
            c = by_feat[feat].get(cell)
            v = c.get("best_val_deriv_corr") if c else None
            row.append(f"| {v:>6.4f}" if v is not None else "|     —")
        row.append("|")
        print("".join(row))
    print()

    # val_median table (legacy comparability)
    print("## val_median at best epoch (loss-function units; not cross-loss comparable)")
    print()
    print("| Head                | A1 | A2 | B1 | B2 |")
    print("|---------------------|----|----|----|----|")
    for feat in sorted(by_feat):
        row = [f"| {feat:<19}"]
        for cell in ("A1", "A2", "B1", "B2"):
            c = by_feat[feat].get(cell)
            v = c.get("best_val_median") if c else None
            row.append(f"| {v:>7.4f}" if v is not None else "|       —")
        row.append("|")
        print("".join(row))
    print()

    # Pass criteria check
    print("## Pass criteria check (per spec §5)")
    print()
    quality_pass = True
    temporal_pass = True
    geometry_pass = 0
    for feat in sorted(by_feat):
        cells = by_feat[feat]
        a1 = cells.get("A1", {})
        a2 = cells.get("A2", {})
        b2 = cells.get("B2", {})
        # 1. Quality: B2.val_point_mae <= A1.val_point_mae
        b2_mae = b2.get("best_val_point_mae")
        a1_mae = a1.get("best_val_point_mae")
        if b2_mae is not None and a1_mae is not None:
            if b2_mae > a1_mae:
                quality_pass = False
                print(f"  [FAIL] quality on {feat}: B2 ({b2_mae:.4f}) > A1 ({a1_mae:.4f})")
            else:
                print(f"  [PASS] quality on {feat}: B2 ({b2_mae:.4f}) <= A1 ({a1_mae:.4f})")
        # 2. Temporal awareness on bass + flux only
        if feat in ("rms_energy_bass", "spectral_flux"):
            b2_dc = b2.get("best_val_deriv_corr")
            a1_dc = a1.get("best_val_deriv_corr")
            if b2_dc is not None and a1_dc is not None:
                if b2_dc - a1_dc < 0.05:
                    temporal_pass = False
                    print(f"  [FAIL] temporal on {feat}: B2 ({b2_dc:.4f}) - A1 ({a1_dc:.4f}) "
                          f"= {b2_dc - a1_dc:+.4f} < 0.05")
                else:
                    print(f"  [PASS] temporal on {feat}: Δ = {b2_dc - a1_dc:+.4f} >= 0.05")
        # 3. Geometry contribution: B2 <= A2 on val_point_mae
        a2_mae = a2.get("best_val_point_mae")
        if b2_mae is not None and a2_mae is not None and b2_mae <= a2_mae:
            geometry_pass += 1

    geometry_overall = geometry_pass >= 2
    print()
    print(f"  Quality (criterion 1):           {'PASS' if quality_pass else 'FAIL'}")
    print(f"  Temporal awareness (crit. 2):    {'PASS' if temporal_pass else 'FAIL'}")
    print(f"  Geometry contribution (crit. 3): {'PASS' if geometry_overall else 'FAIL'} "
          f"({geometry_pass}/3 heads)")
    print()
    overall = quality_pass and temporal_pass and geometry_overall
    print(f"  ===> OVERALL: {'BAKE-OFF PASS' if overall else 'BAKE-OFF MIXED/FAIL'}")
    print()

    # Errors
    errs = [(f, c, by_feat[f][c]) for f in by_feat for c in by_feat[f] if by_feat[f][c].get("error")]
    if errs:
        print("## Errors detected")
        for f, c, info in errs:
            print(f"  {f}/{c}: {info['error']}")
        print()


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "latch_fusion_bakeoff_s1")
    main(root)
