#!/usr/bin/env bash
# FusionOpt + TemporalShapeLoss bake-off (Phase 1).
#
# Spec: docs/superpowers/specs/2026-05-29-fusion-optimiser-design.md §5
#
# 2x2 matrix x 3 pilot heads = 12 cells. All four cells trained at the SAME
# scale (--subset-frac 0.3 --epochs 20) so val_point_mae is comparable across
# the matrix; existing ship retrain data (subset 1.0, epochs 40) is a different
# scale and is recorded separately in LATCH_RESULTS.txt §18 as reference only.
#
# Time budget: ~8-10 min/run × 12 = ~2-2.5 hours total. All runs use the same
# 5% holdout as ship retrain (holdout-frac 0.05, holdout-seed 12345) so the
# val set itself is identical.
#
# Usage:  ./run-fusion-bakeoff.sh
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
SEED=1                       # ship-retrain best seed
LOGDIR="latch_fusion_bakeoff_s${SEED}"
mkdir -p "$LOGDIR"

# Shared knobs across all cells
COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 6 --num-heads 8 --batch-size 64 --epochs 20 --lr 3e-4 \
--seed $SEED --save-best-only --compile --t-injection adaln_zero"

# Heads: name | extra-args (e.g. --standardize for flatness)
declare -A HEAD_EXTRA=(
  ["rms_energy_bass"]=""
  ["spectral_flux"]=""
  ["spectral_flatness"]="--standardize"
)

CELLS=(
  "A1|--optimizer adamw  --loss smoothl1"
  "A2|--optimizer adamw  --loss temporal --lambda-deriv 1.0 --lambda-multi 0.5"
  "B1|--optimizer fusion --loss smoothl1 --mona-alpha 0.2"
  "B2|--optimizer fusion --loss temporal --mona-alpha 0.2 --lambda-deriv 1.0 --lambda-multi 0.5"
)

echo "=== FusionOpt bake-off (seed $SEED, subset 0.3, 20 ep) ==="
echo "Started: $(/usr/bin/date)"
echo ""

for feat in rms_energy_bass spectral_flux spectral_flatness; do
  extra="${HEAD_EXTRA[$feat]}"
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r cell_name cell_args <<< "$cell"
    tag="bakeoff_${feat}_${cell_name}_s${SEED}"
    log="$LOGDIR/${feat}_${cell_name}.log"
    echo "----- [$feat / $cell_name] -----"
    echo "  start: $(/usr/bin/date +%H:%M:%S)"
    "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
      --feature "$feat" $extra $cell_args $COMMON --tag "$tag" \
      > "$log" 2>&1
    rc=$?
    last_metric=$(/usr/bin/grep -E '(new best|Epoch 20)' "$log" | /usr/bin/tail -1)
    echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  $last_metric"
  done
done

echo ""
echo "===== BAKE-OFF DONE ====="
echo "Finished: $(/usr/bin/date)"
echo "Logs in: $LOGDIR/"
echo ""
echo "Cell summary (last best val_median + diagnostics):"
for feat in rms_energy_bass spectral_flux spectral_flatness; do
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r cell_name _cell_args <<< "$cell"
    log="$LOGDIR/${feat}_${cell_name}.log"
    best=$(/usr/bin/grep 'new best (val_median=' "$log" 2>/dev/null | /usr/bin/tail -1)
    diag=$(/usr/bin/grep 'val_point_mae=' "$log" 2>/dev/null | /usr/bin/tail -1)
    printf "  %-20s %-3s  %s | %s\n" "$feat" "$cell_name" "$best" "$diag"
  done
done
