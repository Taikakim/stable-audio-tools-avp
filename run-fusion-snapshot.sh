#!/bin/bash
# Snapshot ensemble (§22-aftermath option 5).
#
# Train the production recipe but KEEP all per-epoch checkpoints (no
# --save-best-only). The Schedule-Free averaging means each x_t at epoch N
# is a valid deployable model — so the trajectory itself produces a
# 30-snapshot ensemble at zero extra training cost.
#
# Run on 2 features. After completion, audition snapshots at ep{10,15,20,25,30}
# to see how the head's "character" drifts through training.
#
# 2 cells × ~50 min = 1.7 hours.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_snapshot_s1"
mkdir -p "$LOGDIR"

COMMON="--subset-frac 1.0 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 4 --num-heads 8 --batch-size 64 --epochs 30 --lr 3e-4 \
--seed 1 --compile --t-injection adaln_zero \
--optimizer fusion --components ns5,normuon,sf --hot-dtype bf16 --loss smoothl1"
# Note: NO --save-best-only, so per-epoch _ep<N>.pt checkpoints are kept.

CELLS=(
  "rms_energy_bass||snap_bass"
  "spectral_flatness|--standardize|snap_flatness"
)

echo "=== Snapshot ensemble (production recipe, all per-epoch ckpts kept) ==="
echo "Started: $(/usr/bin/date)"
echo ""

for cell in "${CELLS[@]}"; do
  IFS='|' read -r feat extra tag <<< "$cell"
  full_tag="snap_${tag}_s1"
  log="$LOGDIR/${tag}.log"
  echo "----- [$feat] $tag -----"
  echo "  start: $(/usr/bin/date +%H:%M:%S)"
  "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
    --feature "$feat" $extra $COMMON --tag "$full_tag" > "$log" 2>&1
  rc=$?
  last_metric=$(/usr/bin/grep -E 'Epoch 30:' "$log" | /usr/bin/tail -1)
  ckpts=$(/bin/ls latch_weights/latch_${feat}_${full_tag}_ep*.pt 2>/dev/null | /usr/bin/wc -l)
  echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  snapshots saved=$ckpts"
  echo "         $last_metric"
done

echo ""
echo "===== SNAPSHOT DONE ====="
echo "Finished: $(/usr/bin/date)"
echo ""
echo "Resulting ensemble (per head): latch_weights/latch_<feat>_snap_<feat>_s1_ep<N>.pt"
echo "Pick epochs of interest for the audition UI; recommended: {10, 15, 20, 25, 30}."
