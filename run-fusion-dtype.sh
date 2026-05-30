#!/bin/bash
# Validate the fp16 + bf16 hot-path predictions from §20A microbench.
#
# Profile predicted NS5 in FP32 (16 ms) -> FP16 (~4 ms), step total
# 38 ms -> ~22 ms = AdamW parity. BF16 is the safer cousin: wider range
# than FP16 but fewer mantissa bits (7 vs 10), important for NS5's
# 5x iterated matmul that compounds rounding error.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_dtype_s1"
mkdir -p "$LOGDIR"

COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 6 --num-heads 8 --batch-size 64 --epochs 20 --lr 3e-4 \
--seed 1 --save-best-only --compile --t-injection adaln_zero \
--optimizer fusion --loss smoothl1 --feature rms_energy_bass"

CELLS=(
  "fp16"
  "bf16"
)

echo "=== fp16 + bf16 hot-dtype validation (B1 = Full Fusion + SmoothL1) ==="
echo "Started: $(/usr/bin/date)"
echo ""

for d in "${CELLS[@]}"; do
  tag="dtype_${d}_s1"
  log="$LOGDIR/rms_energy_bass_${d}.log"
  echo "----- [B1, --hot-dtype $d] -----"
  echo "  start: $(/usr/bin/date +%H:%M:%S)"
  "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
    $COMMON --hot-dtype "$d" --tag "$tag" \
    > "$log" 2>&1
  rc=$?
  last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
  rate=$(/usr/bin/grep -oE '[0-9.]+it/s' "$log" | /usr/bin/tail -5 | /usr/bin/head -1)
  echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  rate=$rate  $last_metric"
done

echo ""
echo "===== DTYPE DONE ====="
echo "Finished: $(/usr/bin/date)"
echo ""
echo "Reference (from §19, --hot-dtype fp32 default):"
echo "  B1 fp32:  ~19 min/cell, val_point_mae=3.1921"
