#!/bin/bash
# Benchmark fp16_safe NS5 vs bf16 reference.
#
# fp16_safe = fp16 matmuls (tensor cores) + fp32 polynomial accumulation
# with per-tensor rescale-and-restore around each matmul. CPU sanity
# matched fp32 NS5 to 8.5e-4 (well below numerical noise). Predicted
# 1.3-1.5x speedup over bf16 if the fp16 tensor-core throughput dominates
# the rescale overhead.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_fp16safe_s1"
mkdir -p "$LOGDIR"

# Match the §20D + §21 cell shape: B1 (Full Fusion + SmoothL1) at d256/dp6
# so the timing is directly comparable to the bf16 reference cell.
COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 6 --num-heads 8 --batch-size 64 --epochs 20 --lr 3e-4 \
--seed 1 --save-best-only --compile --t-injection adaln_zero \
--optimizer fusion --loss smoothl1 --feature rms_energy_bass"

tag="fp16safe_s1"
log="$LOGDIR/rms_energy_bass_fp16safe.log"
echo "=== fp16_safe hot-dtype benchmark ==="
echo "Started: $(/usr/bin/date)"
"$PYTHON" scripts/train_latch.py --config latch_train.yaml $COMMON \
  --hot-dtype fp16_safe --tag "$tag" > "$log" 2>&1
rc=$?
last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
rate=$(/usr/bin/grep -oE '[0-9.]+it/s' "$log" | /usr/bin/tail -5 | /usr/bin/head -1)
echo "End: $(/usr/bin/date) rc=$rc rate=$rate"
echo "Result: $last_metric"
echo ""
echo "Reference numbers (rms_energy_bass, d256/dp6, --compile, 20ep, subset 0.3):"
echo "  fp32 (§19 B1):     ~16 it/s     val_point_mae=3.1921"
echo "  bf16 (§20D):       26.4 it/s    val_point_mae=3.2238"
echo "  fp16 unsafe (§20D): 66.3 it/s    val_point_mae=NaN (diverged)"
echo "  fp16_safe (now):    ${rate}"
