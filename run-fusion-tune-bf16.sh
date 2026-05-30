#!/bin/bash
# Re-tune bf16 kernels in TunableOp (force longer autotuner search), then
# measure the resulting speedup on a 20-epoch bf16 cell.
#
# Cleared the 3 "Default" bf16 entries from the cache before running this,
# so TunableOp re-attempts those shapes with the extended tuning budget.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_bf16_tuned_s1"
mkdir -p "$LOGDIR"

# Aggressive autotuner: longer per-kernel search budget.
# Defaults: max_tuning_duration=30ms, max_tuning_iterations=30
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=500
export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=500

COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 6 --num-heads 8 --batch-size 64 --lr 3e-4 \
--seed 1 --save-best-only --compile --t-injection adaln_zero \
--optimizer fusion --loss smoothl1 --feature rms_energy_bass \
--hot-dtype bf16"

# Phase 1: short tuning pass (5 epochs) to populate the cache
echo "=== Phase 1: 5-epoch tuning pass (populate bf16 kernels) ==="
echo "Started: $(/usr/bin/date)"
tag="tune_bf16_phase1_s1"
log="$LOGDIR/phase1_tune.log"
"$PYTHON" scripts/train_latch.py --config latch_train.yaml $COMMON \
  --epochs 5 --tag "$tag" > "$log" 2>&1
echo "  end: $(/usr/bin/date)  rc=$?"
echo ""

# Show cache state after tuning
echo "BF16 cache entries:        $(/usr/bin/grep -c BFloat16 /home/kim/pytorch-tunings-7.2.3/tunableop_results0.csv)"
echo "BF16 Default entries left: $(/usr/bin/grep -c 'BFloat16.*Default' /home/kim/pytorch-tunings-7.2.3/tunableop_results0.csv)"
echo ""

# Phase 2: full 20-epoch run with populated cache to measure speedup
echo "=== Phase 2: 20-epoch run (measure speed with populated bf16 cache) ==="
tag="tuned_bf16_s1"
log="$LOGDIR/phase2_run.log"
"$PYTHON" scripts/train_latch.py --config latch_train.yaml $COMMON \
  --epochs 20 --tag "$tag" > "$log" 2>&1
echo "  end: $(/usr/bin/date)  rc=$?"

last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
rate=$(/usr/bin/grep -oE '[0-9.]+it/s' "$log" | /usr/bin/tail -5 | /usr/bin/head -1)
echo "Result: rate=$rate"
echo "        $last_metric"
echo ""
echo "===== TUNE-BF16 DONE ====="
echo "Finished: $(/usr/bin/date)"
echo ""
echo "Reference: bf16 without tuned cache = 26.4 it/s, val_point_mae=3.2238 (§20D)"
echo "           fp32 baseline           = ~16 it/s, val_point_mae=3.1921 (§19 B1)"
