#!/bin/bash
# Shrink experiment: can FusionOpt let us use smaller models?
#
# Test cells (B2 = Fusion + Temporal, our winning condition):
#   B2_d256_dp6     reference (already in Phase 1)
#   B2_d128_dp6     half width  - 1/4 params, 1/8 GEMMs
#   B2_d256_dp4     shallower    - 2/3 transformer depth
#   B2_d128_dp4     both         - smallest

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_shrink_s1"
mkdir -p "$LOGDIR"

COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--batch-size 64 --epochs 20 --lr 3e-4 --num-heads 8 --seed 1 \
--save-best-only --compile --t-injection adaln_zero \
--optimizer fusion --loss temporal --mona-alpha 0.2 --lambda-deriv 1.0 --lambda-multi 0.5"

# (cell-tag, --dim, --depth)
CONFIGS=(
  "d128_dp6|128|6"
  "d256_dp4|256|4"
  "d128_dp4|128|4"
)

echo "=== FusionOpt shrink experiment (seed 1, subset 0.3, 20 ep) ==="
echo "Started: $(/usr/bin/date)"
echo "Reference: dim=256, depth=6 already in Phase 1 (B2 results)"
echo ""

for feat in rms_energy_bass spectral_flatness; do
  extra=""
  [ "$feat" = "spectral_flatness" ] && extra="--standardize"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r cell d dp <<< "$cfg"
    tag="shrink_${feat}_${cell}_s1"
    log="$LOGDIR/${feat}_${cell}.log"
    echo "----- [$feat / $cell] dim=$d depth=$dp -----"
    echo "  start: $(/usr/bin/date +%H:%M:%S)"
    "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
      --feature "$feat" $extra $COMMON --dim "$d" --depth "$dp" --tag "$tag" \
      > "$log" 2>&1
    rc=$?
    last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
    echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  $last_metric"
  done
done

echo ""
echo "===== SHRINK DONE ====="
echo "Finished: $(/usr/bin/date)"
