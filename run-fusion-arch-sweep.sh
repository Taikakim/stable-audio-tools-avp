#!/bin/bash
# Architecture sweep — extend §20B shrink with wider/deeper variants user asked.
# Recipe = SF-NorMuon + bf16 + smoothl1 (matches §21 production target).
# Two heads (bass + flatness) × 4 configs = 8 cells.
#
# Predicted timings @ d256/dp4 was ~17 min/cell. Other shapes:
#   d512/dp4  ~25 min (4x bigger spectral mats)
#   d256/dp8  ~22 min (more blocks)
#   d128/dp8  ~12 min
#   d128/dp16 ~17 min
# Total ~150 min.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_arch_s1"
mkdir -p "$LOGDIR"

COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--num-heads 8 --batch-size 64 --epochs 20 --lr 3e-4 \
--seed 1 --save-best-only --compile --t-injection adaln_zero \
--optimizer fusion --components ns5,normuon,sf --loss smoothl1 --hot-dtype bf16"

CONFIGS=(
  "d512_dp4|512|4"
  "d256_dp8|256|8"
  "d128_dp8|128|8"
  "d128_dp16|128|16"
)

echo "=== FusionOpt architecture sweep (SFN + bf16) ==="
echo "Started: $(/usr/bin/date)"
echo ""

for feat in rms_energy_bass spectral_flatness; do
  extra=""
  [ "$feat" = "spectral_flatness" ] && extra="--standardize"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag dim depth <<< "$cfg"
    full_tag="arch_${feat}_${tag}_s1"
    log="$LOGDIR/${feat}_${tag}.log"
    echo "----- [$feat / $tag] dim=$dim depth=$depth -----"
    echo "  start: $(/usr/bin/date +%H:%M:%S)"
    "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
      --feature "$feat" $extra $COMMON --dim "$dim" --depth "$depth" \
      --tag "$full_tag" > "$log" 2>&1
    rc=$?
    last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
    rate=$(/usr/bin/grep -oE '[0-9.]+it/s' "$log" | /usr/bin/tail -5 | /usr/bin/head -1)
    echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  rate=$rate"
    echo "         $last_metric"
  done
done

echo ""
echo "===== ARCH SWEEP DONE ====="
echo "Finished: $(/usr/bin/date)"
echo ""
echo "Reference (§20B + §21 at bf16):"
echo "  bass d256/dp6 (§19 B1 fp32)    val_point_mae 3.1921"
echo "  bass d256/dp4 (§20B shrink)    val_point_mae 3.2294 (+1.6%)"
echo "  bass d256/dp4 (§21 ship 30ep)  val_point_mae 3.0339 (-12.5% vs A1)"
echo "  flatness d256/dp4 (§21 ship)   val_point_mae 0.0450 (-19.6% vs A1)"
