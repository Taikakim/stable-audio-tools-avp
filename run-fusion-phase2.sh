#!/bin/bash
# FusionOpt bake-off Phase 2 — seed-variance test on spectral_flux.
#
# Spec criterion: B2 seed-std <= 0.7 * A1 seed-std (MONA curvature claim).
# We already have s=1 from Phase 1. Run s=2 and s=3 for A1 and B2 only.
# Total: 4 runs, ~50 min.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_bakeoff_s1"   # share dir with Phase 1 logs

COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 6 --num-heads 8 --batch-size 64 --epochs 20 --lr 3e-4 \
--save-best-only --compile --t-injection adaln_zero"

# (cell, args, seed)
RUNS=(
  "A1|--optimizer adamw  --loss smoothl1|2"
  "A1|--optimizer adamw  --loss smoothl1|3"
  "B2|--optimizer fusion --loss temporal --mona-alpha 0.2 --lambda-deriv 1.0 --lambda-multi 0.5|2"
  "B2|--optimizer fusion --loss temporal --mona-alpha 0.2 --lambda-deriv 1.0 --lambda-multi 0.5|3"
)

echo "=== Phase 2: spectral_flux seed-variance test ==="
echo "Started: $(/usr/bin/date)"
echo ""

for run in "${RUNS[@]}"; do
  IFS='|' read -r cell args seed <<< "$run"
  tag="bakeoff_spectral_flux_${cell}_s${seed}"
  log="$LOGDIR/spectral_flux_${cell}_s${seed}.log"
  echo "----- spectral_flux / $cell / seed $seed -----"
  echo "  start: $(/usr/bin/date +%H:%M:%S)"
  "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
    --feature spectral_flux $args $COMMON --seed "$seed" --tag "$tag" \
    > "$log" 2>&1
  rc=$?
  last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
  echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  $last_metric"
done

echo ""
echo "===== PHASE 2 DONE ====="
echo "Finished: $(/usr/bin/date)"
