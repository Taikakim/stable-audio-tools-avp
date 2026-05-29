#!/bin/bash
# Production ship retrain — SF-NorMuon + bf16 + d256/dp4 on pilot heads,
# plus one Full Fusion control head (bass) for ear A/B comparison.
#
# Full data subset, 30 epochs, shared 5% holdout (so val_point_mae is
# comparable to §18, §19 numbers).
#
# Time estimate per cell with --compile + bf16 at d256/dp4:
#   ~30 sec/epoch × 30 = ~15 min/head
# 4 cells (3 SF-NorMuon + 1 Full Fusion) -> ~60 min total.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
SEED=1
LOGDIR="latch_ship_fusion_s${SEED}"
mkdir -p "$LOGDIR"

# Each cell: feature | extra-args | variant-tag | components
CELLS=(
  "rms_energy_bass||sfn|ns5,normuon,sf"
  "spectral_flatness|--standardize|sfn|ns5,normuon,sf"
  "spectral_flux||sfn|ns5,normuon,sf"
  "rms_energy_bass||fullfusion|ns5,normuon,sf,mona,shampoo"
)

echo "=== Production ship retrain (SF-NorMuon + bf16 + d256/dp4) ==="
echo "Started: $(/usr/bin/date)"
echo ""

for cell in "${CELLS[@]}"; do
  IFS='|' read -r feat extra variant comps <<< "$cell"
  tag="ship_${feat}_${variant}_s${SEED}"
  log="$LOGDIR/${feat}_${variant}.log"
  echo "----- [$feat / $variant] components=$comps -----"
  echo "  start: $(/usr/bin/date +%H:%M:%S)"
  "$PYTHON" scripts/train_latch.py --config latch_train_production.yaml \
    --feature "$feat" $extra --components "$comps" --tag "$tag" --seed "$SEED" \
    > "$log" 2>&1
  rc=$?
  last_metric=$(/usr/bin/grep -E 'Epoch 30:' "$log" | /usr/bin/tail -1)
  rate=$(/usr/bin/grep -oE '[0-9.]+it/s' "$log" | /usr/bin/tail -5 | /usr/bin/head -1)
  echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  rate=$rate"
  echo "         $last_metric"
done

echo ""
echo "===== SHIP RETRAIN DONE ====="
echo "Finished: $(/usr/bin/date)"
echo ""
echo "Reference (current ship A1, §18 raw-MAE table):"
echo "  bass     3.4683"
echo "  flatness 0.0560 (was +11.5% regression vs v2; B2 recovered it)"
echo "  flux    16.3180"
