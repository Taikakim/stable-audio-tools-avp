#!/bin/bash
# Per-component optimiser ablation: test each of the inspirations on its own.
#
# Spec inspirations:
#   Muon          spectral SGD via Newton-Schulz quintic
#   MONA          Muon + Nesterov curvature deflection
#   KL-Shampoo    two-sided Kronecker preconditioner (no spectral norm)
#   ScheduleFree+ averaging + Polyak step (no spectral path)
#   SF-NorMuon    Muon + SF averaging + per-neuron row scale + WD-on-z
#
# All cells run with SmoothL1 loss to isolate optimiser effect from loss effect.
# Existing data points from §19 (already in latch_fusion_bakeoff_s1/):
#   A1 = AdamW + SmoothL1                    val_point_mae 3.4683
#   B1 = Full Fusion + SmoothL1              val_point_mae 3.1921
#   B2 = Full Fusion + TemporalShapeLoss     val_point_mae 3.1785

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_components_s1"
mkdir -p "$LOGDIR"

COMMON="--subset-frac 0.3 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 6 --num-heads 8 --batch-size 64 --epochs 20 --lr 3e-4 \
--seed 1 --save-best-only --compile --t-injection adaln_zero \
--optimizer fusion --loss smoothl1"

# (cell-tag, components)
CELLS=(
  "Muon|ns5"
  "MONA|ns5,mona"
  "KLShampoo|shampoo"
  "ScheduleFreePlus|sf"
  "SFNorMuon|ns5,normuon,sf"
)

echo "=== Per-component ablation on rms_energy_bass ==="
echo "Started: $(/usr/bin/date)"
echo ""

for cell in "${CELLS[@]}"; do
  IFS='|' read -r cell_name comps <<< "$cell"
  tag="components_${cell_name}_s1"
  log="$LOGDIR/rms_energy_bass_${cell_name}.log"
  echo "----- [$cell_name] components=$comps -----"
  echo "  start: $(/usr/bin/date +%H:%M:%S)"
  "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
    --feature rms_energy_bass $COMMON --components "$comps" --tag "$tag" \
    > "$log" 2>&1
  rc=$?
  last_metric=$(/usr/bin/grep -E 'Epoch 20:' "$log" | /usr/bin/tail -1)
  echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  $last_metric"
done

echo ""
echo "===== COMPONENTS DONE ====="
echo "Finished: $(/usr/bin/date)"
