#!/bin/bash
# Diversity training experiment — train heads that DIVERGE from our best ship head.
#
# Reference (frozen): ship_sfn_bass (latch_rms_energy_bass_ship_rms_energy_bass_sfn_s1_best.pt)
# val_point_mae of ref = 3.0339 (§21)
#
# Hypothesis: even when val_point_mae is similar across optimizer choices, the
# resulting heads describe the feature DIFFERENTLY (per the soup experiment §20F).
# An explicit diversity penalty during training might find new aesthetic modes —
# heads that nudge the base generator in qualitatively distinct ways for the same
# target feature.
#
# 4 cells × ~45 min each = ~3 hours.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
set -u
PYTHON="${PYTHON:-sat-venv/bin/python}"
LOGDIR="latch_fusion_diversity_s1"
mkdir -p "$LOGDIR"

REF="latch_weights/latch_rms_energy_bass_ship_rms_energy_bass_sfn_s1_best.pt"

COMMON="--subset-frac 1.0 --holdout-frac 0.05 --holdout-seed 12345 --preload ram \
--dim 256 --depth 4 --num-heads 8 --batch-size 64 --epochs 30 --lr 3e-4 \
--seed 1 --save-best-only --compile --t-injection adaln_zero \
--loss smoothl1 --feature rms_energy_bass \
--diversity-ref $REF --lambda-div 0.3 --div-mse-bound 100.0"

# (cell-tag, init-mode, optimizer flags, warmup steps)
# div_warmup ≈ 5 epochs × 2999 batches = 15000 steps for fresh init
# warm-start needs 0 (initial MSE is ~0 already)
CELLS=(
  "A_freshSFN|fresh|--optimizer fusion --components ns5,normuon,sf --hot-dtype bf16|15000"
  "B_freshAdamW|fresh|--optimizer adamw|15000"
  "C_warmAdamW|warm|--optimizer adamw|0"
  "D_warmFusion|warm|--optimizer fusion --components ns5,normuon,sf,mona,shampoo --hot-dtype bf16|0"
)

echo "=== Diversity training experiment (reference: ship_sfn_bass) ==="
echo "Started: $(/usr/bin/date)"
echo ""

for cell in "${CELLS[@]}"; do
  IFS='|' read -r tag init opt_flags warmup <<< "$cell"
  warm_flag=""
  [ "$init" = "warm" ] && warm_flag="--warm-start"
  full_tag="div_${tag}_s1"
  log="$LOGDIR/rms_energy_bass_${tag}.log"
  echo "----- [$tag] init=$init  opt=$opt_flags  warmup_steps=$warmup -----"
  echo "  start: $(/usr/bin/date +%H:%M:%S)"
  "$PYTHON" scripts/train_latch.py --config latch_train.yaml \
    $COMMON $opt_flags --div-warmup-steps "$warmup" $warm_flag \
    --tag "$full_tag" > "$log" 2>&1
  rc=$?
  last_metric=$(/usr/bin/grep -E 'Epoch 30:' "$log" | /usr/bin/tail -1)
  rate=$(/usr/bin/grep -oE '[0-9.]+it/s' "$log" | /usr/bin/tail -5 | /usr/bin/head -1)
  echo "  end:   $(/usr/bin/date +%H:%M:%S) rc=$rc  rate=$rate"
  echo "         $last_metric"
done

echo ""
echo "===== DIVERSITY DONE ====="
echo "Finished: $(/usr/bin/date)"
echo ""
echo "Reference val_point_mae = 3.0339 (§21 ship_sfn_bass)"
echo "Expected: divergence-trained heads have val_point_mae ~3.1-3.5 (worse on the"
echo "metric) but might steer the base generator in qualitatively different ways."
echo "Audition with renders to evaluate the aesthetic merit."
