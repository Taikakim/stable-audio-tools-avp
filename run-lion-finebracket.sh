#!/usr/bin/env bash
# Fine Lion LR bracket around the 1e-5 sweet spot. Single SEEDED runs (reliable as long
# as the seed-variance check shows low cross-seed spread). 250W, dim256, 30ep.
# Wider context points (3e-5, 5e-6) included to confirm the U-shape minimum.
# Usage:  SEED=0 PYTHON=sat-venv/bin/python ./run-lion-finebracket.sh
set -u
PYTHON="${PYTHON:-python}"; SEED="${SEED:-0}"
LOGDIR="latch_lion_finebracket"; mkdir -p "$LOGDIR"
COMMON="--config latch_train_density.yaml --feature onsets_activations --smooth-kind gaussian \
--subset-frac 0.3 --holdout-frac 0.05 --preload gpu --dim 256 --num-heads 8 --batch-size 64 \
--save-best-only --epochs 30 --optimizer lion --seed $SEED"

for lr in 3e-5 2e-5 1.5e-5 1.2e-5 1e-5 9e-6 7e-6 5e-6; do
  tag="lion_${lr/./}_s${SEED}"
  "$PYTHON" scripts/train_latch.py $COMMON --lr "$lr" --tag "optm_$tag" > "$LOGDIR/$tag.log" 2>&1
  echo "  lr=$lr : $(grep 'new best (val_median=' "$LOGDIR/$tag.log" | tail -1)"
done
echo "LION FINE BRACKET DONE -> $LOGDIR/"
