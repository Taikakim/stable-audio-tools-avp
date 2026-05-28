#!/usr/bin/env bash
# Ship-quality retrain of ALL trained heads at FULL data, 40 epochs, AdamW 3e-4,
# with the best seed found by the seed shootout. Saves _best.pt (early-stop)
# AND _last.pt (final epoch) per head — no per-epoch clutter.
#
# Architecture: torch.compile + adaln_zero t-injection + FA-priority SDPA.
# Sanity verified on onsets_activations 2026-05-28 (see LATCH_RESULTS.txt §17):
#   eager+concat:   36.5 it/s, val_med 0.1700, 428 items/Wh
#   compile+film:   57.5 it/s, val_med 0.1726, ~1050 items/Wh
#   compile+adaln:  55.7 it/s, val_med 0.1682 ← chosen (best quality, ~+138% perf/Wh)
# First head pays a one-time ~60s Triton autotune cost; subsequent heads (same
# shape) reuse the cached graph (TORCHINDUCTOR_FX_GRAPH_CACHE=1 in rocm_env.yaml).
#
# Usage:  SEED=<best> PYTHON=sat-venv/bin/python ./run-final-retrain.sh
set -u
PYTHON="${PYTHON:-python}"
SEED="${SEED:?must set best seed from the shootout}"
LOGDIR="latch_ship_retrain_s${SEED}"; mkdir -p "$LOGDIR"
COMMON="--subset-frac 1.0 --holdout-frac 0.05 --preload ram --dim 256 --num-heads 8 \
--batch-size 64 --epochs 40 --lr 3e-4 --optimizer adamw --seed $SEED --save-best-only \
--compile --t-injection adaln_zero"

# feature | extra-args | config
HEADS=(
  "rms_energy_bass||latch_train.yaml"
  "rms_energy_body||latch_train.yaml"
  "rms_energy_mid||latch_train.yaml"
  "rms_energy_air||latch_train.yaml"
  "spectral_flatness|--standardize|latch_train.yaml"
  "spectral_skewness|--standardize|latch_train.yaml"
  "spectral_flux||latch_train.yaml"
  "spectral_kurtosis||latch_train.yaml"
  "onsets_activations||latch_train_density.yaml"
  "hpcp||latch_train_hpcp.yaml"
)
# beat_activations deliberately SKIPPED — verified not a usable control (closed-loop 2026-05-26).

echo "=== ship retrain (seed $SEED, full data, 40ep, AdamW 3e-4) ==="
for h in "${HEADS[@]}"; do
  IFS='|' read -r feat extra cfg <<< "$h"
  tag="ship_${feat}_s${SEED}"
  echo "----- [$feat] -----"
  "$PYTHON" scripts/train_latch.py --config "$cfg" --feature "$feat" $extra $COMMON \
    --tag "$tag" > "$LOGDIR/${feat}.log" 2>&1
  echo "  best: $(grep 'new best (val_median=' "$LOGDIR/${feat}.log" | tail -1)"
done

echo ""
echo "===== SHIP-RETRAIN SUMMARY (seed=$SEED, full data, 40ep) ====="
for h in "${HEADS[@]}"; do
  IFS='|' read -r feat extra cfg <<< "$h"
  best=$(grep 'new best (val_median=' "$LOGDIR/${feat}.log" 2>/dev/null | tail -1)
  printf "  %-22s %s\n" "$feat" "$best"
done
echo "Checkpoints: latch_weights/latch_<feat>_ship_<feat>_s${SEED}_best.pt + _last.pt"
