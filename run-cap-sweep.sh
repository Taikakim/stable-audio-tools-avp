#!/usr/bin/env bash
# Iso-quality energy-vs-power-cap study. Runs the energy-frontier configs at WHATEVER
# GPU power cap is currently set (set it externally first:
#   sudo rocm-smi --setpoweroverdrive <W>   # 212 = card minimum, 250, etc.)
# Same head/data/seed across caps -> val_median is identical, so this isolates the
# cap's effect on speed + energy (perf/watt). Tags + logs record the cap for provenance.
#
# Usage:  CAP=212 PYTHON=sat-venv/bin/python ./run-cap-sweep.sh
set -u
PYTHON="${PYTHON:-python}"
CAP="${CAP:-unknown}"
LOGDIR="latch_capsweep_${CAP}W"; mkdir -p "$LOGDIR"
# provenance: record the actual cap the driver reports
rocm-smi --showmaxpower 2>/dev/null | grep -iE "Power \(W\)" | head -1 | tee "$LOGDIR/_cap_readback.txt"

COMMON="--config latch_train_density.yaml --feature onsets_activations --smooth-kind gaussian \
--subset-frac 0.3 --holdout-frac 0.05 --preload gpu --batch-size 64 --save-best-only --epochs 30 --seed 0"

# Frontier dim256 configs (overhead-bound -> cap ~free) + ONE dim1024 run that is
# compute-bound -> the cap should BITE there. The contrast IS the result.
#   opt | lr | dim | heads
RUNS=( "lion|1e-5|256|8" "adamw|3e-4|256|8" "prodigy|1.0|256|8" "adamw|3e-4|1024|16" )
for e in "${RUNS[@]}"; do
  IFS='|' read -r opt lr dim heads <<< "$e"
  tag="${opt}_${lr/./}_d${dim}_cap${CAP}"
  echo "=== [$tag] @ ${CAP}W (dim$dim) ==="
  "$PYTHON" scripts/train_latch.py $COMMON --optimizer "$opt" --lr "$lr" \
      --dim "$dim" --num-heads "$heads" --tag "optm_$tag" > "$LOGDIR/$tag.log" 2>&1
  echo "best: $(grep 'new best (val_median=' "$LOGDIR/$tag.log" | tail -1)"
done
echo "CAP SWEEP @ ${CAP}W DONE -> $LOGDIR/"
