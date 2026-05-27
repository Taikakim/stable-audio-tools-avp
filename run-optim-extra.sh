#!/usr/bin/env bash
# Extra optimizer runs: 8-bit Adam (bnb) + Schedule-Free AdamW.
# Same base as the matrix (onset-density, dim256, batch64, gaussian sigma3, standardized,
# 30% subset, shared 5% holdout, preload gpu, save-best-only) so val_median is comparable
# to the Phase-1 leaderboard (Prodigy 0.149 / AdamW-3e4 0.153 baseline).
#
# Schedule-Free uses NO LR schedule (built-in warmup + weight averaging; trainer handles
# .train()/.eval()). 8-bit Adam is a memory-economy drop-in (quality ~ AdamW expected).
#
# Run:  PYTHON=sat-venv/bin/python ./run-optim-extra.sh

set -u
PYTHON="${PYTHON:-python}"
LOGDIR="latch_optim_extra_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
COMMON="--config latch_train_density.yaml --feature onsets_activations --smooth-kind gaussian \
--subset-frac 0.3 --holdout-frac 0.05 --preload gpu --dim 256 --num-heads 8 --batch-size 64 \
--save-best-only"

# tag | optimizer | lr | epochs
RUNS=(
  "schedfree_lr3e4|schedulefree|3e-4|30"
  "schedfree_lr1e3|schedulefree|1e-3|30"
  "schedfree_lr3e4_60|schedulefree|3e-4|60"
  "adam8bit_lr3e4|adam8bit|3e-4|30"
)

run() {
    local tag="$1"; shift; local log="$LOGDIR/${tag}.log"
    echo "=== [$tag] ===" | tee -a "$SUMMARY"; echo "CMD: $*" | tee -a "$SUMMARY"
    stdbuf -oL "$@" 2>&1 | tee "$log"
    echo "best: $(grep 'new best (val_median=' "$log" | tail -n 1)" | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"
}

for entry in "${RUNS[@]}"; do
    IFS='|' read -r tag opt lr ep <<< "$entry"
    run "$tag" "$PYTHON" scripts/train_latch.py $COMMON \
        --optimizer "$opt" --lr "$lr" --epochs "$ep" --tag "optm_$tag"
done

echo "==================== EXTRA OPTIMIZERS (lower val_median = better; shared holdout) ====================" | tee -a "$SUMMARY"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r tag opt lr ep <<< "$entry"
    printf "  %-20s %s\n" "$tag" "$(grep 'new best (val_median=' "$LOGDIR/${tag}.log" 2>/dev/null | tail -n 1)" | tee -a "$SUMMARY"
done
echo "Checkpoints: latch_weights/latch_onsets_activations_optm_{schedfree*,adam8bit*}_best.pt" | tee -a "$SUMMARY"
