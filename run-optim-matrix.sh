#!/usr/bin/env bash
# Optimizer / LR / scheduler matrix around the settled optimum.
#
# Base (fixed): onset-density head, dim 256, batch 64, gaussian sigma3, standardized,
# 30% subset, shared 5% holdout (val_median comparable across ALL runs), GPU-preloaded,
# --save-best-only (keeps <stem>_best.pt per run so each can be verified by ear).
#
# Phase 1 — 30ep, constant LR: bracket AdamW LR, plus Lion (lower LR) and Prodigy (LR-free).
# Phase 2 — 60ep, COSINE schedule: the promising optimizers given a longer cosine run.
#
# Compare best val_median across all (same holdout). Checkpoints: latch_weights/
# latch_onsets_activations_<tag>_best.pt.
#
# Run:  PYTHON=sat-venv/bin/python ./run-optim-matrix.sh

set -u
PYTHON="${PYTHON:-python}"
LOGDIR="latch_optmatrix_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"

COMMON="--config latch_train_density.yaml --feature onsets_activations --smooth-kind gaussian \
--subset-frac 0.3 --holdout-frac 0.05 --preload gpu --dim 256 --num-heads 8 --batch-size 64 \
--save-best-only"

# tag | optimizer | lr | epochs | scheduler
RUNS=(
  "adamw_lr1e4|adamw|1e-4|30|none"
  "adamw_lr3e4|adamw|3e-4|30|none"
  "adamw_lr1e3|adamw|1e-3|30|none"
  "lion_lr3e5|lion|3e-5|30|none"
  "lion_lr1e4|lion|1e-4|30|none"
  "prodigy_lr1|prodigy|1.0|30|none"
  "adamw_lr3e4_cos60|adamw|3e-4|60|cosine"
  "adamw_lr1e3_cos60|adamw|1e-3|60|cosine"
  "lion_lr1e4_cos60|lion|1e-4|60|cosine"
  "prodigy_cos60|prodigy|1.0|60|cosine"
)

run() {
    local tag="$1"; shift
    local log="$LOGDIR/${tag}.log"
    echo "=== [$tag] $(date -Iseconds 2>/dev/null) ===" | tee -a "$SUMMARY"
    echo "CMD: $*" | tee -a "$SUMMARY"
    stdbuf -oL "$@" 2>&1 | tee "$log"
    echo "best: $(grep 'new best (val_median=' "$log" | tail -n 1)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

for entry in "${RUNS[@]}"; do
    IFS='|' read -r tag opt lr ep sched <<< "$entry"
    run "$tag" "$PYTHON" scripts/train_latch.py $COMMON \
        --optimizer "$opt" --lr "$lr" --epochs "$ep" --scheduler "$sched" \
        --tag "optm_$tag"
done

echo "==================== OPTIMIZER MATRIX (lower val_median = better; shared holdout) ====================" | tee -a "$SUMMARY"
for entry in "${RUNS[@]}"; do
    IFS='|' read -r tag opt lr ep sched <<< "$entry"
    printf "  %-22s %s\n" "$tag" "$(grep 'new best (val_median=' "$LOGDIR/${tag}.log" 2>/dev/null | tail -n 1)" | tee -a "$SUMMARY"
done
echo "Checkpoints: latch_weights/latch_onsets_activations_optm_*_best.pt  — verify by ear." | tee -a "$SUMMARY"
