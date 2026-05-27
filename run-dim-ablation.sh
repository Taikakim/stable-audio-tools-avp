#!/usr/bin/env bash
# LatCH head WIDTH quality sweep: does a wider head lower val_median, and by how much?
#
# Same head (onset-density, gaussian sigma3, standardized), shared 5% holdout so
# val_median is comparable across widths, 30% subset, GPU-preloaded. Sweeps dim,
# ordered small→large so the important widths finish first.
#
# Batch is held FIXED (default 64) on purpose: it isolates the effect of WIDTH.
# Varying batch per dim would confound width with small-batch generalization. The
# speed-optimal batch per width is a separate question — see latch_shape_bench.py
# --grid (B* drops as dim grows; train the chosen width at its B*).
#
# Run:  ./run-dim-ablation.sh
#       DIMS="256 512 768 1024" EPOCHS=30 BATCH=64 ./run-dim-ablation.sh
#       PYTHON=sat-venv/bin/python ... ./run-dim-ablation.sh   (explicit interpreter)

set -u
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"
DIMS="${DIMS:-256 512 768 1024}"
LOGDIR="latch_dimsweep_$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"

run() {
    local tag="$1"; shift
    local log="$LOGDIR/${tag}.log"
    echo "=== [$tag] ===" | tee -a "$SUMMARY"
    echo "CMD: $*" | tee -a "$SUMMARY"
    stdbuf -oL "$@" 2>&1 | tee "$log"
    echo "best: $(grep 'new best (val_median=' "$log" | tail -n 1)" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

for d in $DIMS; do
    heads=$(( d / 64 )); [ "$heads" -lt 8 ] && heads=8   # head_dim<=32; keeps qkv aligned
    run "d${d}" "$PYTHON" scripts/train_latch.py --config latch_train_density.yaml \
        --feature onsets_activations --smooth-kind gaussian --subset-frac 0.3 \
        --holdout-frac 0.05 --preload gpu --epochs "$EPOCHS" \
        --dim "$d" --num-heads "$heads" --batch-size "$BATCH" --tag "dimsweep_d${d}"
done

echo "==== WIDTH SWEEP (batch=$BATCH, $EPOCHS ep; lower val_median better, shared holdout) ====" | tee -a "$SUMMARY"
for d in $DIMS; do
    printf "  dim %-5s %s\n" "$d" "$(grep 'new best (val_median=' "$LOGDIR/d${d}.log" 2>/dev/null | tail -n 1)" | tee -a "$SUMMARY"
done
echo "Logs in $LOGDIR/  — then verify the winner's control by ear / latch_verify_rms.py" | tee -a "$SUMMARY"
