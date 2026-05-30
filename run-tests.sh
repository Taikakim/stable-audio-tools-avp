#!/usr/bin/env bash
set -u  # don't set -e — we want to continue even if one run fails

LOGDIR="latch_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"

run() {
    local tag="$1"; shift
    local log="$LOGDIR/${tag}.log"
    echo "=== [$tag] $(date -Iseconds) ===" | tee -a "$SUMMARY"
    echo "CMD: $*" | tee -a "$SUMMARY"
    # stdbuf -oL keeps Python's stdout line-buffered so tee writes promptly
    stdbuf -oL "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo "--- last 26 lines of [$tag] (exit=$rc) ---" | tee -a "$SUMMARY"
    tail -n 26 "$log" | tee -a "$SUMMARY"
    echo | tee -a "$SUMMARY"
}

# 1 — Onset-density bake-off
run dens_gauss3  python scripts/train_latch.py --config latch_train_density.yaml --smooth-kind gaussian      --tag dens_gauss3
run dens_lin3    python scripts/train_latch.py --config latch_train_density.yaml --smooth-kind linear        --tag dens_lin3
run dens_lp3     python scripts/train_latch.py --config latch_train_density.yaml --smooth-kind lowpass       --tag dens_lp3
run dens_beatw3  python scripts/train_latch.py --config latch_train_density.yaml --smooth-kind beat_weighted --tag dens_beatw3

# 2 — hpcp harmony head
run hpcp         python scripts/train_latch.py --config latch_train_hpcp.yaml

# 3 — Standardized retrains
run std_flat     python scripts/train_latch.py --config latch_train_spectral.yaml --feature spectral_flatness --standardize --tag std30
run std_skew     python scripts/train_latch.py --config latch_train_spectral.yaml --feature spectral_skewness --standardize --tag std30

echo "All runs complete. Logs in $LOGDIR/"
echo "Summary file: $SUMMARY"
