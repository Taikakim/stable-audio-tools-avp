#!/usr/bin/env bash
# run-sa-finetune.sh — orchestrate the two-stage SA Open Small Goa-Trance finetune.
#
#   bash run-sa-finetune.sh stage1                # train the RF base
#   bash run-sa-finetune.sh unwrap PATH/TO/CKPT   # unwrap a stage1 ckpt -> inference ckpt
#   bash run-sa-finetune.sh stage2 PATH/TO/UNWRAPPED_STAGE1.ckpt   # ARC distill
#   bash run-sa-finetune.sh all                   # do all of the above end-to-end
#
# Stage 2 needs the Stage 1 *unwrapped* (inference) checkpoint patched into
# two places: the .ini's `pretrained_ckpt_path` and the model config's
# `arc.discriminator_base_ckpt`. We patch a sibling copy rather than touching
# the version-controlled files so reruns and git diffs stay clean.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY=${PY:-python3}
S1_INI="configs/finetune/small_stage1.ini"
S2_INI_SRC="configs/finetune/small_stage2.ini"
S2_CFG_SRC="configs/finetune/small_stage2_model_config.json"
DATASET="configs/finetune/small_finetune_dataset.json"
S1_MODEL_CFG="configs/finetune/small_stage1_model_config.json"

cmd=${1:-help}

usage() {
    grep -E '^#( |$)' "$0" | sed 's/^# \?//'
    exit 1
}

run_stage1() {
    echo ">>> Stage 1: finetune rectified-flow base"
    # --name and --compile must be passed on the CLI: prefigure ignores the ini's
    # `name` key (registers --name with default=None) and hardcodes default=False
    # for booleans (the ini `compile` value is ignored). Drop --compile to disable.
    #
    # MIOpen find mode comes from rocm_env.yaml's `training` profile (HYBRID=3).
    # Modes 5/6/7 skip non-dynamic kernels; the VAE conv1d has no dynamic solver on
    # gfx1201, so those leave an empty solution list and MIOpen aborts. 1/2/3 keep
    # the non-dynamic solvers and work — 3 does a real find on a miss (good kernels).
    GOA_SPLIT_MODE=train $PY train.py --config-file "$S1_INI" --name sa_small_stage1 --compile
}

run_unwrap() {
    local wrapped="${1:?path to wrapped Stage 1 ckpt}"
    local name="${2:-sa_small_stage1_unwrapped}"
    echo ">>> Unwrap: $wrapped -> ${name}.ckpt"
    $PY unwrap_model.py --model-config "$S1_MODEL_CFG" --ckpt-path "$wrapped" --name "$name"
}

run_stage2() {
    local stage1_ckpt="${1:?path to UNWRAPPED Stage 1 ckpt (use 'unwrap' first)}"
    if [[ ! -f "$stage1_ckpt" ]]; then
        echo "Stage 1 ckpt not found: $stage1_ckpt" >&2; exit 2
    fi
    local stage1_abs; stage1_abs="$(realpath "$stage1_ckpt")"

    local out_dir=".sa_small_stage2_patched"
    mkdir -p "$out_dir"
    local s2_ini="${out_dir}/small_stage2.ini"
    local s2_cfg="${out_dir}/small_stage2_model_config.json"

    # Sed-patch placeholders. Use | as delimiter so absolute paths with /
    # don't break the regex.
    sed "s|STAGE1_CKPT_PLACEHOLDER|${stage1_abs}|g" "$S2_INI_SRC" > "$s2_ini"
    sed "s|STAGE1_CKPT_PLACEHOLDER|${stage1_abs}|g" "$S2_CFG_SRC" > "$s2_cfg"

    # Point the .ini at the patched model_config rather than the unpatched one.
    sed -i "s|configs/finetune/small_stage2_model_config.json|${s2_cfg}|g" "$s2_ini"

    echo ">>> Stage 2: ARC distillation"
    echo "    stage1 ckpt:       $stage1_abs"
    echo "    patched .ini:      $s2_ini"
    echo "    patched modelcfg:  $s2_cfg"

    GOA_SPLIT_MODE=train $PY train.py --config-file "$s2_ini" --name sa_small_stage2
}

case "$cmd" in
    stage1)  run_stage1 ;;
    unwrap)  run_unwrap "${2:-}" "${3:-sa_small_stage1_unwrapped}" ;;
    stage2)  run_stage2 "${2:-}" ;;
    all)
        run_stage1
        # Locate the freshest Stage 1 checkpoint.
        latest=$(ls -t lightning_logs/*/checkpoints/last.ckpt 2>/dev/null | head -n1 || true)
        if [[ -z "$latest" ]]; then echo "No Stage 1 ckpt found under lightning_logs/" >&2; exit 3; fi
        run_unwrap "$latest"
        run_stage2 "sa_small_stage1_unwrapped.ckpt"
        ;;
    help|--help|-h|"") usage ;;
    *) echo "Unknown command: $cmd" >&2; usage ;;
esac
