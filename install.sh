#!/usr/bin/env bash
# stable-audio-tools/install.sh — set up SAT on the production stack:
# Python 3.10 + torch 2.10 ROCm 7.2.3. This is the known-good combo for SAT's
# LatCH training, FusionOpt, SAO-Small finetune, and audition renders.
#
# Standalone usage:
#   git clone <this-repo> && cd stable-audio-tools && ./install.sh
#
# Options:
#   --venv PATH        venv path (default: sat-venv  — matches MASTER §3)
#   --rocm-index URL   override torch index (default: PyTorch ROCm 7.2 nightly)
#   --help
#
# What this does:
#   1.  Sanity-check uv + python 3.10.
#   2.  Create the venv at sat-venv/ with python3.10.
#   3.  Install torch / torchaudio / torchvision / triton for ROCm 7.2 from
#       the PyTorch nightly index (same source mir's
#       requirements.distributable.txt points at).
#   4.  pip install -e . — picks up SAT's setup.py install_requires.
#
# Notes on flash-attn for SAT:
#   SAT routes through the Triton-AMD backend (aiter). It is NOT built here
#   because the prod 7.2.3 stack's CK path is the SA3 venv's concern. If you
#   need flash-attn in this venv:
#     uv pip install --python sat-venv/bin/python "flash-attn==2.8.3" \
#       --extra-index-url https://download.pytorch.org/whl/nightly/rocm7.2
#   …or build from source per the AMD prebuild wheel set. Triton-AMD is the
#   tried-and-true path for the 2.10 stack.

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
VENV="sat-venv"
ROCM_INDEX="https://download.pytorch.org/whl/nightly/rocm7.2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --rocm-index) ROCM_INDEX="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
command -v uv  &>/dev/null || { echo "ERROR: uv not in PATH (Arch: yay -S uv)" >&2; exit 1; }
command -v git &>/dev/null || { echo "ERROR: git not in PATH" >&2; exit 1; }

echo "==> SAT install — Python 3.10 / torch 2.10 ROCm 7.2.3 (prod stack)"
echo "    Repo:    $REPO"
echo "    Venv:    $REPO/$VENV"
echo "    Index:   $ROCM_INDEX"
echo

# ---------------------------------------------------------------------------
# 1. Venv
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv: $VENV (python 3.10)"
  uv venv --python 3.10 "$VENV"
else
  echo "==> Venv exists: $VENV"
fi
PY="$REPO/$VENV/bin/python"

# ---------------------------------------------------------------------------
# 2. ROCm torch stack
# ---------------------------------------------------------------------------
echo "==> Installing torch + ROCm 7.2 wheels"
# Use --extra-index-url here (not --index-url) because SAT's deps in setup.py
# need PyPI as well; the ROCm-named wheels (torch==X.Y.Z+rocm7.2) take
# priority when explicitly requested.
uv pip install --python "$PY" \
  --extra-index-url "$ROCM_INDEX" \
  torch torchaudio torchvision triton

# ---------------------------------------------------------------------------
# 3. SAT itself (editable; pulls install_requires from setup.py)
# ---------------------------------------------------------------------------
echo "==> Installing SAT (editable; resolves setup.py install_requires from PyPI)"
uv pip install --python "$PY" -e .

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "==> Verifying install"
"$PY" - <<'PY'
import torch
print(f"torch        {torch.__version__}")
print(f"HIP          {torch.version.hip}")
print(f"GPU          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
import stable_audio_tools as sat
print(f"SAT          {sat.__file__}")
try:
    from stable_audio_tools.training.fusion_opt import FusionOpt  # noqa: F401
    from stable_audio_tools.training.temporal_loss import TemporalShapeLoss  # noqa: F401
    print("FusionOpt    OK")
    print("TempShapeL.  OK")
except ImportError as e:
    print(f"NOTE: SAT training stack import failed: {e}")
PY

echo
echo "==> Done."
echo "    Activate:  source $VENV/bin/activate"
echo "    Notes:     SAO/MASTER.md §3 (venv-per-task), SAO/docs/latch.md (training recipe)"
