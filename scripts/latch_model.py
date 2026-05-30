# Legacy module shim. The LatCH model lives in stable_audio_tools.models.latch
# so the Gradio inference pipeline and the training scripts share one source of truth.
# Kept for compatibility with `from latch_model import LatCH` callsites.
from stable_audio_tools.models.latch import (  # noqa: F401
    LatCH,
    LatCHAttention,
    LatCHBlock,
    RotaryEmbedding,
    TimestepEmbedder,
    apply_rotary_emb,
    load_latch_from_checkpoint,
)
