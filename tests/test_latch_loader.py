"""Loader tests for LatCH checkpoint format (legacy + new). Run: `python tests/test_latch_loader.py`."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_audio_tools.models.latch import LatCH, load_latch_from_checkpoint


def _make_state_dict():
    m = LatCH(in_channels=64, out_channels=1, dim=64, depth=2, num_heads=4)
    return m.state_dict()


def test_load_legacy_bare_state_dict():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "legacy.pt"
        torch.save(_make_state_dict(), p)
        m = load_latch_from_checkpoint(str(p), device="cpu")
        assert isinstance(m, LatCH)
        assert m.metadata == {}, f"legacy metadata must be empty, got {m.metadata}"


def test_load_new_format_with_metadata():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "new.pt"
        payload = {
            "state_dict": _make_state_dict(),
            "feature_name": "rms_energy_bass_ts",
            "feature_stats": {"mean": 0.05, "std": 0.02, "min": 0.0, "max": 0.3},
            "target_kind_default": "ramp_up",
        }
        torch.save(payload, p)
        m = load_latch_from_checkpoint(str(p), device="cpu")
        assert m.metadata["feature_name"] == "rms_energy_bass_ts"
        assert m.metadata["feature_stats"]["max"] == 0.3
        assert m.metadata["target_kind_default"] == "ramp_up"


def test_loaded_weights_match():
    sd = _make_state_dict()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.pt"
        torch.save({"state_dict": sd, "feature_name": "x"}, p)
        m = load_latch_from_checkpoint(str(p), device="cpu")
        for k, v in sd.items():
            assert torch.allclose(m.state_dict()[k], v), f"mismatch on {k}"


if __name__ == "__main__":
    test_load_legacy_bare_state_dict()
    test_load_new_format_with_metadata()
    test_loaded_weights_match()
    print("All loader tests passed.")
