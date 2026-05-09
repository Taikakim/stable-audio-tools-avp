"""Unit tests for LatCH target builders. Runnable directly: `python tests/test_latch_targets.py`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_audio_tools.inference.latch_targets import build_target, KINDS


def test_kinds_constant_listed():
    assert "constant" in KINDS
    assert "ramp_up" in KINDS
    assert "ramp_down" in KINDS
    assert "beat_grid" in KINDS


def test_constant_shape_and_value():
    out = build_target("constant", 0.5, batch_size=2, channels=1, frames=256,
                      fps=21.5, device="cpu", dtype=torch.float32)
    assert out.shape == (2, 1, 256), f"shape {out.shape}"
    assert torch.allclose(out, torch.full((2, 1, 256), 0.5))


def test_constant_multichannel():
    out = build_target("constant", 0.3, 1, 12, 100,
                      fps=21.5, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 12, 100)
    assert torch.allclose(out, torch.full((1, 12, 100), 0.3))


def test_ramp_up_endpoints():
    out = build_target("ramp_up", 1.0, 1, 1, 10,
                      fps=21.5, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 1, 10)
    assert out[0, 0, 0].item() == 0.0
    assert abs(out[0, 0, -1].item() - 1.0) < 1e-6


def test_ramp_down_endpoints():
    out = build_target("ramp_down", 2.0, 1, 1, 10,
                      fps=21.5, device="cpu", dtype=torch.float32)
    assert out[0, 0, 0].item() == 2.0
    assert abs(out[0, 0, -1].item() - 0.0) < 1e-6


def test_beat_grid_at_120_bpm():
    # 256 frames at 21.5 fps = 11.91s; 120 BPM = 2 beats/s -> ~24 beats
    out = build_target("beat_grid", 120.0, 1, 1, 256,
                      fps=21.5, device="cpu", dtype=torch.float32)
    assert out.shape == (1, 1, 256)
    n_hits = (out > 0.5).sum().item()
    assert 22 <= n_hits <= 26, f"expected ~24 beats at 120 BPM, got {n_hits}"
    binary = ((out < 1e-6) | ((out - 1.0).abs() < 1e-6)).all().item()
    assert binary, "beat_grid values must be exactly 0 or 1"


def test_beat_grid_zero_bpm_raises():
    try:
        build_target("beat_grid", 0.0, 1, 1, 10, fps=21.5, device="cpu", dtype=torch.float32)
    except ValueError:
        return
    raise AssertionError("expected ValueError for bpm <= 0")


def test_unknown_kind_raises():
    try:
        build_target("garbage", 1.0, 1, 1, 10, fps=21.5, device="cpu", dtype=torch.float32)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kind")


if __name__ == "__main__":
    test_kinds_constant_listed()
    test_constant_shape_and_value()
    test_constant_multichannel()
    test_ramp_up_endpoints()
    test_ramp_down_endpoints()
    test_beat_grid_at_120_bpm()
    test_beat_grid_zero_bpm_raises()
    test_unknown_kind_raises()
    print("All target-builder tests passed.")
