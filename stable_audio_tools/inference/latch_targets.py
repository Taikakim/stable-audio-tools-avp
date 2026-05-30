"""Target tensor builders for LatCH guidance.

Replaces a constant scalar target with feature-typed shapes that match the
distributions LatCH heads were trained on (beat grids, RMS envelopes, etc).

build_target(kind, value, batch_size, channels, frames, *, fps, device, dtype)
    -> torch.Tensor of shape [batch_size, channels, frames]
"""
import math
import torch

KINDS = ("constant", "ramp_up", "ramp_down", "beat_grid", "chroma_major", "chroma_minor")

# Scale degrees (semitone offsets from the root) over the 12-bin chroma C,C#,…,B.
# A binary scale-tone mask is the target *direction* for hpcp's cosine guidance —
# "put the harmony in this key". value = root pitch class (0=C, 1=C#, …, 11=B).
_SCALE_DEGREES = {
    "chroma_major": (0, 2, 4, 5, 7, 9, 11),
    "chroma_minor": (0, 2, 3, 5, 7, 8, 10),  # natural minor
}


def _chroma_vec(kind, root):
    v = [0.0] * 12
    for d in _SCALE_DEGREES[kind]:
        v[(root + d) % 12] = 1.0
    return v


def build_target(kind, value, batch_size, channels, frames, *, fps, device, dtype):
    if kind not in KINDS:
        raise ValueError(f"unknown target kind '{kind}', expected one of {KINDS}")

    if kind == "constant":
        return torch.full((batch_size, channels, frames), float(value),
                          device=device, dtype=dtype)

    if kind in ("ramp_up", "ramp_down"):
        ramp = torch.linspace(0.0, float(value), frames, device=device, dtype=dtype)
        if kind == "ramp_down":
            ramp = ramp.flip(0)
        return ramp.view(1, 1, frames).expand(batch_size, channels, frames).contiguous()

    if kind in ("chroma_major", "chroma_minor"):
        # Target a key/scale for the 12-bin hpcp head. value = root pitch class (0–11).
        if channels != 12:
            raise ValueError(f"{kind} requires a 12-channel (hpcp) head, got channels={channels}")
        root = int(round(float(value))) % 12
        vec = torch.tensor(_chroma_vec(kind, root), device=device, dtype=dtype)  # [12]
        return vec.view(1, 12, 1).expand(batch_size, 12, frames).contiguous()

    if kind == "beat_grid":
        bpm = float(value)
        if bpm <= 0:
            raise ValueError(f"beat_grid requires bpm > 0, got {bpm}")
        frames_per_beat = 60.0 * float(fps) / bpm
        n_beats = int(math.floor(frames / frames_per_beat)) + 2
        idx = torch.tensor(
            [round(i * frames_per_beat) for i in range(n_beats)],
            dtype=torch.long,
        )
        idx = idx[(idx >= 0) & (idx < frames)]
        out = torch.zeros((batch_size, channels, frames), device=device, dtype=dtype)
        out[:, :, idx] = 1.0
        return out

    raise AssertionError(f"unhandled kind '{kind}'")  # unreachable
