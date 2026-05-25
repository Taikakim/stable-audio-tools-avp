#!/home/kim/Projects/SAO/stable-audio-tools/sat-venv/bin/python
"""
latent_server.py — HTTP server for decoding VAE latents to WAV audio.

Loads the Stable Audio Small autoencoder at startup, then serves decoded
WAV audio for latent .npy files on request.  Designed to back the latent
audio player embedded in feature_explorer.html.

Endpoints:
  GET /status
      → JSON: {ok, sample_rate, downsampling_ratio, latent_dir, device}

  GET /decode?track=TRACK_FOLDER_NAME&position=0.5
      Finds the crop whose companion .json "position" field is closest
      to the requested value (0.0–1.0), decodes it through the VAE, and
      returns WAV audio (stereo 44100 Hz int16).
      Response headers:
        X-Crop-Count    — total crops in that track folder
        X-Crop-Position — actual position value of the chosen crop

  GET /crops?track=TRACK_FOLDER_NAME
      → JSON array of {path, position} for all full-mix crops in track.

Usage:
    source rocm_env.sh
    ./latent_server.py \\
        --latent-dir /run/media/kim/Lehto/goa-small \\
        --port 7891
"""

import os

# ROCm env (HIP/MIOpen/TunableOp) is applied here, before torch is imported.
import stable_audio_tools.rocm_env  # noqa: F401

import argparse
import configparser
import io
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from latent_crossfader import STEMS, crossfade_stems, lerp, load_latent, slerp
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

STEM_SUFFIXES = {"_bass", "_drums", "_other", "_vocals"}

REPO_ROOT = Path(__file__).parent
DEFAULT_MODEL_CONFIG = REPO_ROOT / "models/checkpoints/small/base_model_config.json"
DEFAULT_CKPT_PATH    = REPO_ROOT / "models/checkpoints/small/base_model.ckpt"

# ---------------------------------------------------------------------------
# Globals (set in main, read-only afterwards)
# ---------------------------------------------------------------------------
_autoencoder        = None
_sample_rate        = None
_downsampling_ratio = None
_latent_dir         = None
_stem_dir           = None   # separate root for stem latents
_dtype              = None
_device             = None
_decode_lock        = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_autoencoder(model_config_path: Path, ckpt_path: Path,
                     model_half: bool, device: str):
    with open(model_config_path) as f:
        model_config = json.load(f)

    sample_rate        = model_config["sample_rate"]
    sample_size        = model_config["sample_size"]
    downsampling_ratio = model_config["model"]["pretransform"]["config"]["downsampling_ratio"]

    print(f"Creating model from config ({model_config_path.name})")
    model = create_model_from_config(model_config)

    print(f"Loading checkpoint from {ckpt_path.name} ...")
    copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))

    if not hasattr(model, "pretransform") or model.pretransform is None:
        raise ValueError("Model has no pretransform — expected a diffusion_cond model")
    autoencoder = model.pretransform
    del model

    autoencoder.eval().requires_grad_(False)
    if model_half:
        autoencoder.to(torch.float16)
    autoencoder.to(device)

    print(f"Autoencoder ready — sr={sample_rate}, sample_size={sample_size}, "
          f"downsampling_ratio={downsampling_ratio}")
    return autoencoder, sample_rate, sample_size, downsampling_ratio


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

def is_full_mix(path: Path) -> bool:
    for s in STEM_SUFFIXES:
        if path.stem.endswith(s):
            return False
    return True


def find_crops(track_dir: Path) -> list:
    """Return list of (npy_path, position) sorted by position field."""
    crops = []
    for npy in sorted(track_dir.glob("*.npy")):
        if not is_full_mix(npy):
            continue
        json_path = npy.with_suffix(".json")
        position = 0.0
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                position = float(data.get("position", 0.0))
            except Exception:
                pass
        crops.append((npy, position))
    crops.sort(key=lambda x: x[1])
    return crops


def find_stem_crops(track_dir: Path, stem: str) -> list:
    """Return list of (npy_path, position) for a specific stem, sorted by position."""
    suffix = f"_{stem}"
    crops = []
    for npy in sorted(track_dir.glob("*.npy")):
        if not npy.stem.endswith(suffix):
            continue
        json_path = npy.with_suffix(".json")
        position = 0.0
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
                position = float(data.get("position", 0.0))
            except Exception:
                pass
        crops.append((npy, position))
    crops.sort(key=lambda x: x[1])
    return crops


def find_best_crop(crops: list, target_position: float):
    """Return (npy_path, position) with position closest to target."""
    if not crops:
        return None, None
    best = min(crops, key=lambda x: abs(x[1] - target_position))
    return best


# ---------------------------------------------------------------------------
# Smart loop helpers
# ---------------------------------------------------------------------------

def _nearest_zero_crossing(mono: np.ndarray, target: int, window: int) -> int:
    """Return the sample index of the zero-crossing nearest to `target`
    within ±window samples.  Uses the mono mix signal."""
    n  = len(mono)
    lo = max(0, target - window)
    hi = min(n - 2, target + window)
    seg = mono[lo:hi + 1]
    # indices where consecutive samples have opposite (or zero) sign
    crossings = np.where(seg[:-1] * seg[1:] <= 0)[0] + lo
    if len(crossings) == 0:
        return target
    return int(crossings[np.argmin(np.abs(crossings - target))])


def _smart_loop_points(audio_np: np.ndarray, sample_rate: int,
                       bpm: float) -> tuple:
    """Return (start, end) sample indices for a loop whose length is a
    multiple of 4 bars (4 downbeats), bounded by zero-crossings."""
    n          = audio_np.shape[1]
    beat_s     = int(round(60.0 / bpm * sample_rate))
    bar_s      = 4 * beat_s          # samples per 4/4 bar
    four_bar_s = 4 * bar_s           # samples per 4-bar phrase

    # Largest multiple of 4 bars that fits in the available content
    n_phrases  = max(1, n // four_bar_s)
    loop_len   = n_phrases * four_bar_s

    mono   = (audio_np[0] + audio_np[1]) / 2.0
    window = beat_s   # search ±1 beat around each boundary

    start = _nearest_zero_crossing(mono, 0, window)
    end   = _nearest_zero_crossing(mono, min(start + loop_len, n - 1), window)
    end   = min(end, n)

    if end <= start:          # safety fallback
        return 0, n

    duration_s = (end - start) / sample_rate
    n_bars     = round((end - start) / bar_s)
    print(f"  smart loop: {n_bars} bars  {duration_s:.2f}s  "
          f"[{start}–{end}]  bpm={bpm:.1f}")
    return start, end


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_to_wav(npy_path: Path, smart_loop: bool = False, manip_channels: list = None, manip_amounts: list = None) -> bytes:
    """Decode a latent .npy to WAV bytes. Call with _decode_lock held."""
    latent   = np.load(str(npy_path)).astype(np.float32)   # [64, L]
    latent_t = torch.from_numpy(latent).unsqueeze(0).to(device=_device, dtype=_dtype)

    if manip_channels and manip_amounts and len(manip_channels) == len(manip_amounts):
        for ch, amt in zip(manip_channels, manip_amounts):
            if 0 <= ch < 64:
                # latent_t is [1, 64, L]
                latent_t[0, ch, :] += amt

    with torch.no_grad():
        audio = _autoencoder.decode(latent_t)   # [1, 2, samples]

    audio_np = audio.squeeze(0).cpu().float().numpy()   # [2, samples]

    # Read companion JSON once for both padding_mask and bpm
    companion = {}
    json_path = npy_path.with_suffix(".json")
    if json_path.exists():
        try:
            companion = json.loads(json_path.read_text())
        except Exception:
            pass

    # Trim silence using padding_mask
    mask = companion.get("padding_mask", [])
    if mask:
        n_content    = sum(mask)
        actual_samples = n_content * _downsampling_ratio
        if 0 < actual_samples < audio_np.shape[1]:
            audio_np = audio_np[:, :actual_samples]

    # Smart loop: trim to a musically aligned loop region
    if smart_loop:
        raw_bpm = companion.get("bpm")
        if raw_bpm:
            try:
                start, end = _smart_loop_points(audio_np, _sample_rate, float(raw_bpm))
                audio_np = audio_np[:, start:end]
            except Exception as e:
                print(f"  smart_loop error: {e}")

    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)   # [2, samples]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)     # int16
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Crossfade decode
# ---------------------------------------------------------------------------

def crossfade_to_wav(
    track_a: str, track_b: str,
    pos_a: float, pos_b: float,
    alphas: dict, beta_a: float, beta_b: float,
    interp: str = 'slerp',
) -> tuple:
    """Decode a stem crossfade to WAV bytes. Call with _decode_lock held.

    Returns:
        (wav_bytes, stems_found) where stems_found is a list of stems decoded.
    """
    stem_root = _stem_dir if _stem_dir is not None else _latent_dir
    dir_a = stem_root / track_a
    dir_b = stem_root / track_b

    # Load stem latents for both tracks
    stems_a = {}
    stems_b = {}
    missing = []

    for stem in STEMS:
        crops_a = find_stem_crops(dir_a, stem)
        crops_b = find_stem_crops(dir_b, stem)

        npy_a, _ = find_best_crop(crops_a, pos_a) if crops_a else (None, None)
        npy_b, _ = find_best_crop(crops_b, pos_b) if crops_b else (None, None)

        if npy_a is None or npy_b is None:
            missing.append(stem)
            continue

        stems_a[stem] = load_latent(npy_a, device=_device, dtype=_dtype)
        stems_b[stem] = load_latent(npy_b, device=_device, dtype=_dtype)

    if missing:
        raise ValueError(f"Stems not found for: {', '.join(missing)}")

    # Load full-mix latents for β color anchors (always from _latent_dir, not stem_root)
    crops_fm_a = find_crops(_latent_dir / track_a)
    crops_fm_b = find_crops(_latent_dir / track_b)
    npy_fm_a, _ = find_best_crop(crops_fm_a, pos_a)
    npy_fm_b, _ = find_best_crop(crops_fm_b, pos_b)

    if npy_fm_a is None or npy_fm_b is None:
        raise ValueError("Full-mix crops not found for one or both tracks")

    fullmix_a = load_latent(npy_fm_a, device=_device, dtype=_dtype)
    fullmix_b = load_latent(npy_fm_b, device=_device, dtype=_dtype)

    audio = crossfade_stems(
        stems_a, stems_b, fullmix_a, fullmix_b,
        alphas, beta_a, beta_b,
        _autoencoder.decode,
        interp=interp,
        device=_device,
    )

    audio_np  = audio.squeeze(0).cpu().float().numpy()   # [2, samples]
    audio_np  = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        wf.writeframes(audio_i16.T.flatten().tobytes())

    return buf.getvalue(), list(stems_a.keys())


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs     = parse_qs(parsed.query)

        if parsed.path == "/status":
            self._handle_status()
        elif parsed.path == "/decode":
            track      = qs.get("track",      [""])[0]
            position   = float(qs.get("position",   ["0.5"])[0])
            smart_loop = qs.get("smart_loop", ["0"])[0] == "1"
            
            manip_channels = []
            manip_amounts = []
            if "manip_channels" in qs and "manip_amounts" in qs:
                try:
                    manip_channels = [int(c) for c in qs.get("manip_channels")[0].split(",")]
                    manip_amounts = [float(a) for a in qs.get("manip_amounts")[0].split(",")]
                except ValueError:
                    pass

            self._handle_decode(track, position, smart_loop, manip_channels, manip_amounts)
        elif parsed.path == "/crops":
            track = qs.get("track", [""])[0]
            self._handle_crops(track)
        elif parsed.path == "/crossfade":
            self._handle_crossfade(qs)
        else:
            self.send_response(404)
            self._cors()
            self.end_headers()

    # ---- /status ---------------------------------------------------------------
    def _handle_status(self):
        body = json.dumps({
            "ok":               True,
            "sample_rate":      _sample_rate,
            "downsampling_ratio": _downsampling_ratio,
            "latent_dir":       str(_latent_dir),
            "device":           str(_device),
        }).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- /crops ----------------------------------------------------------------
    def _handle_crops(self, track: str):
        if not track:
            self._error(400, "Missing track parameter")
            return
        track_dir = _latent_dir / track
        if not track_dir.is_dir():
            self._error(404, f"Track not found: {track}")
            return
        crops = find_crops(track_dir)
        body  = json.dumps([{"path": c[0].name, "position": c[1]} for c in crops]).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- /decode ---------------------------------------------------------------
    def _handle_decode(self, track: str, position: float, smart_loop: bool = False, manip_channels: list = None, manip_amounts: list = None):
        if not track:
            self._error(400, "Missing track parameter")
            return
        track_dir = _latent_dir / track
        if not track_dir.is_dir():
            self._error(404, f"Track not found: {track}")
            return
        crops = find_crops(track_dir)
        if not crops:
            self._error(404, "No crops found for track")
            return

        best_npy, best_pos = find_best_crop(crops, position)
        print(f"  decode: {track}  pos={position:.3f}  smart_loop={smart_loop}"
              f"  → {best_npy.name} (pos={best_pos:.3f})")

        try:
            with _decode_lock:
                wav_bytes = decode_to_wav(best_npy, smart_loop=smart_loop, manip_channels=manip_channels, manip_amounts=manip_amounts)
        except Exception as e:
            self._error(500, str(e))
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",    "audio/wav")
        self.send_header("Content-Length",  str(len(wav_bytes)))
        self.send_header("X-Crop-Count",    str(len(crops)))
        self.send_header("X-Crop-Position", f"{best_pos:.4f}")
        self.end_headers()
        self.wfile.write(wav_bytes)

    # ---- /crossfade ------------------------------------------------------------
    def _handle_crossfade(self, qs: dict):
        track_a = qs.get("track_a", [""])[0]
        track_b = qs.get("track_b", [""])[0]
        if not track_a or not track_b:
            self._error(400, "Missing track_a or track_b parameter")
            return
        stem_root = _stem_dir if _stem_dir is not None else _latent_dir
        dir_a = stem_root / track_a
        dir_b = stem_root / track_b
        if not dir_a.is_dir():
            self._error(404, f"Stem track not found: {track_a}")
            return
        if not dir_b.is_dir():
            self._error(404, f"Stem track not found: {track_b}")
            return

        pos_a  = float(qs.get("pos_a",  ["0.5"])[0])
        pos_b  = float(qs.get("pos_b",  ["0.5"])[0])
        beta_a = float(qs.get("beta_a", ["0.0"])[0])
        beta_b = float(qs.get("beta_b", ["0.0"])[0])
        interp = qs.get("interp", ["slerp"])[0]
        if interp not in ("slerp", "lerp"):
            interp = "slerp"

        alphas = {
            "drums":  float(qs.get("drums",  ["0.0"])[0]),
            "bass":   float(qs.get("bass",   ["0.0"])[0]),
            "other":  float(qs.get("other",  ["0.0"])[0]),
            "vocals": float(qs.get("vocals", ["0.0"])[0]),
        }

        print(f"  crossfade: A={track_a} B={track_b} "
              f"pos_a={pos_a:.3f} pos_b={pos_b:.3f} "
              f"alphas={alphas} beta_a={beta_a:.3f} beta_b={beta_b:.3f} interp={interp}")

        try:
            with _decode_lock:
                wav_bytes, stems_found = crossfade_to_wav(
                    track_a, track_b, pos_a, pos_b,
                    alphas, beta_a, beta_b, interp=interp,
                )
        except ValueError as e:
            self._error(500, str(e))
            return
        except Exception as e:
            self._error(500, str(e))
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type",   "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-Stems-Found",  ",".join(stems_found))
        self.end_headers()
        self.wfile.write(wav_bytes)

    # ---- helpers ---------------------------------------------------------------
    def _error(self, code: int, msg: str):
        body = msg.encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type",   "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  [{self.address_string()}] {fmt % args}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _autoencoder, _sample_rate, _downsampling_ratio
    global _latent_dir, _stem_dir, _dtype, _device

    parser = argparse.ArgumentParser(description="Latent audio decode server")
    parser.add_argument("--config",       default="latent_player.ini",
                        help="Path to .ini config file (default: latent_player.ini)")
    parser.add_argument("--latent-dir",   default=None,
                        help="Root of the pre-encoded full-mix dataset (overrides ini)")
    parser.add_argument("--stem-dir",     default=None,
                        help="Root of the pre-encoded stem dataset (overrides ini)")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--ckpt-path",    default=None)
    parser.add_argument("--port",         type=int, default=None)
    parser.add_argument("--model-half",    action="store_true", default=None)
    parser.add_argument("--no-model-half", dest="model_half", action="store_false")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    # Read ini file (CLI args override)
    ini = configparser.ConfigParser()
    ini_path = Path(args.config)
    if ini_path.exists():
        ini.read(str(ini_path))
        print(f"Config: {ini_path}")
    elif args.config != "latent_player.ini":
        print(f"Error: config file not found: {ini_path}", file=sys.stderr)
        sys.exit(1)

    def ini_get(section, key, fallback=None):
        return ini.get(section, key, fallback=fallback)

    latent_dir_str   = args.latent_dir   or ini_get("server", "latent_dir")
    stem_dir_str     = args.stem_dir     or ini_get("server", "stem_dir")
    model_config_str = args.model_config or ini_get("model",  "model_config",
                                                     fallback=str(DEFAULT_MODEL_CONFIG))
    ckpt_path_str    = args.ckpt_path    or ini_get("model",  "ckpt_path",
                                                     fallback=str(DEFAULT_CKPT_PATH))
    port             = args.port         or int(ini_get("server", "port", fallback="7891"))
    device           = args.device       or ini_get("model",  "device", fallback="cuda")

    if args.model_half is None:
        model_half = ini_get("model", "model_half", fallback="true").lower() != "false"
    else:
        model_half = args.model_half

    if not latent_dir_str:
        print("Error: latent_dir not set (use --latent-dir or set in latent_player.ini)",
              file=sys.stderr)
        sys.exit(1)

    _latent_dir       = Path(latent_dir_str)
    _stem_dir         = Path(stem_dir_str) if stem_dir_str else None
    model_config_path = Path(model_config_str)
    ckpt_path         = Path(ckpt_path_str)

    for p, label in [(_latent_dir, "latent-dir"),
                     (model_config_path, "model-config"),
                     (ckpt_path, "ckpt-path")]:
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    _device = device
    if _device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        _device = "cpu"

    _autoencoder, _sample_rate, _, _downsampling_ratio = load_autoencoder(
        model_config_path, ckpt_path, model_half, _device
    )
    _dtype = torch.float16 if model_half else torch.float32

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\nLatent server running at http://localhost:{port}")
    print(f"  Latent dir : {_latent_dir}")
    if _stem_dir:
        print(f"  Stem dir   : {_stem_dir}")
    print(f"  Device     : {_device}  half={model_half}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
