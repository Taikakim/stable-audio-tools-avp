#!/usr/bin/env python3
"""LatCH head shape / throughput sweep — find the 'free capacity' knee on this GPU.

Synthetic: random (latent, target) preloaded on the GPU, so it isolates compute +
per-step overhead from the dataloader/disk entirely. Trains FP16 (autocast) like the
real trainer (fp32 params + autocast matmuls, AdamW, SmoothL1). Sweeps dim over
256-aligned widths (per the TunableOp report) and batch, reporting it/s, items/s
(= epoch throughput), params, and peak VRAM.

TunableOp is pointed at a throwaway CSV so it doesn't race/pollute the shared
7.2.3 file the concurrent MIR job may touch; TUNING stays on so numbers reflect
tuned kernels (the first warmup steps absorb the per-shape tuning sweep).
"""
import os
import sys
import time
import importlib.util

os.environ.setdefault("PYTORCH_TUNABLEOP_FILENAME", "/tmp/latch_bench_tunings.csv")
_re = "/home/kim/Projects/SAO/stable-audio-tools/stable_audio_tools/rocm_env.py"
_spec = importlib.util.spec_from_file_location("re_", _re)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_m.apply_profile("training", verbose=False)

import torch
import torch.nn as nn

sys.path.append("/home/kim/Projects/SAO/stable-audio-tools/scripts")
from latch_model import LatCH

DEV = "cuda"


def bench(dim, depth, heads, batch, seq=256, in_ch=64, out_ch=1, warmup=15, iters=60):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = LatCH(in_channels=in_ch, out_channels=out_ch,
                      dim=dim, depth=depth, num_heads=heads).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        x = torch.randn(batch, in_ch, seq, device=DEV)
        tgt = torch.randn(batch, out_ch, seq, device=DEV)
        t = torch.rand(batch, device=DEV)
        crit = nn.SmoothL1Loss()

        def step():
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x, t), tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        its = iters / dt
        params = sum(p.numel() for p in model.parameters()) / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, opt, x, tgt, t
        return its, its * batch, params, peak
    except RuntimeError as e:
        return None, None, None, str(e)[:40]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true",
                    help="dim x batch matrix — the joint speed optimum (B* drops as dim grows)")
    args = ap.parse_args()
    print(f"{'dim':>5} {'heads':>5} {'depth':>5} {'batch':>6} "
          f"{'params(M)':>9} {'it/s':>8} {'items/s':>9} {'VRAM(GB)':>8}")
    if args.grid:
        # every (dim, batch) pair — read off the best batch per width from items/s
        dims, batches = (256, 512, 768, 1024, 1536), (1, 4, 16, 64, 256)
        configs = [(d, 6, max(4, d // 64), b) for d in dims for b in batches]
    else:
        # head_dim 64 (WMMA-friendly) for the dim sweep; batch 64 = current
        dim_sweep = [(d, 6, max(4, d // 64), 64) for d in (256, 512, 768, 1024, 1536)]
        # batch doubling 1→512 at dim 256 (overhead→compute knee)
        batch_sweep = [(256, 6, 8, b) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)]
        configs = dim_sweep + batch_sweep
    for d, depth, h, b in configs:
        its, ips, params, peak = bench(d, depth, h, b)
        if its is None:
            print(f"{d:5d} {h:5d} {depth:5d} {b:6d} {'—':>9} {'OOM/err':>8}  {peak}")
        else:
            print(f"{d:5d} {h:5d} {depth:5d} {b:6d} {params:9.1f} "
                  f"{its:8.1f} {ips:9.0f} {peak:8.2f}")


if __name__ == "__main__":
    main()
