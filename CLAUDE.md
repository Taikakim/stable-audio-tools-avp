# CLAUDE.md

Guidance for Claude Code when working in the **audio-tools-AVP** fork of
`stable-audio-tools` (LatCH, FusionOpt, SAO-Small training, audition renders).

> **Cross-project coordination (read first).** This repo is one of three in the
> mir + Stable Audio pipeline. Shared facts — data paths, which venv for which
> task, gotchas that span repos — live in `/home/kim/Projects/SAO/MASTER.md`.
> Read it before cross-cutting work, and append to
> `/home/kim/Projects/SAO/WORKLOG.md` when you finish something another repo's
> agent would want to know.
@/home/kim/Projects/SAO/MASTER.md

## This repo at a glance

- **venv:** `sat-venv/bin/python` (Python 3.10, torch 2.10 ROCm). See MASTER §3.
- **ROCm env:** config-driven via `rocm_env.yaml` (repo root) + `stable_audio_tools/rocm_env.py`.
  Package `__init__` applies the `inference` profile on import; `scripts/train_latch.py`
  loads `rocm_env.py` standalone and applies `training`. Shell exports override the YAML
  (`setdefault`). Tunings: `~/pytorch-tunings-7.2.3` (torch 2.10).

## Key local docs (the durable record)

- **`LATCH_RESULTS.txt`** — the running experiment log (§1–23): smoothing/optimizer/
  architecture bake-offs, the validated ship recipe (§21: SF-NorMuon, d256/dp4, bf16,
  `--compile`, adaln_zero), energy/power studies, seed-variance findings. Read before
  proposing any LatCH training change.
- **`docs/FUSION_SHAREABLE.md`** — FusionOpt (Muon+MONA+KL-Shampoo+ScheduleFree) +
  TimeConditioningCache, what transfers to other models, what to skip.
- **`docs/fusion-optimiser.md`**, **`docs/specs/`** — design specs.

## LatCH essentials

- LatCH heads = small transformers that predict an MIR feature from a noisy latent and
  steer generation (training-free guidance). Targets come from mir (see MASTER §4).
- Training: `scripts/train_latch.py` (flags `--optimizer fusion --components ns5,normuon,sf
  --t-injection adaln_zero --compile --hot-dtype bf16`). Dataset: `scripts/latch_dataset.py`
  (`--target-source db|whole_track`, `--npz-root`).
- Audition: `scripts/render_audition*.py` → `renders/<set>/` (`manifest.json` + `index.html`).
  Score with mir's Audiobox (mir venv). See MASTER §4.

## Conventions

- Branch discipline: LatCH work has lived on `latch-rms-control`; merge to `main` and note
  it in `WORKLOG.md` so SA3 (which mirrors the LatCH model) stays in sync.
- When staging files for commit, stage only your own — leave unrelated working-tree changes.
