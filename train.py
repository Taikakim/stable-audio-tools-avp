import matplotlib
matplotlib.use('Agg')  # headless backend to avoid segfault on ROCm

# Apply the ROCm/AMD *training* profile from rocm_env.yaml BEFORE torch (and before
# the stable_audio_tools package __init__, which would otherwise apply the
# *inference* profile). Values use setdefault, so whichever profile runs first
# wins — we want the training MIOpen find mode + TunableOp tuning for long runs.
# Loaded standalone via importlib so importing it does not drag in torch.
def _apply_training_rocm_profile():
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _re = _Path(__file__).resolve().parent / "stable_audio_tools" / "rocm_env.py"
    _spec = _ilu.spec_from_file_location("_sat_rocm_env_training", _re)
    _m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    _m.apply_profile("training", verbose=True)
_apply_training_rocm_profile()

import stable_audio_tools.rocm_env  # no-op now (training profile already set); kept for clarity
import torch
import json
import os
import pytorch_lightning as pl

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config

# Monkey-patch: matplotlib's spectrogram rendering segfaults on ROCm/headless
# Replace with a no-op that returns a blank PIL image
from PIL import Image as _PILImage
import stable_audio_tools.interface.aeiou as _aeiou
import stable_audio_tools.training.diffusion as _diffusion_training
def _noop_spectrogram(*args, **kwargs):
    return _PILImage.new('RGBA', (1, 1), (0, 0, 0, 0))
_aeiou.audio_spectrogram_image = _noop_spectrogram
_diffusion_training.audio_spectrogram_image = _noop_spectrogram

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    val_dl = None
    val_dataset_config = None

    if args.val_dataset_config:
        with open(args.val_dataset_config) as f:
            val_dataset_config = json.load(f)

        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            shuffle=False
        )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        pretrained_sd = load_ckpt_state_dict(args.pretrained_ckpt_path)
        copy_state_dict(model, pretrained_sd)

        # AdaLN-zero stability fix for ADDED global conditioners (partial zero-init).
        # When we append NumberConditioners, the global-conditioning input projection
        # (to_global_embed.0) grows in width, so copy_state_dict skips it and it stays
        # RANDOMLY initialised. Two failure modes follow:
        #   1) A fully-random projection injects an unmodulated signal into every AdaLN
        #      block of the pretrained DiT and diverges it to NaN within ~2k steps.
        #   2) Zeroing the WHOLE projection (naive AdaLN-zero) also kills the base
        #      model's pretrained seconds_total global signal, which is always-on
        #      (no CFG dropout) precisely so the model doesn't "cheese the loss" by
        #      generating silence — so that collapses to silent outputs instead.
        # Correct fix: copy the pretrained block back (seconds_total is global_cond_ids[0],
        # occupying the first ckpt-width input columns) and zero ONLY the appended
        # new-conditioner columns. The model then starts IDENTICAL to the base — full
        # seconds_total conditioning intact, new conditioners contributing nothing —
        # and the new columns grow in as training learns them.
        fixed = []
        for name, param in model.named_parameters():
            if name.endswith("to_global_embed.0.weight"):
                ckpt_t = pretrained_sd.get(name)
                if ckpt_t is None or tuple(ckpt_t.shape) != tuple(param.shape):
                    with torch.no_grad():
                        param.zero_()
                        preserved = 0
                        if (ckpt_t is not None and ckpt_t.shape[0] == param.shape[0]
                                and ckpt_t.shape[1] <= param.shape[1]):
                            param[:, :ckpt_t.shape[1]] = ckpt_t.to(param.dtype, copy=True)
                            preserved = ckpt_t.shape[1]
                    fixed.append((name, tuple(param.shape), preserved))
        for name, shape, preserved in fixed:
            print(f"[stability] partial zero-init {name} {shape}: preserved first {preserved} "
                  f"cols (pretrained seconds_total global cond), zeroed the rest (new conditioners)")

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    # Optional torch.compile (Inductor) on the DiT network. Opt-in via --compile;
    # 1st iter pays a graph-build cost, the rest run faster. Inductor FX-graph cache
    # is persisted across runs by rocm_env.yaml (TORCHINDUCTOR_*). We compile the
    # innermost transformer so EMA/demo/ARC machinery around it stays untouched.
    #
    # Use the in-place nn.Module.compile() (NOT `m = torch.compile(m)`): the latter
    # wraps the module in an OptimizedModule and injects `_orig_mod.` into every
    # state_dict key, which would break unwrap_model.py and the Stage-2 ckpt load.
    # In-place compile leaves the module tree — and therefore checkpoint keys — clean.
    if getattr(args, "compile", False):
        mode = getattr(args, "compile_mode", "default") or "default"
        target = training_wrapper.diffusion.model
        inner = getattr(target, "model", None)
        compile_target = inner if inner is not None else target
        compile_target.compile(mode=mode)
        what = "DiT inner network" if inner is not None else "DiT wrapper"
        print(f"torch.compile (in-place): {what} (mode={mode}); 1st iter will be slow.")

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        logger = pl.loggers.WandbLogger(project=args.name)
        logger.watch(training_wrapper)

        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.id, "checkpoints")
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project_name=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints")
        else:
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    val_args = {}

    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
        **val_args
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
