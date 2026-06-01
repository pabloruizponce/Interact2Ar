import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.swa_utils import get_ema_avg_fn

from data.dataloader import get_dataset_motion_loader
from evaluate import evaluate
from models.wrapper import LitInteract2ArModel
from utils.ema import WeightAveraging
from utils.options import get_options

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"


class EMAWeightAveraging(WeightAveraging):
    """Enable EMA weight averaging after the warm-up phase.

    The checkpoint selected for the paper was trained with EMA. Keeping the
    callback in the public training script makes new runs follow the same
    optimization recipe without requiring any scheduler-specific code in the
    model wrapper.
    """

    def __init__(self):
        """Initialize the EMA callback with PyTorch's standard EMA averaging."""
        super().__init__(avg_fn=get_ema_avg_fn())

    def should_update(self, step_idx=None, epoch_idx=None):
        """Return whether the EMA weights should be updated at this step.

        Args:
            step_idx: Global optimization step provided by Lightning.
            epoch_idx: Epoch index provided by Lightning; unused here.

        Returns:
            bool: ``True`` once the first 100 optimizer steps are complete.
        """
        return (step_idx is not None) and (step_idx >= 100)


def parse_gpu_ids(gpu_arg):
    """Parse a single GPU id or a bracketed list from the command line.

    Args:
        gpu_arg: Value passed through ``--gpu`` such as ``"0"`` or ``"[0,1]"``.

    Returns:
        list[int]: GPU ids passed to the Lightning trainer.
    """
    gpu_arg = gpu_arg.strip()
    if gpu_arg.startswith("[") and gpu_arg.endswith("]"):
        return [int(x.strip()) for x in gpu_arg[1:-1].split(",") if x.strip()]
    return [int(gpu_arg)]


def build_run_name(user_name):
    """Build a stable checkpoint/logging name for this training run.

    Args:
        user_name: Optional user-provided run name.

    Returns:
        str: Run name used by checkpoints and TensorBoard logs.
    """
    if user_name:
        return user_name
    # Timestamped names avoid accidental checkpoint overwrites during sweeps.
    return "interact2ar_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def propagate_autoregressive_options(opt_dataset, opt_diffusion):
    """Copy autoregressive memory settings into the dataset options.

    Args:
        opt_dataset: Dataset options loaded from ``options/dataset``.
        opt_diffusion: Diffusion options loaded from ``options/diffusion``.

    Returns:
        object: The same dataset namespace, updated in place for dataloader use.
    """
    opt_dataset.PRED_MOTION_LENGTH = opt_diffusion.PRED_MOTION_LENGTH
    opt_dataset.PREFIX_MOTION_LENGTH = opt_diffusion.PREFIX_MOTION_LENGTH

    # Mixed Memory uses a short prefix plus a downsampled long-term prefix.
    if hasattr(opt_diffusion, "MAX_LONG_TERM"):
        opt_dataset.MAX_LONG_TERM = opt_diffusion.MAX_LONG_TERM
        opt_dataset.DONWSAMPLING_LONG_TERM = opt_diffusion.DONWSAMPLING_LONG_TERM

    opt_dataset.FIXED_DIVISIONS = getattr(opt_diffusion, "FIXED_DIVISIONS", False)
    return opt_dataset


def parse_limit_train_batches(value):
    """Parse Lightning's ``limit_train_batches`` override.

    Args:
        value: ``None``, an integer string such as ``"2"``, or a fractional
            string such as ``"0.1"``.

    Returns:
        int | float | None: Value passed to the Lightning trainer.
    """
    if value is None:
        return None
    if "." in value:
        return float(value)
    return int(value)


def parse_args():
    """Parse command-line arguments for public Interact2Ar training.

    Returns:
        argparse.Namespace: Training, logging, checkpoint, and hardware options.
    """
    parser = argparse.ArgumentParser(description="Train the Interact2Ar paper model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="paper.yaml",
        help="Dataset config under options/dataset/.",
    )
    parser.add_argument(
        "--diffusion",
        type=str,
        default="ar/paper.yaml",
        help="Diffusion config under options/diffusion/.",
    )
    parser.add_argument(
        "--denoiser",
        type=str,
        default="paper.yaml",
        help="Denoiser config under options/denoiser/.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="paper.yaml",
        help="Training config under options/train/.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="[0]",
        help="GPU id(s), for example 0 or [0,1,2,3].",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Training accelerator. Use cpu for smoke tests on unsupported GPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Seed for Lightning, PyTorch workers, and dataloader workers.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional local run name used for checkpoints and TensorBoard logs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where training checkpoints are written.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where TensorBoard logs are written.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional Lightning checkpoint to resume from.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers for the training split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional training batch-size override for smoke tests or short runs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional epoch override for smoke tests or short runs.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=str,
        default=None,
        help="Optional Lightning limit_train_batches override, e.g. 2 or 0.1.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable checkpointing/logging and run with the same model code.",
    )
    parser.add_argument(
        "--skip-eval-after-train",
        action="store_true",
        help="Skip the simple test-set evaluation after training finishes.",
    )
    return parser.parse_args()


def main():
    """Train the main-paper Interact2Ar model from the Python entry point."""
    args = parse_args()
    run_name = build_run_name(args.run_name)
    gpu_ids = parse_gpu_ids(args.gpu)

    # These switches make repeated runs as close as PyTorch allows on a machine.
    L.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt_dataset = get_options(os.path.join("options/dataset", args.dataset))
    opt_diffusion = get_options(os.path.join("options/diffusion", args.diffusion))
    opt_denoiser = get_options(os.path.join("options/denoiser", args.denoiser))
    opt_train = get_options(os.path.join("options/train", args.train))

    if not hasattr(opt_diffusion, "TEXT_EMBEDDER"):
        opt_diffusion.TEXT_EMBEDDER = "clip"
    if args.batch_size is not None:
        # Keep the paper config untouched unless the CLI explicitly overrides it.
        opt_train.BATCH_SIZE = args.batch_size
    if args.max_epochs is not None:
        # Scheduler milestones should follow the requested short run.
        opt_train.EPOCHS = args.max_epochs

    opt_dataset = propagate_autoregressive_options(opt_dataset, opt_diffusion)

    dataloader, _ = get_dataset_motion_loader(
        opt_dataset,
        "train",
        opt_train.BATCH_SIZE,
        "cuda" if torch.cuda.is_available() else "cpu",
        normalize=opt_diffusion.NORMALIZE,
        autoregressive=True,
        num_workers=args.num_workers,
    )

    lit_model = LitInteract2ArModel(
        vars(opt_diffusion),
        vars(opt_denoiser),
        vars(opt_train),
        autoregressive=True,
    )

    callbacks = [ModelSummary(max_depth=3)]
    logger = False
    if not args.debug:
        checkpoint_path = args.checkpoint_dir / run_name
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename="epoch-{epoch:04d}",
                every_n_epochs=500,
                save_top_k=-1,
                save_last=True,
            ),
        )
        logger = TensorBoardLogger(save_dir=args.log_dir, name=run_name)

    if opt_train.EMA is True:
        callbacks.append(EMAWeightAveraging())

    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU training was requested but CUDA is not available.")
    use_gpu = args.accelerator == "gpu" or (
        args.accelerator == "auto" and torch.cuda.is_available()
    )
    max_epochs = opt_train.EPOCHS
    limit_train_batches = parse_limit_train_batches(args.limit_train_batches)

    trainer_kwargs = {
        "accelerator": "gpu" if use_gpu else "cpu",
        "strategy": DDPStrategy(find_unused_parameters=True)
        if use_gpu and len(gpu_ids) > 1
        else "auto",
        "max_epochs": max_epochs,
        "enable_progress_bar": True,
        "logger": logger,
        "devices": gpu_ids if use_gpu else 1,
        "gradient_clip_val": 0.5,
        "callbacks": callbacks,
        "deterministic": "warn",
        "enable_checkpointing": not args.debug,
    }
    if limit_train_batches is not None:
        # Keep full-paper training untouched unless the user requests a short run.
        trainer_kwargs["limit_train_batches"] = limit_train_batches

    trainer = L.Trainer(**trainer_kwargs)

    resume_path = str(args.resume_checkpoint) if args.resume_checkpoint else None
    trainer.fit(model=lit_model, train_dataloaders=dataloader, ckpt_path=resume_path)

    if not args.debug and not args.skip_eval_after_train and trainer.is_global_zero:
        lit_model.eval()
        opt_dataset_eval = get_options(os.path.join("options/dataset/paper.yaml"))
        metrics = evaluate(run_name, lit_model, opt_dataset_eval, simple=True, first_frame=True)
        trainer.logger.log_metrics(metrics, step=trainer.global_step)


if __name__ == "__main__":
    main()
