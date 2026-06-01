import argparse
from pathlib import Path

import lightning as pl
import torch

from models.wrapper import LitInteract2ArModel

DEFAULT_PROMPTS = (
    "The first person raises his/her right hand and waves it happily from side to side towards the second person. Then, the second person enthusiastically waves back with his/her right hand.",
    "One person stands with his/her back facing the other person. The person behind extends both hands and pushes the other person\'s back, causing him/her to take a few steps forward.",
)


def parse_args():
    """Parse command-line arguments for text-conditioned inference.

    Returns:
        argparse.Namespace: User-provided checkpoint, prompt, sampling, and rendering options.
    """
    parser = argparse.ArgumentParser(description="Generate Interact2Ar motions from text.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/interact2ar_mixed_memory.ckpt"),
        help="Path to the Interact2Ar Lightning checkpoint.",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="prompts",
        help="Text prompt to generate. Repeat the flag for multiple prompts.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=150,
        help="Number of frames to generate for each prompt.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.5,
        help="Classifier-free guidance weight stored on the loaded model.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for checkpoint loading and sampling.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Seed PyTorch and Lightning for repeatable sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Seed used when --deterministic is passed.",
    )
    parser.add_argument(
        "--novis",
        action="store_true",
        help="Skip aitviewer rendering and only run generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory where generated tensors and videos are written.",
    )
    return parser.parse_args()


def configure_reproducibility(seed):
    """Configure deterministic sampling for reproducible inference runs.

    Args:
        seed (int): Seed passed to Lightning and PyTorch workers.
    """
    pl.seed_everything(seed, workers=True)
    # CuDNN benchmark mode can pick different kernels between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    """Resolve a requested runtime device.

    Args:
        device_name (str): One of ``auto``, ``cpu``, or ``cuda``.

    Returns:
        torch.device: Device used for model loading and sampling.
    """
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint, cfg_weight, device_name):
    """Load a released Interact2Ar checkpoint for inference.

    Args:
        checkpoint (Path): Lightning checkpoint path.
        cfg_weight (float): Classifier-free guidance weight to write into the model options.
        device_name (str): Device selector passed by the CLI.

    Returns:
        LitInteract2ArModel: Evaluation-mode model loaded on the requested device.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = resolve_device(device_name)
    model = LitInteract2ArModel.load_from_checkpoint(str(checkpoint), map_location=device)
    model = model.to(device)
    model.opt_diffusion["CFG_WEIGHT"] = cfg_weight
    model.model.opt.CFG_WEIGHT = cfg_weight
    model.eval()
    return model


def build_inference_batch(prompt, length):
    """Build the minimal batch tuple expected by the autoregressive model.

    Args:
        prompt (str): Text condition for one generated interaction.
        length (int): Number of frames requested from the sampler.

    Returns:
        tuple: Placeholder batch matching the training dataloader contract.
    """
    # The Interact2Ar generation path only reads caption, m_length, and optional text tensors.
    return (
        None,
        None,
        [prompt],
        None,
        None,
        [length],
        None,
        None,
        None,
    )


def save_prediction(prediction, output_dir, stem):
    """Save the generated tensor before optional rendering.

    Args:
        prediction (torch.Tensor): Generated motion with shape (T, D).
        output_dir (Path): Directory that receives the tensor file.
        stem (str): Base file name without suffix.

    Returns:
        Path: Path to the written ``.pt`` tensor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / f"{stem}.pt"
    torch.save(prediction.detach().cpu(), tensor_path)
    return tensor_path


def render_prediction(prediction, output_dir, stem):
    """Render one generated motion with aitviewer.

    Args:
        prediction (torch.Tensor): Generated motion with shape (T, D).
        output_dir (Path): Directory used for the rendered video.
        stem (str): Base output name.
    """
    # Import lazily so --novis works in environments without rendering support.
    from vis import visualize_motion

    video_stem = str(output_dir / stem)
    visualize_motion(prediction, motion_id=video_stem)


def main():
    """Run text-conditioned Interact2Ar inference from the command line."""
    args = parse_args()
    prompts = args.prompts or list(DEFAULT_PROMPTS)

    if args.deterministic:
        configure_reproducibility(args.seed)

    model = load_model(args.checkpoint, args.cfg, args.device)

    for index, prompt in enumerate(prompts):
        # Use a stable zero-padded name so repeated runs are easy to compare.
        stem = f"sample_{index:03d}"
        batch = build_inference_batch(prompt, args.length)
        with torch.no_grad():
            prediction = model(batch)[0]

        tensor_path = save_prediction(prediction, args.output_dir, stem)
        print(f"Saved tensor: {tensor_path}")

        if not args.novis:
            render_prediction(prediction, args.output_dir, stem)


if __name__ == "__main__":
    main()
