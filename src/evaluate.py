# ruff: noqa: E402
import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime


# Parse GPU argument early, BEFORE importing torch, to set CUDA_VISIBLE_DEVICES
def _parse_gpu_arg():
    """Parse --gpu argument before torch import to set CUDA_VISIBLE_DEVICES."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_str = sys.argv[i + 1].strip()
            if gpu_str.startswith("[") and gpu_str.endswith("]"):
                gpu_ids = [x.strip() for x in gpu_str[1:-1].split(",") if x.strip()]
            else:
                gpu_ids = [gpu_str]
            return ",".join(gpu_ids)
    return None

if __name__ == "__main__":
    _gpu_env = _parse_gpu_arg()
    if _gpu_env is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_env

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dataloader import get_dataset_motion_loader
from data.dataset import collate_fn as interx_collate_fn
from evaluation.evaluator_wrapper import EvaluatorModelWrapper
from evaluation.metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_top_k,
    euclidean_distance_matrix,
)
from evaluation.model_motion_loaders import get_motion_loader
from models.wrapper import LitInteract2ArModel
from utils.options import get_options

torch.multiprocessing.set_sharing_strategy("file_system")

# Add these two lines for full GPU determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    """Resolve a requested evaluation device.

    Args:
        device_name: One of ``auto``, ``cpu``, or ``cuda``.

    Returns:
        torch.device: Device used for model and evaluator execution.
    """
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_matching_score(motion_loaders, eval_wrapper):
    """Compute matching score, R-precision, and motion embeddings.

    Args:
        motion_loaders: Mapping from model names to evaluator dataloaders.
        eval_wrapper: Loaded text-motion evaluator network.

    Returns:
        tuple: Matching scores, R-precision values, and motion embeddings by model.
    """
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print("========== Evaluating Matching Score ==========")
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                (
                    word_embeddings,
                    pos_one_hots,
                    _,
                    sent_lens,
                    motions,
                    m_lens,
                    _,
                    _,
                    _,
                ) = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens,
                )
                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
                )
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}")
        print(
            f"---> [{motion_loader_name}] Matching Score: {matching_score:.4f}",
            flush=True,
        )

        line = f"---> [{motion_loader_name}] R_precision: "
        for i in range(len(R_precision)):
            line += "(top %d): %.4f " % (i + 1, R_precision[i])
        print(line)
        print(line, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, eval_wrapper):
    """Compute FID between generated and ground-truth motion embeddings.

    Args:
        groundtruth_loader: Dataloader for real Inter-X motions.
        activation_dict: Generated motion embeddings keyed by model name.
        eval_wrapper: Loaded evaluator network used to embed motions.

    Returns:
        OrderedDict: FID score for each evaluated model.
    """
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print("========== Evaluating FID ==========")
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            (
                _,
                _,
                _,
                sent_lens,
                motions,
                m_lens,
                _,
                _,
                _,
            ) = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions, m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f"---> [{model_name}] FID: {fid:.4f}")
        print(f"---> [{model_name}] FID: {fid:.4f}", flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, diversity_times):
    """Compute diversity over generated motion embeddings.

    Args:
        activation_dict: Generated motion embeddings keyed by model name.
        diversity_times: Number of random embedding pairs to sample.

    Returns:
        OrderedDict: Diversity score for each evaluated model.
    """
    eval_dict = OrderedDict({})
    print("========== Evaluating Diversity ==========")
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f"---> [{model_name}] Diversity: {diversity:.4f}")
        print(f"---> [{model_name}] Diversity: {diversity:.4f}", flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, eval_wrapper, mm_num_times):
    """Compute multimodality from repeated generations per caption.

    Args:
        mm_motion_loaders: Dataloaders containing repeated generated motions.
        eval_wrapper: Loaded evaluator network used to embed motions.
        mm_num_times: Number of repeat pairs sampled for each caption.

    Returns:
        OrderedDict: Multimodality score for each evaluated model.
    """
    eval_dict = OrderedDict({})
    print("========== Evaluating MultiModality ==========")
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(
                    motions[0], m_lens[0]
                )
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f"---> [{model_name}] Multimodality: {multimodality:.4f}")
        print(
            f"---> [{model_name}] Multimodality: {multimodality:.4f}",
            flush=True,
        )
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    """Compute mean and 95% confidence interval across replications.

    Args:
        values: Metric values collected over evaluation replications.
        replication_times: Number of independent replications.

    Returns:
        tuple: Mean and confidence interval for the metric values.
    """
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def build_limited_ground_truth_loader(dataset, batch_size, num_motions):
    """Create a small reference loader for bounded evaluation smoke tests.

    Args:
        dataset: Full Inter-X test dataset returned by the public loader.
        batch_size: Batch size used by the evaluator networks.
        num_motions: Maximum number of reference motions to evaluate.

    Returns:
        DataLoader: Deterministic dataloader over the first ``num_motions`` items.
    """
    # Keep smoke tests deterministic and avoid worker startup overhead on CPU.
    subset_size = min(num_motions, len(dataset))
    subset = Subset(dataset, range(subset_size))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=interx_collate_fn,
    )


def evaluation(
    replication_times,
    eval_motion_loaders,
    gt_loader,
    eval_wrappers,
    mm_num_times,
    diversity_times,
    simple=False,
):
    """Run the full metric loop over replications and evaluator checkpoints.

    Args:
        replication_times: Number of evaluation replications.
        eval_motion_loaders: Mapping of model names to generated-loader factories.
        gt_loader: Ground-truth Inter-X dataloader.
        eval_wrappers: Evaluator checkpoints keyed by display name.
        mm_num_times: Number of multimodality pairs sampled per caption.
        diversity_times: Number of diversity pairs sampled across generated motions.
        simple: Whether to skip diversity and multimodality for quick validation.

    Returns:
        dict: Flattened summary metrics suitable for logging.
    """
    if not simple:
        all_metrics = OrderedDict(
            {
                "Matching Score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "FID": OrderedDict({}),
                "Diversity": OrderedDict({}),
                "MultiModality": OrderedDict({}),
            }
        )
    else:
        all_metrics = OrderedDict(
            {
                "Matching Score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "FID": OrderedDict({}),
            }
        )

    for replication in range(replication_times):
        motion_loaders = {}
        mm_motion_loaders = {}
        motion_loaders["ground truth"] = gt_loader
        for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
            motion_loader, mm_motion_loader = motion_loader_getter()
            motion_loaders[motion_loader_name] = motion_loader
            mm_motion_loaders[motion_loader_name] = mm_motion_loader

        print(f"==================== Replication {replication} ====================")

        # --- Start of the inner loop for iterating over eval_wrappers ---
        for wrapper_name, eval_wrapper in eval_wrappers.items():
            print(f"\n---> Evaluating Wrapper: {wrapper_name} <---\n")

            print(f"Time: {datetime.now()}")
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                motion_loaders, eval_wrapper
            )

            print(f"Time: {datetime.now()}")
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, eval_wrapper)

            if not simple:
                print(f"Time: {datetime.now()}")
                div_score_dict = evaluate_diversity(acti_dict, diversity_times)

                print(f"Time: {datetime.now()}")
                mm_score_dict = evaluate_multimodality(
                    mm_motion_loaders, eval_wrapper, mm_num_times
                )

            print("Evaluation pass completed.")

            # Store metrics with keys modified to include the wrapper name
            for key, item in mat_score_dict.items():
                new_key = f"{key} ({wrapper_name})"
                if new_key not in all_metrics["Matching Score"]:
                    all_metrics["Matching Score"][new_key] = [item]
                else:
                    all_metrics["Matching Score"][new_key] += [item]

            for key, item in R_precision_dict.items():
                new_key = f"{key} ({wrapper_name})"
                if new_key not in all_metrics["R_precision"]:
                    all_metrics["R_precision"][new_key] = [item]
                else:
                    all_metrics["R_precision"][new_key] += [item]

            for key, item in fid_score_dict.items():
                new_key = f"{key} ({wrapper_name})"
                if new_key not in all_metrics["FID"]:
                    all_metrics["FID"][new_key] = [item]
                else:
                    all_metrics["FID"][new_key] += [item]

            if not simple:
                for key, item in div_score_dict.items():
                    new_key = f"{key} ({wrapper_name})"
                    if new_key not in all_metrics["Diversity"]:
                        all_metrics["Diversity"][new_key] = [item]
                    else:
                        all_metrics["Diversity"][new_key] += [item]

                for key, item in mm_score_dict.items():
                    new_key = f"{key} ({wrapper_name})"
                    if new_key not in all_metrics["MultiModality"]:
                        all_metrics["MultiModality"][new_key] = [item]
                    else:
                        all_metrics["MultiModality"][new_key] += [item]
        # --- End of the inner loop ---

    final_metrics = {}

    for metric_name, metric_dict in all_metrics.items():
        print("========== %s Summary ==========" % metric_name)

        for model_name, values in metric_dict.items():
            mean, conf_interval = get_metric_statistics(
                np.array(values), replication_times
            )

            if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                final_metrics[metric_name] = mean
                if not simple:
                    final_metrics[f"{metric_name}_conf_interval"] = conf_interval
                print(
                    f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}"
                )
            elif isinstance(mean, np.ndarray):
                line = f"---> [{model_name}]"
                for i in range(len(mean)):
                    final_metrics[f"{metric_name}_T{i + 1}"] = mean[i]
                    if not simple:
                        final_metrics[f"{metric_name}_T{i + 1}_conf_interval"] = (
                            conf_interval[i]
                        )
                    line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                        i + 1,
                        mean[i],
                        conf_interval[i],
                    )
                print(line)

    return final_metrics


def evaluate(
    model_name,
    model,
    opt_dataset,
    simple=False,
    first_frame=False,
    device=None,
    num_motions=None,
    batch_size_override=None,
):
    """Evaluate one Interact2Ar model with the main-paper joint evaluators.

    Args:
        model_name: Name used in printed metric tables.
        model: Loaded Interact2Ar Lightning module.
        opt_dataset: Dataset options namespace for Inter-X.
        simple: Whether to run one lightweight replication without diversity metrics.
        first_frame: Whether to seed generated motions with a ground-truth first frame.
        device: Optional torch device. ``None`` keeps automatic CUDA/CPU selection.
        num_motions: Optional cap on generated motions for smoke tests.
        batch_size_override: Optional evaluator/generator batch size override.

    Returns:
        dict: Summary metrics from the evaluator loop.
    """
    # Device is cuda:0 when CUDA_VISIBLE_DEVICES remaps the requested GPU list.
    device = device or resolve_device("auto")

    # Move model to the correct device
    model = model.to(device)

    if not simple:
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10

        diversity_times = 300
        replication_times = 5
        batch_size = 32
    else:
        mm_num_samples = 1
        mm_num_repeats = 1
        mm_num_times = 1

        diversity_times = 1
        replication_times = 1
        batch_size = 32

    if batch_size_override is not None:
        batch_size = batch_size_override

    gt_loader, gt_dataset = get_dataset_motion_loader(
        opt_dataset,
        "test",
        batch_size,
        device,
        normalize=model.opt_diffusion["NORMALIZE"],
        autoregressive=False,
    )
    if num_motions is not None:
        # Bound the reference side too; otherwise CPU smoke tests still run the
        # full metric pass over the complete test split.
        gt_loader = build_limited_ground_truth_loader(gt_dataset, batch_size, num_motions)

    # Define generated-motion loaders after the ground-truth dataset exists.
    eval_motion_loaders = {
        model_name: lambda: get_motion_loader(
            opt_dataset,
            batch_size,
            gt_dataset,
            mm_num_samples,
            mm_num_repeats,
            device,
            model=model,
            first_frame=first_frame,
            num_motions=num_motions,
            generation_batch_size=batch_size,
        )
    }

    # The main paper reports the joint evaluators without normalization.
    eval_wrappers = {
        "JointsFull": EvaluatorModelWrapper(
            opt_dataset,
            device,
            "checkpoints-evaluator/joints_full/model/finest.tar",
            representation="joints",
            part="full",
            normalized=False,
        ),
        "JointsBody": EvaluatorModelWrapper(
            opt_dataset,
            device,
            "checkpoints-evaluator/joints_body/model/finest.tar",
            representation="joints",
            part="body",
            normalized=False,
        ),
        "JointsHands": EvaluatorModelWrapper(
            opt_dataset,
            device,
            "checkpoints-evaluator/joints_hands/model/finest.tar",
            representation="joints",
            part="hands",
            normalized=False,
        ),
    }

    return evaluation(
        replication_times,
        eval_motion_loaders,
        gt_loader,
        eval_wrappers,
        mm_num_times,
        diversity_times,
        simple=simple,
    )


if __name__ == "__main__":
    # Get option lines from arguments for easier sweeps
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="paper.yaml",
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.5,
        help="Weight for the classifier-free guidance",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for checkpoint loading and evaluation.",
    )


    parser.add_argument(
        "--first_frame",
        action="store_true",
        help="Whether to use the first frame for autoregressive sampling",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Whether to use the full evaluation",
    )
    parser.add_argument(
        "--num-motions",
        type=int,
        default=None,
        help="Optional generated-motion cap for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluator/generator batch-size override for smoke tests.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to use deterministic sampling",
    )


    parser.add_argument(
        "--gpu",
        type=str,
        default="[0]",
        help="GPU id(s) to use. Can be a single int (e.g., 0) or a list (e.g., [0,1,2] or [0, 3, 5]).",
    )

    args = parser.parse_args()

    if args.deterministic:
        seed = 44
        pl.seed_everything(seed, workers=True)

    # Load option files for the different modules
    opt_dataset = get_options(os.path.join("options/dataset", args.dataset))

    # Device is cuda:0 when CUDA_VISIBLE_DEVICES remaps the requested GPU list.
    device = resolve_device(args.device)

    # Load the model onto the correct device
    model = LitInteract2ArModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        map_location=device,
    )
    model.opt_diffusion["CFG_WEIGHT"] = args.cfg
    model.model.opt.CFG_WEIGHT = args.cfg
    model = model.to(device)

    evaluate(
        "Interact2Ar",
        model,
        opt_dataset,
        simple=not args.full,
        first_frame=args.first_frame,
        device=device,
        num_motions=args.num_motions,
        batch_size_override=args.batch_size,
    )
