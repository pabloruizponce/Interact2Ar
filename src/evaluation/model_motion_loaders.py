from os.path import join as pjoin

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from evaluation.diff_model_dataset import DiffGeneratedDataset
from utils.word_vectorizer import WordVectorizer


def collate_fn(batch):
    """Sort evaluator batches by caption length before PyTorch collation.

    Args:
        batch: List of generated or ground-truth evaluator samples.

    Returns:
        tuple: Collated batch sorted from longest to shortest caption.
    """
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class MMGeneratedDataset(Dataset):
    """Expose repeated generations for the multimodality metric."""
    def __init__(self, opt, motion_dataset, w_vectorizer):
        """Store generated multimodality samples.

        Args:
            opt: Dataset options with the maximum motion length.
            motion_dataset: Generated dataset containing repeated samples.
            w_vectorizer: GloVe/POS vectorizer kept for evaluator compatibility.
        """
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        """Return the number of captions with repeated generations.

        Returns:
            int: Number of multimodality entries.
        """
        return len(self.dataset)

    def __getitem__(self, item):
        """Return repeated motions for one caption.

        Args:
            item: Multimodality sample index.

        Returns:
            tuple: Repeated padded motions and their sorted lengths.
        """
        data = self.dataset[item]
        mm_motions = data["mm_motions"]
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion["length"])
            motion = mm_motion["motion"]
            if len(motion) < self.opt.MAX_MOTION_LENGTH:
                motion = np.concatenate(
                    [
                        motion,
                        np.zeros(
                            (self.opt.MAX_MOTION_LENGTH - len(motion), motion.shape[1])
                        ),
                    ],
                    axis=0,
                )
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens


def get_motion_loader(
    opt,
    batch_size,
    ground_truth_dataset,
    mm_num_samples,
    mm_num_repeats,
    device,
    model,
    first_frame=False,
    num_motions=None,
    generation_batch_size=32,
    num_workers=4,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    verbose=True,
):
    """Create dataloaders for generated motions and multimodality repeats.

    Args:
        opt: Dataset options namespace.
        batch_size: Batch size used by the metric evaluators.
        ground_truth_dataset: Dataset supplying captions, lengths, and references.
        mm_num_samples: Number of samples selected for multimodality evaluation.
        mm_num_repeats: Number of generated repeats per selected caption.
        device: Runtime device used by the generator.
        model: Loaded Interact2Ar model used for sampling.
        first_frame: Whether to seed generated motions with a ground-truth first frame.
        num_motions: Optional cap for quick smoke tests.
        generation_batch_size: Batch size used while sampling generated motions.
        num_workers: Number of dataloader workers for generated samples.
        pin_memory: Whether dataloader workers pin host memory.
        persistent_workers: Whether workers persist between epochs.
        prefetch_factor: Number of batches each worker prefetches.
        verbose: Whether to print loader creation status.

    Returns:
        tuple: Generated-motion dataloader and multimodality dataloader.
    """
    w_vectorizer = WordVectorizer(pjoin(opt.DATA_ROOT, "glove"), "hhi_vab")
    dataset = DiffGeneratedDataset(
        model,
        ground_truth_dataset,
        w_vectorizer,
        mm_num_samples,
        mm_num_repeats,
        first_frame=first_frame,
        num_motions=num_motions,
        generation_batch_size=generation_batch_size,
    )

    mm_dataset = MMGeneratedDataset(opt, dataset, w_vectorizer)

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    motion_loader = DataLoader(dataset, **loader_kwargs)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    if verbose:
        print("Generated dataset loading completed.")

    return motion_loader, mm_motion_loader
