from os.path import join as pjoin

from torch.utils.data import DataLoader

from data.dataset import ARText2MotionDatasetV2HHI, Text2MotionDatasetV2HHI, collate_fn
from utils.word_vectorizer import WordVectorizer


def get_dataset_motion_loader(
    opt,
    split,
    batch_size,
    device,
    normalize,
    autoregressive,
    num_workers=8,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=2,
    verbose=True,
    shuffle=True,
):
    """Build the Inter-X dataset and PyTorch dataloader for one split.

    Args:
        opt: Dataset options namespace loaded from ``options/dataset``.
        split: Split name such as ``train``, ``val``, or ``test``.
        batch_size: Number of motions per batch.
        device: Kept for call-site compatibility; tensors are moved by the model.
        normalize: Whether motion features should be normalized by the dataset.
        autoregressive: Whether to use the autoregressive crop/memory dataset.
        num_workers: Number of PyTorch dataloader workers.
        pin_memory: Whether dataloader workers pin host memory.
        persistent_workers: Whether workers persist between epochs.
        prefetch_factor: Number of batches each worker prefetches.
        verbose: Whether to print dataset loading messages.
        shuffle: Whether to shuffle the dataset.

    Returns:
        tuple: ``(dataloader, dataset)`` for the requested split.
    """

    # Resolve split and motion files from the normalized public data layout.
    split_file = pjoin(opt.DATA_ROOT, f"splits/{split}.txt")
    motion_file = pjoin(opt.MOTION_DIR, f"{split}.h5")

    # The evaluator protocol uses the Inter-X GloVe/POS vocabulary.
    w_vectorizer = WordVectorizer(pjoin(opt.DATA_ROOT, "glove"), "hhi_vab")

    # Use the AR dataset only for training-style memory crops.
    if verbose:
        print("Loading dataset %s ..." % opt.DATASET_NAME)
    dataset_cls = (
        ARText2MotionDatasetV2HHI if autoregressive else Text2MotionDatasetV2HHI
    )
    dataset = dataset_cls(
        opt, split_file, w_vectorizer, motion_file, normalize=normalize
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": True,
        "collate_fn": collate_fn,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader, dataset
