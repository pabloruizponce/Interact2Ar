import codecs as cs
import math
import random
from contextlib import ExitStack
from os.path import join as pjoin
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import utils.rotation_conversions as geometry
from utils.dataset import MotionNormalizerTorch
from utils.tensor import to_torch


def collate_fn(batch):
    """Collate Inter-X samples while preserving optional cache fields.

    Args:
        batch: List of dataset samples sorted by caption length.

    Returns:
        tuple: Batch fields collated with PyTorch defaults, with optional fields
        such as cached CLIP tensors or joint caches left as ``None`` when the
        whole batch does not provide them.
    """
    batch.sort(key=lambda x: x[3], reverse=True)
    collated = []
    for values in zip(*batch):
        if all(value is None for value in values):
            collated.append(None)
        elif any(value is None for value in values):
            raise ValueError("Optional dataset fields must be present for all items or none.")
        else:
            collated.append(default_collate(values))
    return tuple(collated)


def _read_split_ids(split_file):
    """Read one Inter-X split file.

    Args:
        split_file: Path to the split text file.

    Returns:
        list[str]: Motion ids in the split.
    """
    with cs.open(split_file, "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def _optional_dir(path, description):
    """Resolve an optional directory from the dataset config.

    Args:
        path: Directory path or ``None``.
        description: Human-readable cache name used in warnings.

    Returns:
        Path | None: Existing directory, or ``None`` when the cache is absent.
    """
    if path is None:
        return None
    resolved = Path(path)
    if resolved.exists():
        return resolved
    print(f"{description} not found at {resolved}; using the on-the-fly fallback.")
    return None


def _load_text_records(text_dir, name):
    """Load captions and token tags for one Inter-X motion.

    Args:
        text_dir: Directory containing ``<motion_id>.txt`` files.
        name: Motion id to read.

    Returns:
        list[dict]: Captions and token lists that describe the full motion.
    """
    text_data = []
    with cs.open(pjoin(text_dir, name + ".txt")) as file:
        for line in file.readlines():
            caption, token_text, f_tag, to_tag = line.strip().split("#")
            f_tag = 0.0 if np.isnan(float(f_tag)) else float(f_tag)
            to_tag = 0.0 if np.isnan(float(to_tag)) else float(to_tag)
            if f_tag != 0.0 or to_tag != 0.0:
                raise ValueError("Segment-level captions are not used by this release.")
            text_data.append({"caption": caption, "tokens": token_text.split(" ")})
    return text_data


def _load_text_tensor(text_tensor_dir, name):
    """Load cached CLIP sequence features when they are available.

    Args:
        text_tensor_dir: Optional directory containing ``<motion_id>.npy`` files.
        name: Motion id to load.

    Returns:
        torch.Tensor | None: Cached CLIP features, or ``None`` to compute CLIP
        features inside the model at runtime.
    """
    if text_tensor_dir is None:
        return None
    tensor_path = text_tensor_dir / f"{name}.npy"
    if not tensor_path.exists():
        return None
    return torch.tensor(np.load(tensor_path))


def _load_records(
    opt,
    split_file,
    motion_file,
    min_motion_len,
    num_samples_to_keep,
):
    """Load motion, optional joint caches, captions, and text caches.

    Args:
        opt: Dataset options namespace.
        split_file: Split file with motion ids.
        motion_file: HDF5 file containing SMPL-X motion parameters.
        min_motion_len: Minimum accepted motion length in frames.
        num_samples_to_keep: Optional truncation used for small debug runs.

    Returns:
        tuple: ``(data_dict, name_list, length_arr, keys)`` ready for a dataset.
    """
    id_list = _read_split_ids(split_file)
    text_tensor_dir = _optional_dir(
        getattr(opt, "TEXT_TENSOR_DIR", None), "Cached CLIP text tensor directory"
    )
    joints_file = Path(str(motion_file).replace("motions", "joints"))
    has_joint_cache = joints_file.exists()
    if not has_joint_cache:
        print(
            f"Joint cache not found at {joints_file}; target joints will be computed "
            "from SMPL-X when the training loss needs them."
        )

    data_dict = {}
    new_name_list = []
    length_list = []

    with ExitStack() as stack:
        motion_h5 = stack.enter_context(h5py.File(motion_file, "r"))
        joints_h5 = stack.enter_context(h5py.File(joints_file, "r")) if has_joint_cache else None
        keys = list(motion_h5.keys())

        for name in tqdm(id_list):
            try:
                motion = motion_h5[name][:].astype("float32")
                if len(motion) < min_motion_len or len(motion) >= 1000:
                    continue

                joints = joints_h5[name][:].astype("float32") if joints_h5 is not None else None
                text_data = _load_text_records(opt.TEXT_DIR, name)
                if len(text_data) == 0:
                    continue

                data_dict[name] = {
                    "motion": motion,
                    "joints": joints,
                    "length": len(motion),
                    "text": text_data,
                    "text_tensor": _load_text_tensor(text_tensor_dir, name),
                }
                new_name_list.append(name)
                length_list.append(len(motion))
            except (KeyError, FileNotFoundError, ValueError) as exc:
                print(f"Skipping {name}: {exc}")

    if len(new_name_list) == 0:
        raise ValueError(f"No usable motions were loaded from {motion_file}.")

    name_list, length_list = zip(
        *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
    )
    name_list = list(name_list)
    length_arr = np.array(length_list)

    # Keeping a tiny prefix of the split is useful for smoke tests and CI jobs.
    if 0 < num_samples_to_keep < len(name_list):
        name_list = name_list[:num_samples_to_keep]
        length_arr = length_arr[:num_samples_to_keep]
        data_dict = {name: data_dict[name] for name in name_list}

    return data_dict, name_list, length_arr, keys


def _choose_caption(text_list, text_tensor):
    """Choose one caption and matching cached CLIP tensor for a motion.

    Args:
        text_list: List of captions for the current motion.
        text_tensor: Optional cached CLIP tensor with one entry per caption.

    Returns:
        tuple: ``(caption, tokens, selected_text_tensor)``.
    """
    text_index = random.randint(0, len(text_list) - 1)
    selected_tensor = text_tensor[text_index] if text_tensor is not None else None
    text_data = text_list[text_index]
    return text_data["caption"], text_data["tokens"], selected_tensor


def _pad_or_trim_tokens(tokens, max_text_length):
    """Add special tokens and pad/truncate to the evaluator vocabulary length.

    Args:
        tokens: Raw ``word/POS`` tokens from the processed text file.
        max_text_length: Maximum number of text tokens before SOS/EOS are added.

    Returns:
        tuple: ``(tokens, sent_len)`` after padding or truncation.
    """
    if len(tokens) < max_text_length:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_length + 2 - sent_len)
    else:
        tokens = tokens[:max_text_length]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
    return tokens, sent_len


def _vectorize_tokens(tokens, w_vectorizer):
    """Convert processed tokens into GloVe and POS arrays.

    Args:
        tokens: Padded token list.
        w_vectorizer: Word vectorizer loaded from the Inter-X GloVe cache.

    Returns:
        tuple[np.ndarray, np.ndarray]: Word embeddings and POS one-hot vectors.
    """
    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        try:
            word_emb, pos_oh = w_vectorizer[token]
        except KeyError:
            word_emb, pos_oh = w_vectorizer["unk/OTHER"]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    return np.concatenate(word_embeddings, axis=0), np.concatenate(pos_one_hots, axis=0)


def _sample_unit_length(m_length, unit_length):
    """Choose the augmented crop length used by the original T2M loader.

    Args:
        m_length: Full motion length in frames.
        unit_length: Unit used to align crops to the evaluator stride.

    Returns:
        int: Crop length rounded down to a multiple of ``unit_length``.
    """
    coin = np.random.choice(["single", "single", "double"]) if unit_length < 10 else "single"
    if coin == "double":
        return (m_length // unit_length - 1) * unit_length
    return (m_length // unit_length) * unit_length


def _slice_optional(array, start, end):
    """Slice an optional array without special cases at the call site.

    Args:
        array: ``None`` or a NumPy array.
        start: Start index for slicing.
        end: End index for slicing.

    Returns:
        np.ndarray | None: Sliced array or ``None``.
    """
    return None if array is None else array[start:end]


def _concat_optional(parts, axis=0):
    """Concatenate optional NumPy arrays.

    Args:
        parts: Sequence of arrays or ``None`` placeholders.
        axis: Axis passed to ``np.concatenate``.

    Returns:
        np.ndarray | None: Concatenated array, or ``None`` if any input is absent.
    """
    if any(part is None for part in parts):
        return None
    return np.concatenate(parts, axis=axis)


def _axis_angle_motion_to_rot6d(data, num_person):
    """Convert paired SMPL-X axis-angle motion to the model's 6D rotation layout.

    Args:
        data: Motion array with pose joints and final translation row.
        num_person: Number of people represented in the final channel dimension.

    Returns:
        torch.Tensor: Motion tensor flattened to ``(T, D)``.
    """
    translations = to_torch(np.expand_dims(data[:, -1, :], axis=1))[:, 0, :]
    pose = to_torch(data[:, :-1, :])

    pose_all = []
    for person_idx in range(num_person):
        start = 3 * person_idx
        end = start + 3
        pose_all.append(
            geometry.matrix_to_rotation_6d(
                geometry.axis_angle_to_matrix(pose[:, :, start:end])
            )
        )
    rotations = torch.cat(pose_all, dim=2)

    padded_trans = torch.zeros((rotations.shape[0], rotations.shape[2]), dtype=rotations.dtype)
    for person_idx in range(num_person):
        start = 6 * person_idx
        padded_trans[:, start : start + 3] = translations[:, 3 * person_idx : 3 * person_idx + 3]

    return torch.cat((rotations, padded_trans[:, None]), 1).reshape(data.shape[0], -1)


class ARText2MotionDatasetV2HHI(data.Dataset):
    """Inter-X text-to-interaction dataset for autoregressive training."""

    def __init__(
        self,
        opt,
        split_file,
        w_vectorizer,
        motion_file,
        num_samples_to_keep=-1,
        normalize=True,
    ):
        """Load an autoregressive Inter-X split.

        Args:
            opt: Dataset and autoregressive memory options.
            split_file: File containing motion ids for the requested split.
            w_vectorizer: GloVe/POS vectorizer used by the evaluator protocol.
            motion_file: HDF5 file containing SMPL-X motion parameters.
            num_samples_to_keep: Optional split truncation for smoke tests.
            normalize: Whether to apply the motion normalizer to SMPL-X features.
        """
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.PRED_MOTION_LENGTH
        self.prefix_length = opt.PREFIX_MOTION_LENGTH
        self.num_person = 2
        self.normalize = normalize
        self.normalizer = MotionNormalizerTorch() if normalize else None

        self.data_dict, self.name_list, self.length_arr, self.keys = _load_records(
            opt,
            split_file,
            motion_file,
            min_motion_len=45,
            num_samples_to_keep=num_samples_to_keep,
        )
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        """Move the dataset pointer to the first sample at least ``length`` frames.

        Args:
            length: Minimum motion length to expose from ``__len__`` and
                ``__getitem__``.
        """
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        """Return data unchanged because the public path stores raw motions.

        Args:
            data: Motion tensor to return.

        Returns:
            Same object passed in ``data``.
        """
        return data

    def __len__(self):
        """Return the number of samples visible after the length pointer.

        Returns:
            int: Number of usable motions.
        """
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        """Return one autoregressive training sample.

        Args:
            item: Dataset index after applying the internal length pointer.

        Returns:
            tuple: Text features, motion crop, optional CLIP cache, optional joint
            cache, and the padding index used by autoregressive masks.
        """
        idx = self.pointer + item
        data_item = self.data_dict[self.name_list[idx]]
        motion = data_item["motion"]
        joints = data_item["joints"]
        m_length = data_item["length"]
        caption, tokens, text_tensor = _choose_caption(data_item["text"], data_item["text_tensor"])
        tokens, sent_len = _pad_or_trim_tokens(tokens, self.opt.MAX_TEXT_LENGHT)
        word_embeddings, pos_one_hots = _vectorize_tokens(tokens, self.w_vectorizer)

        m_length = _sample_unit_length(m_length, self.opt.UNIT_LENGTH)
        start = random.randint(0, len(motion) - m_length)
        motion = motion[start : start + m_length]
        joints = _slice_optional(joints, start, start + m_length)

        if self.opt.FIXED_DIVISIONS:
            n_intervals = math.ceil(m_length / self.max_motion_length)
            idx = random.randint(0, n_intervals - 1) * self.max_motion_length
        else:
            idx = random.randint(0, m_length - self.max_motion_length)

        has_long_term_memory = hasattr(self.opt, "MAX_LONG_TERM") and self.opt.MAX_LONG_TERM > 0
        whole_memory_length = (
            self.opt.MAX_LONG_TERM * self.opt.DONWSAMPLING_LONG_TERM + self.prefix_length
            if has_long_term_memory
            else self.prefix_length
        )

        if whole_memory_length > idx:
            pad = whole_memory_length - idx
            motion = np.concatenate(
                [np.zeros((pad, motion.shape[1], motion.shape[2])), motion[: idx + self.max_motion_length]],
                axis=0,
            )
            joints = _concat_optional(
                [
                    None if joints is None else np.zeros((pad, joints.shape[1], joints.shape[2])),
                    None if joints is None else joints[: idx + self.max_motion_length],
                ]
            )
            init_idx = pad
        else:
            start_frame = idx - whole_memory_length
            motion = motion[start_frame : idx + self.max_motion_length]
            joints = _slice_optional(joints, start_frame, idx + self.max_motion_length)
            init_idx = 0

        if idx + self.max_motion_length > m_length:
            pad = self.max_motion_length - (m_length - idx)
            motion = np.concatenate(
                [motion, np.zeros((pad, motion.shape[1], motion.shape[2]))], axis=0
            )
            joints = _concat_optional(
                [
                    joints,
                    None if joints is None else np.zeros((pad, joints.shape[1], joints.shape[2])),
                ]
            )
            m_length = m_length - idx
        else:
            m_length = self.max_motion_length

        if has_long_term_memory:
            motion = np.concatenate(
                [
                    motion[:: self.opt.DONWSAMPLING_LONG_TERM][: self.opt.MAX_LONG_TERM],
                    motion[-(self.prefix_length + self.max_motion_length) :],
                ],
                axis=0,
            )
            if joints is not None:
                joints = np.concatenate(
                    [
                        joints[:: self.opt.DONWSAMPLING_LONG_TERM][: self.opt.MAX_LONG_TERM],
                        joints[-(self.prefix_length + self.max_motion_length) :],
                    ],
                    axis=0,
                )

            if init_idx > 0:
                if init_idx > (whole_memory_length - self.prefix_length):
                    init_idx = self.opt.MAX_LONG_TERM + init_idx - (whole_memory_length - self.prefix_length)
                else:
                    init_idx = math.ceil(init_idx / self.opt.DONWSAMPLING_LONG_TERM)

        motion = self.to_rot_6d(motion).float()
        if self.normalize:
            motion = self.normalizer.forward(motion)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            text_tensor,
            joints,
            init_idx,
        )

    def to_rot_6d(self, data):
        """Convert a raw paired SMPL-X motion array to flattened 6D rotations.

        Args:
            data: Axis-angle motion array with a final translation row.

        Returns:
            torch.Tensor: Flattened 6D rotation representation expected by the model.
        """
        return _axis_angle_motion_to_rot6d(data, self.num_person)


class Text2MotionDatasetV2HHI(data.Dataset):
    """Inter-X text-to-interaction dataset for evaluation and simple loading."""

    def __init__(
        self,
        opt,
        split_file,
        w_vectorizer,
        motion_file,
        num_samples_to_keep=-1,
        normalize=True,
    ):
        """Load a non-autoregressive Inter-X split.

        Args:
            opt: Dataset options.
            split_file: File containing motion ids for the requested split.
            w_vectorizer: GloVe/POS vectorizer used by the evaluator protocol.
            motion_file: HDF5 file containing SMPL-X motion parameters.
            num_samples_to_keep: Optional split truncation for smoke tests.
            normalize: Whether to apply the motion normalizer to SMPL-X features.
        """
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.MAX_MOTION_LENGTH
        self.num_person = 2
        self.normalize = normalize
        self.normalizer = MotionNormalizerTorch() if normalize else None

        self.data_dict, self.name_list, self.length_arr, self.keys = _load_records(
            opt,
            split_file,
            motion_file,
            min_motion_len=30,
            num_samples_to_keep=num_samples_to_keep,
        )
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        """Move the dataset pointer to the first sample at least ``length`` frames.

        Args:
            length: Minimum motion length to expose from ``__len__`` and
                ``__getitem__``.
        """
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        """Return data unchanged because the public path stores raw motions.

        Args:
            data: Motion tensor to return.

        Returns:
            Same object passed in ``data``.
        """
        return data

    def __len__(self):
        """Return the number of samples visible after the length pointer.

        Returns:
            int: Number of usable motions.
        """
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        """Return one evaluation sample.

        Args:
            item: Dataset index after applying the internal length pointer.

        Returns:
            tuple: Text features, padded motion, optional CLIP cache, and optional
            joint cache.
        """
        idx = self.pointer + item
        data_item = self.data_dict[self.name_list[idx]]
        motion = data_item["motion"]
        joints = data_item["joints"]
        m_length = data_item["length"]
        caption, tokens, text_tensor = _choose_caption(data_item["text"], data_item["text_tensor"])
        tokens, sent_len = _pad_or_trim_tokens(tokens, self.opt.MAX_TEXT_LENGHT)
        word_embeddings, pos_one_hots = _vectorize_tokens(tokens, self.w_vectorizer)

        m_length = _sample_unit_length(m_length, self.opt.UNIT_LENGTH)
        start = random.randint(0, len(motion) - m_length)
        motion = motion[start : start + m_length]
        joints = _slice_optional(joints, start, start + m_length)

        if m_length < self.max_motion_length:
            pad = self.max_motion_length - m_length
            motion = np.concatenate(
                [motion, np.zeros((pad, motion.shape[1], motion.shape[2]))], axis=0
            )
            joints = _concat_optional(
                [
                    joints,
                    None if joints is None else np.zeros((pad, joints.shape[1], joints.shape[2])),
                ]
            )
        else:
            motion = motion[: self.max_motion_length]
            joints = _slice_optional(joints, 0, self.max_motion_length)
            m_length = self.max_motion_length

        motion = self.to_rot_6d(motion).float()
        if self.normalize:
            motion = self.normalizer.forward(motion)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            text_tensor,
            joints,
        )

    def to_rot_6d(self, data):
        """Convert a raw paired SMPL-X motion array to flattened 6D rotations.

        Args:
            data: Axis-angle motion array with a final translation row.

        Returns:
            torch.Tensor: Flattened 6D rotation representation expected by the model.
        """
        return _axis_angle_motion_to_rot6d(data, self.num_person)
