import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataset import collate_fn as interx_collate_fn


class DiffGeneratedDataset(Dataset):
    """Dataset wrapper that materializes model generations for metric evaluation."""
    def __init__(
        self,
        model,
        dataset,
        w_vectorizer,
        mm_num_samples,
        mm_num_repeats,
        first_frame=False,
        num_motions=None,
        generation_batch_size=32,
    ):
        """Generate motions once and expose them through a dataset interface.

        Args:
            model: Loaded Interact2Ar model used for sampling.
            dataset: Ground-truth Inter-X dataset that provides captions and lengths.
            w_vectorizer: GloVe/POS vectorizer used by evaluator text encoders.
            mm_num_samples: Number of batches selected for multimodality repeats.
            mm_num_repeats: Number of generations per selected multimodality batch.
            first_frame: Whether to seed generation with a ground-truth first frame.
            num_motions: Optional cap used by quick evaluation smoke tests.
            generation_batch_size: Batch size used while sampling generated motions.
        """
        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=generation_batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=interx_collate_fn,
        )

        model = model.eval()
        self.max_motion_length = dataset.max_motion_length
        real_num_batches = len(dataloader)

        generated_motion = []
        mm_generated_motions = []

        if mm_num_samples > 0:
            mm_idxs = np.random.choice(
                real_num_batches,
                mm_num_samples // dataloader.batch_size + 1,
                replace=False,
            )
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                if num_motions is not None and len(generated_motion) >= num_motions:
                    break
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens, _, _ = (
                    data
                )

                tokens = [tokens[bs_i].split("_") for bs_i in range(len(tokens))]
                word_emb = word_emb.detach().to(model.device).float()
                pos_ohot = pos_ohot.detach().to(model.device).float()

                # Smoke tests may request a final partial batch; store only the
                # requested number of generated samples while keeping the model
                # call shape unchanged.
                keep_count = len(m_lens)
                if num_motions is not None:
                    keep_count = min(keep_count, num_motions - len(generated_motion))

                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    if first_frame:
                        idx = random.randint(0, dataset.opt.UNIT_LENGTH)
                        first_frame_motion = motions[:, idx, :]

                        pred_motions = model(data, first_frame=first_frame_motion)
                    else:
                        pred_motions = model(data)


                    if t == 0:
                        sub_dicts = [
                            {
                                "motion": pred_motions[bs_i, : m_lens[bs_i]]
                                .cpu()
                                .numpy(),
                                "length": m_lens[bs_i].item(),
                                "cap_len": cap_lens[bs_i].item(),
                                "caption": caption[bs_i],
                                "tokens": tokens[bs_i],
                            }
                            for bs_i in range(keep_count)
                        ]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [
                            {
                                "motion": pred_motions[bs_i, : m_lens[bs_i]]
                                .cpu()
                                .numpy(),
                                "length": m_lens[bs_i].item(),
                            }
                            for bs_i in range(keep_count)
                        ]
                if is_mm:
                    mm_generated_motions += [
                        {
                            "caption": caption[bs_i],
                            "tokens": tokens[bs_i],
                            "cap_len": cap_lens[bs_i].item(),
                            "mm_motions": mm_motions[bs_i::keep_count],
                        }
                        for bs_i in range(keep_count)
                    ]

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        """Return the number of generated motions.

        Returns:
            int: Number of generated samples stored in memory.
        """
        return len(self.generated_motion)

    def __getitem__(self, item):
        """Return one generated motion in the evaluator dataloader format.

        Args:
            item: Generated-sample index.

        Returns:
            tuple: Text features, caption metadata, padded motion, and motion length.
        """
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = (
            data["motion"],
            data["length"],
            data["caption"],
            data["tokens"],
        )
        sent_len = data["cap_len"]
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except KeyError:
                word_emb, pos_oh = self.w_vectorizer["unk/OTHER"]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            motion,
            motion,
        )
