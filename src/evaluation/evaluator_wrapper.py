import numpy as np
import torch
import torch.nn as nn
from human_body_prior.body_model.body_model import BodyModel
from torch.nn.utils.rnn import pack_padded_sequence

from utils.dataset import MotionNormalizerTorch, extract_motion_data
from utils.word_vectorizer import POS_enumerator


def init_weight(m):
    """Initialize linear and convolution layers with Xavier weights.

    Args:
        m: PyTorch module visited by ``Module.apply``.
    """
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose1d)
    ):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TextEncoderBiGRUCo(nn.Module):
    """Bidirectional GRU text encoder used by the evaluator checkpoint."""
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        """Build the evaluator text encoder.

        Args:
            word_size: Dimension of GloVe word embeddings.
            pos_size: Dimension of POS one-hot tags.
            hidden_size: Hidden dimension of the bidirectional GRU.
            output_size: Dimension of the shared text-motion embedding space.
            device: Device used by the evaluator wrapper.
        """
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        """Encode padded word/POS sequences into text embeddings.

        Args:
            word_embs: Word embedding tensor with shape ``(B, L, D)``.
            pos_onehot: POS one-hot tensor with shape ``(B, L, P)``.
            cap_lens: Original caption lengths before padding.

        Returns:
            torch.Tensor: Text embeddings with shape ``(B, output_size)``.
        """
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    """Bidirectional GRU motion encoder used by the evaluator checkpoint."""
    def __init__(self, input_size, hidden_size, output_size, device):
        """Build the evaluator motion encoder.

        Args:
            input_size: Dimension of movement features produced by the CNN encoder.
            hidden_size: Hidden dimension of the bidirectional GRU.
            output_size: Dimension of the shared text-motion embedding space.
            device: Device used by the evaluator wrapper.
        """
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(
            torch.randn((2, 1, self.hidden_size), requires_grad=True)
        )

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        """Encode movement features into motion embeddings.

        Args:
            inputs: Movement feature tensor with shape ``(B, T, D)``.
            m_lens: Motion lengths after movement-encoder downsampling.

        Returns:
            torch.Tensor: Motion embeddings with shape ``(B, output_size)``.
        """
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MovementConvEncoder(nn.Module):
    """Temporal convolution encoder that downsamples motion features."""
    def __init__(self, input_size, hidden_size, output_size):
        """Build the temporal convolutional movement encoder.

        Args:
            input_size: Input pose or joint feature dimension.
            hidden_size: Hidden channel dimension.
            output_size: Output movement feature dimension.
        """
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        """Encode per-frame motion features into downsampled movement features.

        Args:
            inputs: Motion tensor with shape ``(B, T, D)``.

        Returns:
            torch.Tensor: Encoded movement tensor with shape ``(B, T / 4, D_out)``.
        """
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


def build_models(opt, ckpt_path):
    """Build evaluator networks and load a released evaluator checkpoint.

    Args:
        opt: Evaluator option namespace with architecture dimensions and device.
        ckpt_path: Path to ``finest.tar`` evaluator weights.

    Returns:
        tuple: Text encoder, motion encoder, and movement encoder in eval format.
    """
    movement_enc = MovementConvEncoder(
        opt.dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=opt.dim_word,
        pos_size=opt.dim_pos_ohot,
        hidden_size=opt.dim_text_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent,
        hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    # Released evaluator checkpoints are trusted project assets.
    checkpoint = torch.load(
        ckpt_path,
        map_location=opt.device,
        weights_only=False,
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(
        "Loading Evaluation Model Wrapper (Epoch %d) completed."
        % (checkpoint["epoch"])
    )
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):
    """Main-paper text-motion evaluator wrapper."""
    def __init__(
        self,
        opt,
        device,
        checkpoint="data/checkpoints/text_mot_match/model/finest.tar",
        part="full",
        representation="smpl",
        normalized=False,
    ):
        """Load one evaluator checkpoint and configure its motion representation.

        Args:
            opt: Dataset options namespace; evaluator dimensions are added in place.
            device: Device where evaluator networks and SMPL-X body model run.
            checkpoint: Path to evaluator weights.
            part: Body subset evaluated: ``full``, ``body``, or ``hands``.
            representation: Motion representation: ``smpl`` or ``joints``.
            normalized: Whether evaluator inputs should use stored normalization stats.
        """
        if representation == "smpl":
            if part == "full":
                dim_pose = 56 * 12
            elif part == "body":
                dim_pose = 23 * 12
            elif part == "hands":
                dim_pose = 32 * 12
            else:
                raise ValueError("Part must be either 'full', 'body' or 'hands'")
        elif representation == "joints":
            if part == "full":
                dim_pose = 55 * 6
            elif part == "body":
                dim_pose = 22 * 6
            elif part == "hands":
                dim_pose = 30 * 6
            else:
                raise ValueError("Part must be either 'full', 'body' or 'hands'")

        opt.dim_pose = dim_pose
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        opt.max_motion_length = 150
        opt.max_text_len = 35
        opt.dim_movement_enc_hidden = 512
        opt.dim_movement_latent = 512
        opt.device = device
        opt.unit_length = 4

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(
            opt, checkpoint
        )
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

        # Body Model
        num_betas = 10
        bm_path = "data/body_models/smplx/SMPLX_NEUTRAL.npz"
        self.bm = BodyModel(bm_fname=bm_path, num_betas=num_betas).to(device)

        self.part = part
        self.representation = representation
        self.normalized = normalized
        # Main-paper joint evaluators are stored without normalization.
        self.normalizer = (
            MotionNormalizerTorch(representation=representation) if normalized else None
        )

    def get_joints_from_motion(self, motion):
        """Convert generated SMPL-X motion features into world-space joints.

        Args:
            motion: Flattened 6D SMPL-X motion tensor with shape ``(B, T, D)``.

        Returns:
            torch.Tensor: Paired-person joints with shape ``(B, T, 55, 6)``.
        """
        smpl_motion1, smpl_motion2 = extract_motion_data(motion)
        B, T, _ = smpl_motion1["pose_body"].shape

        trans_1 = smpl_motion1["trans"]
        trans_2 = smpl_motion2["trans"]

        joints_1 = self.bm(
            pose_body=smpl_motion1["pose_body"].reshape(B * T, -1),
            pose_hand=torch.cat(
                [smpl_motion1["pose_lhand"], smpl_motion1["pose_rhand"]], dim=-1
            ).reshape(B * T, -1),
            root_orient=smpl_motion1["root_orient"].reshape(B * T, -1),
        ).Jtr.reshape(B, T, 55, 3)
        joints_1 = joints_1 + trans_1.unsqueeze(2)

        joints_2 = self.bm(
            pose_body=smpl_motion2["pose_body"].reshape(B * T, -1),
            pose_hand=torch.cat(
                [smpl_motion2["pose_lhand"], smpl_motion2["pose_rhand"]], dim=-1
            ).reshape(B * T, -1),
            root_orient=smpl_motion2["root_orient"].reshape(B * T, -1),
        ).Jtr.reshape(B, T, 55, 3)
        joints_2 = joints_2 + trans_2.unsqueeze(2)

        joints = torch.cat([joints_1, joints_2], dim=-1)
        return joints

    def correct_representation_motion(self, motion):
        """Select the evaluator representation and body subset.

        Args:
            motion: Flattened generated motion tensor with shape ``(B, T, D)``.

        Returns:
            torch.Tensor: Motion features matching the loaded evaluator checkpoint.
        """
        if self.representation == "smpl":
            if self.normalized:
                motion = self.normalizer.forward(motion)

            root_orient = motion[:, :, :12].reshape(
                motion.shape[0], motion.shape[1], 1, 12
            )
            poses = motion[:, :, 12:-12].reshape(
                motion.shape[0], motion.shape[1], 54, 12
            )
            trans = motion[:, :, -12:].reshape(motion.shape[0], motion.shape[1], 1, 12)

            if self.part == "full":
                return motion
            elif self.part == "body":
                poses_body = torch.cat(
                    [root_orient, poses[:, :, :21, :], trans], dim=2
                ).reshape(motion.shape[0], motion.shape[1], -1)
                return poses_body
            elif self.part == "hands":
                poses_hands = torch.cat(
                    [root_orient, poses[:, :, -30:, :], trans], dim=2
                ).reshape(motion.shape[0], motion.shape[1], -1)
                return poses_hands
            else:
                raise NotImplementedError

        elif self.representation == "joints":
            joints = self.get_joints_from_motion(motion)
            if self.normalized:
                joints = self.normalizer.forward(
                    joints.reshape(motion.shape[0], motion.shape[1], -1)
                ).reshape(joints.shape)

            if self.part == "full":
                return joints.reshape(motion.shape[0], motion.shape[1], -1)
            elif self.part == "body":
                body_joints = joints[:, :, :22, :].reshape(
                    motion.shape[0], motion.shape[1], -1
                )
                return body_joints
            elif self.part == "hands":
                hands_joints = joints[:, :, -30:, :].reshape(
                    motion.shape[0], motion.shape[1], -1
                )
                return hands_joints
            else:
                raise NotImplementedError

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        """Embed text and motion batches in the shared evaluator space.

        Args:
            word_embs: Word embeddings with shape ``(B, L, D)``.
            pos_ohot: POS one-hot features with shape ``(B, L, P)``.
            cap_lens: Caption lengths before padding.
            motions: Motion tensor with shape ``(B, T, D)``.
            m_lens: Motion lengths before temporal downsampling.

        Returns:
            tuple: Text embeddings and motion embeddings. Motion rows are length-sorted.
        """
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            motions = self.correct_representation_motion(motions)

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            # Encode motion after sorting by length for packed GRU input.
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            # Reorder text embeddings with the same sorted indices as motions.
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    def get_motion_embeddings(self, motions, m_lens):
        """Embed a motion batch in the evaluator motion space.

        Args:
            motions: Motion tensor with shape ``(B, T, D)``.
            m_lens: Motion lengths before temporal downsampling.

        Returns:
            torch.Tensor: Length-sorted motion embeddings.
        """
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            motions = self.correct_representation_motion(motions)

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            # Encode motion after sorting by length for packed GRU input.
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
