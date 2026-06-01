import torch
import torch.nn as nn

from models.layers import FinalLayer, TransformerBlock
from models.utils import PositionalEncoding, TimestepEmbedder, zero_module


def split_motion(x):
    """
    Splits the motion tensor into two parts: poses and trajectories.

    Args:
        x: Motion tensor of shape (B, T, D)

    Returns:
        x_a: Poses part of the motion tensor
        x_b: Trajectories part of the motion tensor
    """
    B, T, _ = x.shape

    poses = x[:, :, : 55 * 12].reshape(B, T, 55, 12)
    poses1 = poses[:, :, :, :6].reshape(B, T, -1)
    trans1 = x[:, :, 55 * 12 : 55 * 12 + 6]
    poses2 = poses[:, :, :, 6:].reshape(B, T, -1)
    trans2 = x[:, :, 55 * 12 + 6 : 55 * 12 + 12]

    x1 = torch.cat((poses1, trans1), dim=-1)
    x2 = torch.cat((poses2, trans2), dim=-1)

    return x1, x2


def merge_motion(x1, x2):
    """
    Merges two motion tensors back into a single tensor.
    This is the inverse of the split_motion function.

    Args:
        x1: The first motion part (poses1 and trans1).
        x2: The second motion part (poses2 and trans2).

    Returns:
        x: The reconstructed original motion tensor of shape (B, T, D).
    """
    # Get the batch size and sequence length from the input shape
    B, T, _ = x1.shape

    # 1. Extract the pose and trajectory parts from x1 and x2
    # The last 6 elements in each are the trajectory parts
    poses1 = x1[:, :, :-6]
    trans1 = x1[:, :, -6:]

    poses2 = x2[:, :, :-6]
    trans2 = x2[:, :, -6:]

    # 2. Reshape the flat pose parts back to their original 4D shape
    # The shape was (B, T, 55 * 6), so we reshape to (B, T, 55, 6)
    poses1_reshaped = poses1.reshape(B, T, 55, 6)
    poses2_reshaped = poses2.reshape(B, T, 55, 6)

    # 3. Concatenate the two pose parts along the last dimension
    # This reconstructs the original poses tensor of shape (B, T, 55, 12)
    poses = torch.cat((poses1_reshaped, poses2_reshaped), dim=-1)

    # 4. Flatten the poses tensor back to its 3D representation
    # Shape becomes (B, T, 55 * 12) or (B, T, 660)
    poses_flat = poses.reshape(B, T, -1)

    # 5. Concatenate the flattened poses and the two trajectory parts in the correct order
    x = torch.cat((poses_flat, trans1, trans2), dim=-1)

    return x


class TransDenoiser(nn.Module):
    """Part-aware denoiser branch for paired root translations."""

    def __init__(self, opt, pred_length):
        """Build one transformer denoising branch.

        Args:
            opt: Denoiser configuration namespace.
            pred_length: Number of prediction frames generated per Interact2Ar chunk.
        """

        super().__init__()

        self.opt = opt
        self.pred_length = pred_length

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.opt.TRANS_N_BLOCKS):
            self.blocks.append(
                TransformerBlock(
                    num_heads=self.opt.TRANS_N_HEADS,
                    latent_dim=self.opt.LATENT_DIM,
                    dropout=self.opt.DROPOUT,
                    ff_size=self.opt.TRANS_FF_SIZE,
                )
            )

        # Output Module
        self.out = zero_module(FinalLayer(self.opt.LATENT_DIM, self.opt.TRANS_REP // 2))

    def forward(self, h_x1, h_x2, emb, key_padding_mask):
        """
        Denoiser forward pass.

        Args:
            x: Noisy motion tensor of shape (B, T, D)
            timesteps: Denoising timesteps tensor of shape (B,)
            mask: Optional mask tensor of shape (B, T, 2) indicating valid motion parts
            cond: Optional condition tensor of shape (B, D) for additional conditioning
        Returns:
            output: Denoised motion tensor of shape (B, T, D)
        """
        h_x1_prev = h_x1
        h_x2_prev = h_x2

        # Process the input through the transformer blocks
        for _, block in enumerate(self.blocks):
            h_x1 = block(h_x1_prev, h_x2_prev, emb, key_padding_mask)
            h_x2 = block(h_x2_prev, h_x1_prev, emb, key_padding_mask)
            h_x1_prev = h_x1
            h_x2_prev = h_x2

        if self.pred_length is not None:
            h_x1 = h_x1[:, -self.pred_length :]
            h_x2 = h_x2[:, -self.pred_length :]

        out1 = self.out(h_x1)
        out2 = self.out(h_x2)

        return out1, out2


class PosesDenoiser(nn.Module):
    """Part-aware denoiser branch for body pose tokens."""

    def __init__(self, opt, pred_length):
        """Build one transformer denoising branch.

        Args:
            opt: Denoiser configuration namespace.
            pred_length: Number of prediction frames generated per Interact2Ar chunk.
        """

        super().__init__()

        self.opt = opt
        self.pred_length = pred_length

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.opt.N_BLOCKS):
            self.blocks.append(
                TransformerBlock(
                    num_heads=self.opt.N_HEADS,
                    latent_dim=self.opt.LATENT_DIM,
                    dropout=self.opt.DROPOUT,
                    ff_size=self.opt.FF_SIZE,
                )
            )

        # Output Module
        self.out = zero_module(
            FinalLayer(
                self.opt.LATENT_DIM,
                self.opt.MOTION_REP // 2 - self.opt.TRANS_REP - self.opt.HANDS_REP,
            )
        )

    def forward(self, h_x1, h_x2, emb, key_padding_mask):
        """
        Denoiser forward pass.

        Args:
            x: Noisy motion tensor of shape (B, T, D)
            timesteps: Denoising timesteps tensor of shape (B,)
            mask: Optional mask tensor of shape (B, T, 2) indicating valid motion parts
            cond: Optional condition tensor of shape (B, D) for additional conditioning
        Returns:
            output: Denoised motion tensor of shape (B, T, D)
        """
        h_x1_prev = h_x1
        h_x2_prev = h_x2

        # Process the input through the transformer blocks
        for _, block in enumerate(self.blocks):
            h_x1 = block(h_x1_prev, h_x2_prev, emb, key_padding_mask)
            h_x2 = block(h_x2_prev, h_x1_prev, emb, key_padding_mask)
            h_x1_prev = h_x1
            h_x2_prev = h_x2

        if self.pred_length is not None:
            h_x1 = h_x1[:, -self.pred_length :]
            h_x2 = h_x2[:, -self.pred_length :]

        out1 = self.out(h_x1)
        out2 = self.out(h_x2)

        return out1, out2


class HandsDenoiser(nn.Module):
    """Part-aware denoiser branch for hand pose tokens."""

    def __init__(self, opt, pred_length):
        """Build one transformer denoising branch.

        Args:
            opt: Denoiser configuration namespace.
            pred_length: Number of prediction frames generated per Interact2Ar chunk.
        """

        super().__init__()

        self.opt = opt
        self.pred_length = pred_length

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.opt.HANDS_N_BLOCKS):
            self.blocks.append(
                TransformerBlock(
                    num_heads=self.opt.HANDS_N_HEADS,
                    latent_dim=self.opt.LATENT_DIM,
                    dropout=self.opt.DROPOUT,
                    ff_size=self.opt.HANDS_FF_SIZE,
                )
            )

        # Output Module
        self.out = zero_module(FinalLayer(self.opt.LATENT_DIM, self.opt.HANDS_REP))

    def forward(self, h_x1, h_x2, emb, key_padding_mask):
        """
        Denoiser forward pass.

        Args:
            x: Noisy motion tensor of shape (B, T, D)
            timesteps: Denoising timesteps tensor of shape (B,)
            mask: Optional mask tensor of shape (B, T, 2) indicating valid motion parts
            cond: Optional condition tensor of shape (B, D) for additional conditioning
        Returns:
            output: Denoised motion tensor of shape (B, T, D)
        """
        h_x1_prev = h_x1
        h_x2_prev = h_x2

        # Process the input through the transformer blocks
        for _, block in enumerate(self.blocks):
            h_x1 = block(h_x1_prev, h_x2_prev, emb, key_padding_mask)
            h_x2 = block(h_x2_prev, h_x1_prev, emb, key_padding_mask)
            h_x1_prev = h_x1
            h_x2_prev = h_x2

        if self.pred_length is not None:
            h_x1 = h_x1[:, -self.pred_length :]
            h_x2 = h_x2[:, -self.pred_length :]

        out1 = self.out(h_x1)
        out2 = self.out(h_x2)

        return out1, out2


class PartAwareInteractionDenoiser(nn.Module):
    """Part-aware Interact2Ar denoiser for body, hands, and translations."""

    def __init__(self, opt, pred_length=None):
        """Build the part-aware denoiser used by the selected checkpoint.

        Args:
            opt: Denoiser configuration namespace.
            pred_length: Number of prediction frames generated per Interact2Ar chunk.
        """

        super().__init__()

        self.opt = opt

        self.poses_denoiser = PosesDenoiser(opt, pred_length)
        self.hands_denoiser = HandsDenoiser(opt, pred_length)
        self.trans_denoiser = TransDenoiser(opt, pred_length)

        # Positionals Encoding (motion lengths and denoising timesteps)
        self.sequence_pos_encoder = PositionalEncoding(self.opt.LATENT_DIM, dropout=0)
        self.embed_timestep = TimestepEmbedder(
            self.opt.LATENT_DIM, self.sequence_pos_encoder
        )

        # Input Embedding
        self.motion_embed = nn.Linear(self.opt.MOTION_REP // 2, self.opt.LATENT_DIM)
        self.text_embed = nn.Linear(self.opt.TEXT_EMB_DIM, self.opt.LATENT_DIM)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.opt.N_BLOCKS):
            self.blocks.append(
                TransformerBlock(
                    num_heads=self.opt.N_HEADS,
                    latent_dim=self.opt.LATENT_DIM,
                    dropout=self.opt.DROPOUT,
                    ff_size=self.opt.FF_SIZE,
                )
            )

    def forward(self, x, timesteps, mask=None, cond=None):
        """
        Denoiser forward pass.

        Args:
            x: Noisy motion tensor of shape (B, T, D)
            timesteps: Denoising timesteps tensor of shape (B,)
            mask: Optional mask tensor of shape (B, T, 2) indicating valid motion parts
            cond: Optional condition tensor of shape (B, D) for additional conditioning
        Returns:
            output: Denoised motion tensor of shape (B, T, D)
        """
        B, T = x.shape[0], x.shape[1]

        x1, x2 = split_motion(x)

        # Embed the conditions and the noisy motion
        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        x1_emb = self.motion_embed(x1)
        x2_emb = self.motion_embed(x2)

        h_x1_prev = self.sequence_pos_encoder(x1_emb)
        h_x2_prev = self.sequence_pos_encoder(x2_emb)

        # Prepare the key padding mask
        if mask is not None:
            mask = mask[..., 0]
        else:
            mask = torch.ones(B, T).to(x.device)
        key_padding_mask = ~(mask > 0.5)

        # Process the input through the transformer blocks
        for _, block in enumerate(self.blocks):
            h_x1 = block(h_x1_prev, h_x2_prev, emb, key_padding_mask)
            h_x2 = block(h_x2_prev, h_x1_prev, emb, key_padding_mask)
            h_x1_prev = h_x1
            h_x2_prev = h_x2

        poses1, poses2 = self.poses_denoiser(
            h_x1.clone(), h_x2.clone(), emb.clone(), key_padding_mask
        )
        hands1, hands2 = self.hands_denoiser(
            h_x1.clone(), h_x2.clone(), emb.clone(), key_padding_mask
        )
        trans1, trans2 = self.trans_denoiser(
            h_x1.clone(), h_x2.clone(), emb.clone(), key_padding_mask
        )

        motion1 = torch.cat([poses1, hands1, trans1, torch.zeros_like(trans1)], dim=-1)
        motion2 = torch.cat([poses2, hands2, trans2, torch.zeros_like(trans2)], dim=-1)

        motions = merge_motion(motion1, motion2)

        return motions
