import numpy as np
import torch
from torch import nn


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning-rate scheduler with linear warmup and cosine decay."""
    def __init__(self, optimizer, warmup, max_iters):
        """Configure warmup length and total scheduler duration.

        Args:
            optimizer: Optimizer whose learning rate will be scheduled.
            warmup: Number of initial epochs used for linear warmup.
            max_iters: Total number of epochs in the cosine schedule.
        """
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        """Return scheduled learning rates for all optimizer groups.

        Returns:
            list[float]: Learning rate for each optimizer parameter group.
        """
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """Compute the scalar warmup/cosine multiplier for one epoch.

        Args:
            epoch: Current scheduler epoch.

        Returns:
            float: Multiplicative factor applied to base learning rates.
        """
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1) * 1.0 / self.warmup
        return lr_factor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        """Precompute sinusoidal positions.

        Args:
            d_model: Transformer feature dimension.
            dropout: Dropout probability applied after adding positions.
            max_len: Maximum supported sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encodings to a sequence tensor.

        Args:
            x: Tensor with shape ``(B, T, D)``.

        Returns:
            torch.Tensor: Position-aware tensor with the same shape as ``x``.
        """
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps with sinusoidal positions and an MLP."""
    def __init__(self, latent_dim, sequence_pos_encoder):
        """Build the timestep embedding MLP.

        Args:
            latent_dim: Embedding dimension used by the denoiser.
            sequence_pos_encoder: Positional encoding table indexed by timesteps.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        """Embed integer diffusion timesteps.

        Args:
            timesteps: Timestep indices with shape ``(B,)``.

        Returns:
            torch.Tensor: Timestep embeddings with shape ``(B, latent_dim)``.
        """
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


def set_requires_grad(modules, requires_grad=False):
    """Enable or disable gradients for one module or a list of modules.

    Args:
        modules (nn.Module | list[nn.Module]): Module or modules to update.
        requires_grad (bool): Whether parameters should receive gradients.
    """
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad



def zero_module(module):
    """Zero the parameters of a module before returning it.

    Args:
        module (nn.Module): Module whose parameters should start at zero.

    Returns:
        nn.Module: The same module, with all parameters zeroed in place.
    """
    # Zero-initialized output layers make residual diffusion blocks start gently.
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module
