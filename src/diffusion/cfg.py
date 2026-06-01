import torch
import torch.nn as nn


class ClassifierFreeSampleModel(nn.Module):
    """Wrap a denoiser with classifier-free guidance at sampling time."""

    def __init__(self, model, cfg_scale):
        """Store the conditional denoiser and CFG interpolation weight.

        Args:
            model: Denoiser that accepts conditional and unconditional batches.
            cfg_scale: Guidance scale; values above 1.0 push samples toward text.
        """
        super().__init__()
        self.model = model
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None):
        """Run conditional and unconditional denoising in one batched call.

        Args:
            x: Noisy motion tensor with shape ``(B, T, D)``.
            timesteps: Diffusion timestep tensor with shape ``(B,)``.
            cond: Optional text-conditioning tensor with shape ``(B, C)``.
            mask: Optional temporal/person mask passed through to the denoiser.

        Returns:
            torch.Tensor: Guided denoiser prediction with shape ``(B, T, D)``.
        """
        batch_size = x.shape[0]

        # Duplicate the batch so the denoiser sees conditional and null text together.
        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask)
        out_cond = out[:batch_size]
        out_uncond = out[batch_size:]

        return self.s * out_cond + (1 - self.s) * out_uncond
