"""Timestep samplers for diffusion training."""

import numpy as np
import torch as th


def create_named_schedule_sampler(name, diffusion):
    """Create the timestep sampler requested by the paper configuration.

    Args:
        name (str): Sampler name from the diffusion YAML file.
        diffusion: Diffusion process that defines the number of timesteps.

    Returns:
        UniformSampler: Uniform timestep sampler used in the main paper.
    """
    if name != "uniform":
        raise NotImplementedError(f"unknown schedule sampler: {name}")
    return UniformSampler(diffusion)


class UniformSampler:
    """Uniformly samples diffusion timesteps and returns unbiased weights."""

    def __init__(self, diffusion):
        """Store a uniform distribution over the diffusion timesteps.

        Args:
            diffusion: Diffusion process with a ``num_timesteps`` attribute.
        """
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        """Return one unnormalized weight per diffusion timestep.

        Returns:
            np.ndarray: Positive timestep weights.
        """
        return self._weights

    def sample(self, batch_size, device):
        """Sample timesteps for one training batch.

        Args:
            batch_size (int): Number of timesteps to sample.
            device: Torch device that receives the tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sampled timestep ids and weights.
        """
        weights = self.weights()
        probabilities = weights / np.sum(weights)
        indices_np = np.random.choice(len(probabilities), size=(batch_size,), p=probabilities)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(probabilities) * probabilities[indices_np])
        sampler_weights = th.from_numpy(weights_np).float().to(device)
        return indices, sampler_weights
