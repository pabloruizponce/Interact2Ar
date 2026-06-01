"""Small neural-network helpers used by the diffusion objective."""


def mean_flat(tensor):
    """Average a tensor over every non-batch dimension.

    Args:
        tensor: Tensor whose first dimension is the batch dimension.

    Returns:
        Tensor: Per-sample means with shape ``(batch_size,)``.
    """
    # Diffusion losses are reported per sample, so the batch axis is preserved.
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
