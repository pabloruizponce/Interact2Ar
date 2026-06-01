import numpy as np
from scipy import linalg


def euclidean_distance_matrix(matrix1, matrix2):
    """Compute pairwise Euclidean distances between two embedding matrices.

    Args:
        matrix1: Matrix with shape ``(N1, D)``.
        matrix2: Matrix with shape ``(N2, D)``.

    Returns:
        np.ndarray: Distance matrix with shape ``(N1, N2)``.
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    return np.sqrt(d1 + d2 + d3)


def calculate_top_k(mat, top_k):
    """Convert sorted retrieval indices into top-k correctness flags.

    Args:
        mat: Retrieval index matrix where each row is sorted by distance.
        top_k: Number of retrieval ranks to evaluate.

    Returns:
        np.ndarray: Boolean matrix with cumulative top-k correctness per row.
    """
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    return np.concatenate(top_k_list, axis=1)


def calculate_activation_statistics(activations):
    """Compute mean and covariance for motion embeddings.

    Args:
        activations: Embedding matrix with shape ``(N, D)``.

    Returns:
        tuple: Mean vector and covariance matrix.
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    """Estimate diversity from random pairs of generated embeddings.

    Args:
        activation: Embedding matrix with shape ``(N, D)``.
        diversity_times: Number of random pairs to sample.

    Returns:
        float: Average Euclidean distance between sampled pairs.
    """
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    """Estimate multimodality from repeated generations per caption.

    Args:
        activation: Embeddings with shape ``(N, repeats, D)``.
        multimodality_times: Number of repeat pairs sampled per caption.

    Returns:
        float: Average distance between repeated generations for the same caption.
    """
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet distance between two Gaussian embedding distributions.

    Args:
        mu1: Mean vector for generated motion embeddings.
        sigma1: Covariance matrix for generated motion embeddings.
        mu2: Mean vector for ground-truth motion embeddings.
        sigma2: Covariance matrix for ground-truth motion embeddings.
        eps: Diagonal offset used when the covariance product is singular.

    Returns:
        float: Frechet distance, used as FID in the text-motion protocol.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            "FID covariance product is singular; "
            f"adding {eps} to the covariance diagonals."
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imaginary = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {max_imaginary}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
