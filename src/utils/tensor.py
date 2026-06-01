import torch


def to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.

    :param tensor: A PyTorch tensor or a NumPy array.
    :return: A NumPy array.
    """
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    """
    Convert a NumPy array or a PyTorch tensor to a PyTorch tensor.

    :param ndarray: A NumPy array or a PyTorch tensor.
    :return: A PyTorch tensor.
    """
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray
