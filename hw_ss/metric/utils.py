from torch import norm, log10
from torch import Tensor


def get_r_b(preds, targets):
    """
    A contains L shifts of target,
    R = A^T @ A
    b = A^T @ preds
    """
    raise NotImplemented()


def calc_si_sdr(preds: Tensor, targets: Tensor):
    batch_size = 1 if len(preds.shape) == 1 else preds.shape[0]

    alpha = ((targets * preds).sum(axis=-1) / norm(targets, dim=-1) ** 2).reshape(-1, 1)
    result = 20 * log10(norm(alpha * targets, dim=-1) / (norm(alpha * targets - preds, dim=-1) + 1e-6) + 1e-6)

    return result.sum() / batch_size

