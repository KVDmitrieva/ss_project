import numpy as np

from numpy.linalg import norm
from torch import Tensor

from hw_ss.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs):
        batch_size = 1 if len(preds.shape) == 1 else preds.shape[0]

        preds = preds.detach().numpy()
        targets = targets.detach().numpy()

        alpha = ((targets * preds).sum(axis=-1) / norm(targets, axis=-1) ** 2).reshape(-1, 1)
        result = 20 * np.log10(norm(alpha * targets, axis=-1) / (norm(alpha * targets - preds, axis=-1) + 1e-6) + 1e-6)

        return result.sum() / batch_size

