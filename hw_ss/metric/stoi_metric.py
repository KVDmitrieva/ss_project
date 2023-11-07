from torch import Tensor
from torchmetrics.audio import ShortTimeObjectiveIntelligibility

from hw_ss.base.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = ShortTimeObjectiveIntelligibility(fs)

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        return self.stoi(preds, targets)
