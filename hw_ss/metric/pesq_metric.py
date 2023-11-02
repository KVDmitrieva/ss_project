from torch import Tensor
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from hw_ss.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, fs, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs):
        return self.pesq(preds, targets)
