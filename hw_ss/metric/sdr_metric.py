import numpy as np
from torch import Tensor

from torchmetrics.audio import SignalDistortionRatio

from hw_ss.base.base_metric import BaseMetric
from hw_ss.metric.utils import get_r_b


class SDRMetric(BaseMetric):
    # SDR from https://arxiv.org/pdf/2110.06440.pdf
    # use SDR from torchmetrics for a while
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdr = SignalDistortionRatio()

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs):
        return self.sdr(preds, targets)



