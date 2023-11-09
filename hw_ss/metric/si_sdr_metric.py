from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from torchmetrics import ScaleInvariantSignalDistortionRatio


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        return self.si_sdr(signal, target)
