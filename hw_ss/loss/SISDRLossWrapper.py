from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_ss.metric.si_sdr_metric import SISDRMetric


class SISDRLossWrapper:
    def __init__(self):
        self.si_sdr = SISDRMetric()

    def forward(self,  prediction, target, alpha=0.5, **batch) -> Tensor:
        raise NotImplementedError()
