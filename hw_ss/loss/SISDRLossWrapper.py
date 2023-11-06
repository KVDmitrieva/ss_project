from torch import Tensor

from hw_ss.base.base_loss import BaseLoss
from hw_ss.metric.si_sdr_metric import SISDRMetric


class SISDRLossWrapper(BaseLoss):
    def __init__(self):
        super().__init__()
        self.si_sdr = SISDRMetric()

    def forward(self,  prediction, target, **batch) -> Tensor:
        return self.si_sdr(prediction, target)
