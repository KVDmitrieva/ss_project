from torch import Tensor

from hw_ss.loss.base_loss import BaseLoss
from hw_ss.metric.utils import calc_si_sdr


class SISDRLossWrapper(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self,  prediction, target, **batch) -> Tensor:
        return calc_si_sdr(prediction, target)
