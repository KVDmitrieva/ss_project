from torch import Tensor

from hw_ss.base.base_metric import BaseMetric
from hw_ss.metric.utils import calc_si_sdr


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, signal: Tensor, target: Tensor, **kwargs):
        signal = signal.cpu().detach()
        target = target.cpu().detach()
        return calc_si_sdr(signal, target)
