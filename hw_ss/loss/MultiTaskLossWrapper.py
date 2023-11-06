from torch import Tensor

from hw_ss.base.base_loss import BaseLoss
from hw_ss.loss.SISDRLossWrapper import SISDRLossWrapper
from hw_ss.loss.CELossWrapper import CELossWrapper


class MultiTaskLossWrapper(BaseLoss):
    def __init__(self):
        super().__init__()
        self.cross_entropy = CELossWrapper()
        self.si_sdr = SISDRLossWrapper()

    def forward(self,  signals, log_probs, target, speaker, gamma, alphas, **batch) -> Tensor:
        si_sdr_loss = 0.
        for i in range(signals.shape[0]):
            alpha = 1 - sum(alphas) if i == 0 else alphas[i - 1]
            si_sdr_loss += alpha * self.si_sdr(signals[:, i], target)

        if log_probs is not None:
            ce_loss = self.cross_entropy(log_probs, speaker)
            return si_sdr_loss + gamma * ce_loss

        return si_sdr_loss

