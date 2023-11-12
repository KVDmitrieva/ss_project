import torch
from torch import Tensor

from hw_ss.base.base_loss import BaseLoss
from hw_ss.loss.SISDRLossWrapper import SISDRLossWrapper
from hw_ss.loss.CELossWrapper import CELossWrapper


class MultiTaskLossWrapper(BaseLoss):
    def __init__(self, alphas, gamma):
        super().__init__()
        self.cross_entropy = CELossWrapper()
        self.si_sdr = SISDRLossWrapper()
        self.alphas = alphas
        self.gamma = gamma

    def forward(self,  signals, target, log_probs, speaker, audio_len, **batch) -> Tensor:
        si_sdr_loss = torch.zeros_like(signals[:, 0])

        for i in range(signals.shape[1]):
            alpha = 1 - sum(self.alphas) if i == 0 else self.alphas[i - 1]
            si_sdr_loss -= alpha * self.si_sdr(signals[:, i], target)

        if log_probs is not None:
            ce_loss = self.cross_entropy(log_probs, speaker)
            si_sdr_loss += self.gamma * ce_loss

        return torch.mean(si_sdr_loss)
