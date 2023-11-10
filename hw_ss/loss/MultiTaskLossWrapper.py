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
        max_len = target.shape[-1]
        mask = torch.arange(max_len).expand(len(audio_len), max_len).to(audio_len.device) < audio_len.unsqueeze(1)
        masked_signal = torch.zeros_like(signals)
        masked_target = torch.zeros_like(target)
        masked_target[mask] = target[mask]
        mask = mask.unsqueeze(1).repeat(1, signals.shape[1], 1)
        masked_signal[mask] = signals[mask]

        si_sdr_loss = torch.zeros_like(signals[:, 0])

        for i in range(signals.shape[1]):
            alpha = 1 - sum(self.alphas) if i == 0 else self.alphas[i - 1]
            si_sdr_loss -= alpha * self.si_sdr(masked_signal[:, i], masked_target)

        if log_probs is not None:
            ce_loss = self.cross_entropy(log_probs, speaker)
            si_sdr_loss += self.gamma * ce_loss

        return torch.mean(si_sdr_loss)
