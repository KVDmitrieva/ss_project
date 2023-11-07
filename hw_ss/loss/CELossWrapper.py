import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot


class CELossWrapper(CrossEntropyLoss):
    def __init__(self):
        super().__init__()

    def forward(self, probs, target, **batch) -> Tensor:
        if probs.shape != target:
            target = one_hot(target, num_classes=probs.shape[-1]).float()
        return super().forward(probs, target)
