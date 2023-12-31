from torch import Tensor
from torch.nn import CrossEntropyLoss


class CELossWrapper(CrossEntropyLoss):
    def __init__(self):
        super().__init__()

    def forward(self, probs, target, **batch) -> Tensor:
        return super().forward(probs, target)
