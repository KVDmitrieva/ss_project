from torch import nn
from torch import Tensor


class BaseLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, **batch) -> Tensor:
        raise NotImplementedError()
