import torch
from torch import Tensor
from torch import nn


class DepthWiseConv(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels,
                      out_channels=num_channels,
                      kernel_size=(kernel_size - 1) * dilation + 1,
                      dilation=dilation),
            nn.Conv1d(in_channels=num_channels,
                      out_channels=num_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int,
                 out_channels: int, kernel_size: int, block_num: int):
        super().__init__()
        self.prolog = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1),
            nn.PReLU())
        self.d_conv = nn.Sequential(
            DepthWiseConv(num_channels=mid_channels, kernel_size=kernel_size, dilation=2 ** block_num),
            nn.PReLU()
        )
        self.epilog = nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.prolog(x)
        x = nn.functional.layer_norm(x, normalized_shape=x.shape[1:])
        x = self.d_conv(x)
        x = nn.functional.layer_norm(x, normalized_shape=x.shape[1:])
        return self.epilog(x)


class TCNStack(nn.Module):
    def __init__(self, block_num: int, in_channels: int, mid_channels: int,
                 out_channels: int, kernel_size: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(in_channels=in_channels if i == 0 else out_channels,
                     mid_channels=mid_channels, out_channels=out_channels,
                     kernel_size=kernel_size, block_num=i) for i in range(block_num)
        ])

    def forward(self, x: Tensor, speaker_embedding: Tensor):
        _, _, h_1 = x.shape
        _, _, h_2 = speaker_embedding.shape
        n = (h_1 + h_2 - 1) // h_2
        x = torch.cat([x, speaker_embedding.repeat(1, 1, n)[:, :, :h_1]], dim=1)
        x = self.blocks[0](x)
        for block in self.blocks[1:]:
            x = block(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.prolog = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.PReLU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1),
            nn.BatchNorm1d(1)
        )
        self.epilog = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=(1, 3))
        )

    def forward(self, x):
        return self.epilog(self.prolog(x) + x)
