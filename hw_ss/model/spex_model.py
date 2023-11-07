from typing import Union

import torch
from torch import Tensor
from torch import nn

from hw_ss.base import BaseModel
from hw_ss.model.utils import TCNStack, ResNetBlock


class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, avg_pool, out_features, num_blocks):
        super().__init__()
        self.prolog = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        )
        self.res = nn.ModuleList([
            ResNetBlock(num_channels=mid_channels) for _ in range(num_blocks)
        ])
        self.speaker = nn.Sequential(
            nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=avg_pool)
        )

        self.epilog = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_features),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.prolog(x)
        for resnet in self.res:
            x = resnet(x)

        v = self.speaker(x)
        logits = self.epilog(v.transpose(1, 2))

        return logits, v


class SpeakerExtractor(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, speaker_dim, num_stack, mask_num, tcn_params):
        super().__init__()
        self.prolog = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        )
        self.tcn = nn.ModuleList([
            TCNStack(in_channels=mid_channels + speaker_dim,
                     out_channels=mid_channels, **tcn_params) for _ in range(num_stack)
        ])

        self.mask = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1),
                nn.ReLU()
            ) for _ in range(mask_num)
        ])

    def forward(self, y, v) -> list[Tensor]:
        y = self.prolog(y)

        for tcn in self.tcn:
            y = tcn(y, v)

        masks = [mask(y) for mask in self.mask]
        return masks


class SpEXModel(BaseModel):
    def __init__(self, n_feats, n_class, speech_out, padding: list, speaker_dim,
                 filter_lengths: list, encoder_params: dict, extractor_params: dict, **batch):
        super().__init__(n_feats, n_class, **batch)

        n = len(filter_lengths)

        self.speech_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=n_feats, out_channels=speech_out, kernel_size=filter_lengths[i],
                          stride=filter_lengths[0] // 2, padding=padding[i]),
                nn.ReLU()
            ) for i in range(n)
        ])

        self.speaker_encoder = SpeakerEncoder(in_channels=3 * speech_out, out_features=n_class,
                                              out_channels=speaker_dim, **encoder_params)
        self.speaker_extractor = SpeakerExtractor(in_channels=3 * speech_out, mask_num=n,
                                                  out_channels=speech_out, speaker_dim=speaker_dim, **extractor_params)

        self.decoders = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=speech_out, out_channels=n_feats, kernel_size=filter_lengths[i],
                               stride=filter_lengths[0] // 2, padding=padding[i]) for i in range(n)
        ])

    def forward(self, audio, ref, **batch) -> Union[Tensor, dict]:
        encoder_outputs = [speech_encoder(audio) for speech_encoder in self.speech_encoder]

        x = torch.cat([speech_encoder(ref) for speech_encoder in self.speech_encoder], dim=1)
        logits, v = self.speaker_encoder(x)

        masks = self.speaker_extractor(torch.cat(encoder_outputs, dim=1), v)

        signals = []
        for i, y in enumerate(encoder_outputs):
            signals.append(self.decoders[i](masks[i] * y).unsqueze(1))

        signals = torch.cat(signals, dim=1)
        return {"signals": signals, "logits": logits}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
