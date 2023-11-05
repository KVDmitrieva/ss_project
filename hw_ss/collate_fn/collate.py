import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectrogram, audio_path = [], []
    spectrogram_length, text = [], []
    ref_path, target_path = [], []

    for item in dataset_items:
        text.append(item["text"])
        ref_path.append(item["ref_path"])
        audio_path.append(item["audio_path"])
        target_path.append(item["target_path"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        spectrogram_length.append(item["spectrogram"].shape[2])

    return {
        "text": text,
        "audio": item["audio"],
        "ref_path": ref_path,
        "audio_path": audio_path,
        "target_path": target_path,
        "spectrogram_length": torch.tensor(spectrogram_length),
        "spectrogram":  torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    }
