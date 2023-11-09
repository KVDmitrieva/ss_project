import logging
import torch
import torch.nn.functional as F
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio_len = []
    text, speaker = [], []
    ref, ref_path = [], []
    audio, audio_path = [], []
    target_path, target = [], []
    spectrogram, spectrogram_length = [], []

    for item in dataset_items:
        text.append(item["text"])
        ref.append(item["ref"].T)
        audio.append(item["audio"].T)
        target.append(item["target"].T)
        speaker.append(item["speaker_id"])
        ref_path.append(item["ref_path"])
        audio_path.append(item["audio_path"])
        target_path.append(item["target_path"])
        audio_len.append(item["audio"].shape[-1])
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        spectrogram_length.append(item["spectrogram"].shape[2])

    audio_target = pad_sequence(audio + target, batch_first=True)
    audio_target = F.pad(audio_target, pad=(0, 0, 0, audio_target.shape[1] % 2, 0, 0))
    audio, target = audio_target[:len(dataset_items)].transpose(1, 2), audio_target[len(dataset_items):].transpose(1, 2)
    target = target.squeeze(1)

    return {
        "text": text,
        "audio": audio,
        "target": target,
        "ref_path": ref_path,
        "audio_path": audio_path,
        "target_path": target_path,
        "speaker": torch.tensor(speaker),
        "audio_len": torch.tensor(audio_len),
        "spectrogram_length": torch.tensor(spectrogram_length),
        "ref": pad_sequence(ref, batch_first=True).transpose(1, 2),
        "spectrogram":  pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    }
