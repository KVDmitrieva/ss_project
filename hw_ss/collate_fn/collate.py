import logging
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    text, speaker = [], []
    ref, ref_path = [], []
    audio, audio_path = [], []
    target_path, target = [], []
    spectrogram, spectrogram_length = [], []

    for item in dataset_items:
        text.append(item["text"])
        speaker.append(item["speaker_id"])
        ref_path.append(item["ref_path"])
        ref.append(item["ref"].unsqueeze(0))
        audio_path.append(item["audio_path"])
        audio.append(item["audio"].unsqueeze(0))
        target_path.append(item["target_path"])
        target.append(item["target"].unsqueeze(0))
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        spectrogram_length.append(item["spectrogram"].shape[2])

    audio_target = pad_sequence(audio + target, batch_first=True)
    audio, target = audio_target[:len(dataset_items)], audio_target[len(audio_target):]

    return {
        "text": text,
        "audio": audio,
        "target": target,
        "speaker": speaker,
        "ref_path": ref_path,
        "audio_path": audio_path,
        "target_path": target_path,
        "ref": pad_sequence(ref, batch_first=True),
        "spectrogram_length": torch.tensor(spectrogram_length),
        "spectrogram":  pad_sequence(spectrogram, batch_first=True).transpose(1, 2)
    }
