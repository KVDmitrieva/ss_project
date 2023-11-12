import os
import logging
from pathlib import Path
from glob import glob

from tqdm import tqdm

import torchaudio

from hw_ss.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, *args, **kwargs):
        audio_dir = Path(audio_dir)
        refs = sorted(glob(os.path.join(audio_dir / "refs", '*-ref.wav')))
        mixes = sorted(glob(os.path.join(audio_dir / "mix", '*-mixed.wav')))
        targets = sorted(glob(os.path.join(audio_dir / "targets", '*-target.wav')))

        data = list(zip(refs, mixes, targets))

        super().__init__(data, *args, **kwargs)
