import logging

import torchaudio
from tqdm import tqdm

from hw_ss.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = []
        for ref, mix, target in tqdm(data, desc="Preparing mixture triplets"):
            t_info = torchaudio.info(mix)
            length = t_info.num_frames / t_info.sample_rate
            index.append(
                {
                    "path": mix,
                    "ref_path": ref,
                    "target_path": target,
                    "audio_len": length,
                    "speaker": -1,
                    "speaker_id": -1
                }
            )

        super().__init__(index, *args, **kwargs)
