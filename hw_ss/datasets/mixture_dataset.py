import json
import logging
import os
import shutil
from glob import glob
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_ss.mixture.utils import LibriSpeechSpeakerFiles
from hw_ss.mixture.mixture_generator import MixtureGenerator
from hw_ss.datasets.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class MixtureDataset(BaseDataset):
    def __init__(self, part, mixture_dir=None, index_dir=None, source_dir=None,
                 generate_mixture=False, mixture_params=None, add_texts=False, *args, **kwargs):
        if generate_mixture:
            assert source_dir is not None or part in URL_LINKS, \
                "Provide source dir path or choose correct librispeech part to download"

        self._source_dir = source_dir
        self._mixture_dir = mixture_dir
        self._generated_triplets = None
        self._add_texts = add_texts

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "datasets" / "mixture"
            index_dir.mkdir(exist_ok=True, parents=True)
        else:
            index_dir = Path(index_dir)

        self._index_dir = index_dir

        if generate_mixture:
            self._generated_triplets = self._generate_mix(part, **mixture_params)

        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        self._source_dir = self._index_dir / "source"
        arch_path = self._index_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._source_dir)
        for fpath in (self._source_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._source_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._source_dir / "LibriSpeech"))
        self._source_dir = self._source_dir / part

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []

        refs = sorted(glob(os.path.join(self._mixture_dir, '*-ref.wav')))
        mixes = sorted(glob(os.path.join(self._mixture_dir, '*-mixed.wav')))
        targets = sorted(glob(os.path.join(self._mixture_dir, '*-target.wav')))

        target_speakers = sorted(list(set([int(path.split('/')[-1].split('_')[0]) for path in targets])))
        self._speaker_dict = {speaker: i for i, speaker in enumerate(target_speakers)}

        for ref, mix, target in tqdm(
                zip(refs, mixes, targets), desc="Preparing mixture triplets"
        ):
            target_speaker = int(mix.split('/')[-1].split('_')[0])
            t_info = torchaudio.info(mix)
            length = t_info.num_frames / t_info.sample_rate
            index.append(
                {
                    "path": mix,
                    "ref_path": ref,
                    "target_path": target,
                    "audio_len": length,
                    "speaker": target_speaker,
                    "speaker_id": self._speaker_dict[target_speaker]
                }
            )

            if self._add_texts:
                idx = int(mix.split('/')[-1].split('_')[2])
                target_path = self._generated_triplets["target"][idx]
                target_file = target_path.split('/')[-1]
                target_idx = int(target_path.split('-')[-1].split('.')[0])

                text_file = '-'.join(target_file.split('-')[:-1]) + ".trans.txt"
                text_path = "/".join(target_path.split('/')[:-1]) + '/' + text_file

                if Path(text_path).exists():
                    with Path(text_path).open() as f:
                        f_text = " ".join(f.readlines()[target_idx].split()[1:])
                        index[-1]["text"] = f_text.lower()
                else:
                    index[-1]["text"] = ""
                print("DEBUG", target_path, text_path, index[-1]["text"])
        return index

    def _generate_mix(self, part, mixture_init_params, mixture_generate_params):
        if self._source_dir is None:
            self._load_part(part)

        if self._mixture_dir is None:
            self._mixture_dir = self._index_dir

        self._mixture_dir = self._mixture_dir / part / "mixture"

        speakers = [el.name for el in os.scandir(self._source_dir)]

        speakers_file = [LibriSpeechSpeakerFiles(i, self._source_dir, audio_template="*.flac") for i in speakers]

        mixture = MixtureGenerator(speakers_file, self._mixture_dir, **mixture_init_params)
        triplets = mixture.generate_mixes(**mixture_generate_params)

        return triplets
