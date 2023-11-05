import json
import logging
import os
import shutil
from glob import glob

from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_ss.mixture.utils import LibriSpeechSpeakerFiles
from hw_ss.mixture.mixture_generator import MixtureGenerator
from hw_ss.base.base_dataset import BaseDataset
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
                 generate_mixture=False, mixture_params=None, *args, **kwargs):
        if generate_mixture:
            assert source_dir is not None or part in URL_LINKS, \
                "Provide source dir path or choose correct librispeech part to download"

        self._source_dir = source_dir
        self._mixture_dir = mixture_dir

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "datasets" / "mixture"
            index_dir.mkdir(exist_ok=True, parents=True)

        self._index_dir = index_dir

        if generate_mixture:
            self._generate_mix(part, **mixture_params)

        assert self._mixture_dir is not None, "provide mixture path or generate mix"

        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
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

        for ref, mix, target in tqdm(
                zip(refs, mixes, targets), desc="Preparing mixture triplets"
        ):
            index.append(
                {
                    "path": mix,
                    "ref_path": ref,
                    "target_path": target,
                    "text": "",
                    "audio_len": None
                }
            )

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
        mixture.generate_mixes(**mixture_generate_params)
