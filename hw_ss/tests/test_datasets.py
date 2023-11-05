import unittest

import torch

from hw_ss.datasets import LibrispeechDataset, CustomDirAudioDataset, CustomAudioDataset, MixtureDataset
from hw_ss.tests.utils import clear_log_folder_after_use
from hw_ss.utils import ROOT_PATH
from hw_ss.utils.parse_config import ConfigParser


class TestDataset(unittest.TestCase):
    def test_librispeech(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean",
                config_parser=config_parser,
                max_audio_length=13,
                limit=10,
            )
            self._assert_training_example_is_good(ds[0])

    def test_mixture(self):
        config_parser = ConfigParser.get_test_configs()

        init_params = {"n_files": 100, "test": False}
        gen_params = {"snr_levels": [-5, 5],
                       "num_workers": 2,
                       "update_steps": 100,
                       "trim_db": 20,
                       "vad_db": 20,
                       "audioLen": 3}

        mixture_params = {"mixture_init_params": init_params, "mixture_generate_params": gen_params}

        with clear_log_folder_after_use(config_parser):
            ds = MixtureDataset(
                "dev-clean",
                config_parser=config_parser,
                limit=10,
                generate_mixture=True,
                max_audio_length=13,
                mixture_params=mixture_params
            )
            self._assert_training_example_is_good(ds[0])

    def test_custom_dir_dataset(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            audio_dir = str(ROOT_PATH / "test_data" / "audio")
            transc_dir = str(ROOT_PATH / "test_data" / "transcriptions")

            ds = CustomDirAudioDataset(
                audio_dir,
                transc_dir,
                config_parser=config_parser,
                limit=10,
                max_audio_length=8
            )
            self._assert_training_example_is_good(ds[0])

    def test_custom_dataset(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            audio_path = ROOT_PATH / "test_data" / "audio"
            transc_path = ROOT_PATH / "test_data" / "transcriptions"
            with (transc_path / "84-121550-0000.txt").open() as f:
                transcription = f.read().strip()
            data = [
                {
                    "path": str(audio_path / "84-121550-0001.flac"),
                },
                {
                    "path": str(audio_path / "84-121550-0000.flac"),
                    "text": transcription
                }
            ]

            ds = CustomAudioDataset(
                data=data,
                config_parser=config_parser,
            )
            self._assert_training_example_is_good(ds[0], contains_text=False)
            self._assert_training_example_is_good(ds[1])

    def _assert_training_example_is_good(self, training_example: dict, contains_text=True):

        for field, expected_type in [
            ("audio", torch.Tensor),
            ("spectrogram", torch.Tensor),
            ("duration", float),
            ("audio_path", str),
            ("text", str),
        ]:
            self.assertIn(field, training_example, f"Error during checking field {field}")
            self.assertIsInstance(training_example[field], expected_type,
                                  f"Error during checking field {field}")

        # check waveform dimensions
        batch_dim, audio_dim, = training_example["audio"].size()
        self.assertEqual(batch_dim, 1)
        self.assertGreater(audio_dim, 1)

        # check spectrogram dimensions
        batch_dim, freq_dim, time_dim = training_example["spectrogram"].size()
        self.assertEqual(batch_dim, 1)
        self.assertEqual(freq_dim, 128)
        self.assertGreater(time_dim, 1)

        # check text tensor dimensions
        length_dim = len(training_example["text"])
        if contains_text:
            self.assertGreater(length_dim, 1)
        else:
            self.assertEqual(length_dim, 0)
            self.assertEqual(training_example["text"], "")
