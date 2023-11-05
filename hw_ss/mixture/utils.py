import os
from glob import glob
import numpy as np

import librosa


def snr_mixer(clean, noise, snr):
    """
    Mix audio with a noise at any signal-to-noise ratio
    """
    amp_noise = np.linalg.norm(clean) / 10 ** (snr / 20)
    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise
    mix = clean + noise_norm
    return mix


def vad_merge(x, top_db):
    """
    (Voice Activity Detector) Remove silent intervals from audio
    """
    intervals = librosa.effects.split(x, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(x[s:e])
    return np.concatenate(temp, axis=None)


def cut_audios(s1, s2, sec, sr):
    """
    Split audio into equal parts
    """
    cut_len = sr * sec

    len1, len2 = len(s1), len(s2)
    s1_cut, s2_cut = [], []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    return s1_cut, s2_cut


def fix_length(s1, s2, min_or_max='max'):
    """
    Fix audio lengths
    """
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


class LibriSpeechSpeakerFiles:
    def __init__(self, speaker_id, audios_dir, audio_template="*-norm.wav"):
        self.id = speaker_id
        self.files = []
        self.audio_template = audio_template
        self.files = self.find_files_by_worker(audios_dir)

    def find_files_by_worker(self, audios_dir):
        speaker_dir = os.path.join(audios_dir, self.id)
        chapter_dirs = os.scandir(speaker_dir)
        files = []
        for chapterDir in chapter_dirs:
            files = files + [
                file for file in glob(os.path.join(speaker_dir, chapterDir.name) + "/" + self.audio_template)
            ]
        return files
