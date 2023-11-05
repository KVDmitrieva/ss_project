import os

import numpy as np

import librosa
import soundfile as sf
import pyloudnorm as pyln

from hw_ss.mixture.utils import vad_merge, cut_audios, snr_mixer, fix_length


def create_mix(idx, triplet, snr_levels, out_dir, test=False, sr=16000, **kwargs):
    trim_db, vad_db = kwargs["trim_db"], kwargs["vad_db"]
    audio_len = kwargs["audioLen"]

    s1, s2, ref = read_files(triplet)

    meter = pyln.Meter(sr)  # create BS.1770 meter

    s1_norm = normalize_loudness(meter, s1, -29)
    s2_norm = normalize_loudness(meter, s2, -29)
    ref_norm = normalize_loudness(meter, ref)

    if silence_check(s1_norm, s2_norm, ref_norm):
        return

    # cut silence at the beginning / ending
    if trim_db:
        ref, _ = librosa.effects.trim(ref_norm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1_norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2_norm, top_db=trim_db)

    if len(ref) < sr:
        return

    snr = np.random.choice(snr_levels, 1).item()

    if not test:
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audio_len, sr)

        for i in range(len(s1_cut)):
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            s1_cut[i] = normalize_loudness(meter, s1_cut[i])
            mix = normalize_loudness(meter, mix)

            write_files(mix, s1_cut[i], ref, sr, idx, triplet, out_dir, cut=i)

    else:
        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)

        s1 = normalize_loudness(meter, s1)
        mix = normalize_loudness(meter, mix)

        write_files(mix, s1, ref, sr, idx, triplet, out_dir)


def read_files(triplet):
    s1, _ = sf.read(os.path.join('', triplet["target"]))
    s2, _ = sf.read(os.path.join('', triplet["noise"]))
    ref, _ = sf.read(os.path.join('', triplet["reference"]))

    return s1, s2, ref


def normalize_loudness(meter, signal, target_loudness=-23.0):
    input_loudness = meter.integrated_loudness(signal)
    return pyln.normalize.loudness(signal, input_loudness, target_loudness)


def silence_check(s1, s2, ref):
    amp_s1 = np.max(np.abs(s1))
    amp_s2 = np.max(np.abs(s2))
    amp_ref = np.max(np.abs(ref))

    return amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0


def write_files(mix, target, ref, sr, idx, triplet, out_dir, cut=None):
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    cut_ind = "" if cut is None else f"_{cut}"

    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + cut_ind + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + cut_ind + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + cut_ind + "-ref.wav")

    sf.write(path_mix, mix, sr)
    sf.write(path_target, target, sr)
    sf.write(path_ref, ref, sr)
