import numpy as np
import torch
import torch.nn as nn
import torchcrepe  
import functools
from diffsynth.util import pad_or_trim_to_expected_length
FMIN = 32
FMAX = 2000

def process_f0(f0_hz, periodicity):
    # Shape [1, 1 + int(time // hop_length,]
    # Postprocessing on f0_hz
    # replace unvoiced regions with NaN
    # win_length = 3
    # periodicity = torchcrepe.filter.mean(periodicity, win_length)
    threshold = 1e-3
    # if all noisy, do not perform thresholding
    if (periodicity > threshold).any():
        f0_hz = torchcrepe.threshold.At(1e-3)(f0_hz, periodicity)
    # f0_hz = torchcrepe.filter.mean(f0_hz, win_length)
    f0_hz = f0_hz[0]
    # interpolate Nans
    # https://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value
    f0_hz = f0_hz.numpy()
    mask = np.isnan(f0_hz)
    f0_hz[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), f0_hz[~mask])
    return torch.from_numpy(f0_hz)# Shape [1 + int(time // hop_length,]

def compute_f0(audio, sample_rate, frame_rate):
    """ For preprocessing
    Args:
        audio: torch.Tensor of single audio example. Shape [audio_length,].
        sample_rate: Sample rate in Hz.

    Returns:
        f0_hz: Fundamental frequency in Hz. Shape [n_frames]
        periodicity: Basically, confidence of pitch value. Shape [n_frames]
    """
    audio = audio[None, :]

    hop_length = sample_rate // frame_rate
    # Compute f0 with torchcrepe.
    # uses viterbi by default
    # pad=False is probably center=False
    # [output_shape=(1, 1 + int(time // hop_length))]
    with torch.no_grad():
        f0_hz, periodicity = torchcrepe.predict(audio, sample_rate, hop_length=hop_length, pad=True, device='cuda:1', batch_size=2048, model='full', fmin=FMIN, fmax=FMAX, return_periodicity=True)

    f0_hz = f0_hz[0]
    periodicity = periodicity[0]

    n_secs = audio.shape[-1] / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)
    f0_hz = pad_or_trim_to_expected_length(f0_hz, expected_len, 'replicate')
    return f0_hz, periodicity