import numpy as np
import torch
import torchcrepe  
import math
from diffsynth.util import pad_or_trim_to_expected_length
FMIN = 110.0
FMAX = 2000.0

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
        f0_hz, periodicity = torchcrepe.predict(audio, sample_rate, hop_length=hop_length, pad=True, device='cuda', batch_size=64, model='full', fmin=FMIN, fmax=FMAX, return_periodicity=True)

    f0_hz = f0_hz[0]
    periodicity = periodicity[0]

    n_secs = audio.shape[-1] / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)
    f0_hz = pad_or_trim_to_expected_length(f0_hz, expected_len, 'replicate')
    return f0_hz, periodicity

"""
Code below ported from torchyin
https://github.com/brentspell/torch-yin/blob/main/torchyin/yin.py
License:
    MIT License
    Copyright © 2022 Brent M. Spell
"""

def yin_frame(audio_frame, sample_rate:int , pitch_min:float =50, pitch_max:float =2000, threshold:float =0.1):
    # audio_frame: (n_frames, frame_length)
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    assert audio_frame.shape[-1] > tau_max
    
    cmdf = _diff(audio_frame, tau_max)[..., tau_min:]
    tau = _search(cmdf, tau_max, threshold)

    return sample_rate / (tau + tau_min + 1).type(audio_frame.dtype)

def _diff(frames: torch.Tensor, tau_max: int) -> torch.Tensor:
    # frames: n_frames, frame_length
    # compute the frame-wise autocorrelation using the FFT
    fft_size = int(2 ** (-int(-math.log(frames.shape[-1]) // math.log(2)) + 1))
    fft = torch.fft.rfft(frames, fft_size, dim=-1)
    corr = torch.fft.irfft(fft * fft.conj())[..., :tau_max]

    # difference function (equation 6)
    sqrcs = torch.nn.functional.pad((frames * frames).cumsum(-1), [1, 0])
    corr_0 = sqrcs[..., -1:]
    corr_tau = sqrcs.flip(-1)[..., :tau_max] - sqrcs[..., :tau_max]
    diff = corr_0 + corr_tau - 2 * corr

    # cumulative mean normalized difference function (equation 8)
    return (
        diff[..., 1:]
        * torch.arange(1, diff.shape[-1])
        / torch.clamp(diff[..., 1:].cumsum(-1), min=1e-5)
    )

@torch.jit.script
def _search(cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
    # mask all periods after the first cmdf below the threshold
    # if none are below threshold (argmax=0), this is a non-periodic frame
    first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
    first_below = torch.where(first_below > 0, first_below, tau_max)
    beyond_threshold = torch.arange(cmdf.shape[-1]) >= first_below

    # mask all periods with upward sloping cmdf to find the local minimum
    increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1.0)

    # find the first period satisfying both constraints
    return (beyond_threshold & increasing_slope).int().argmax(-1)