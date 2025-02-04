import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import MelScale
from torchaudio.functional import create_dct

amp = lambda x: x[...,0]**2 + x[...,1]**2
DB_RANGE = 80.0

def spectrogram(audio, size=2048, hop_length=1024, power=2, center=True, window=None):
    power_spec = amp(torch.view_as_real(torch.stft(audio, size, window=window, hop_length=hop_length, center=center, return_complex=True)))
    if power == 2:
        spec = power_spec
    elif power == 1:
        spec = power_spec.sqrt()
    return spec

def compute_lsd(resyn_audio, orig_audio):
    window = torch.hann_window(1024).to(orig_audio.device)
    orig_power_s = spectrogram(orig_audio, 1024, 256, window=window).detach()
    resyn_power_s = spectrogram(resyn_audio, 1024, 256, window=window).detach()
    lsd = torch.sqrt(((10 * (torch.log10(resyn_power_s+1e-5)-torch.log10(orig_power_s+1e-5)))**2).sum(dim=(1,2))) / orig_power_s.shape[-1]
    lsd = lsd.mean()
    return lsd

def spectral_convergence(resyn_audio, target_audio):
    window = torch.hann_window(1024).to(target_audio.device)
    target_power_s = spectrogram(target_audio, 1024, 256, window=window).detach()
    resyn_power_s = spectrogram(resyn_audio, 1024, 256, window=window).detach()
    sc_loss = torch.linalg.norm(resyn_power_s - target_power_s, 'fro', dim=(1,2)) / torch.linalg.norm(target_power_s, 'fro', dim=(1,2))
    return sc_loss.mean()

def multiscale_fft(audio, sizes=[64, 128, 256, 512, 1024, 2048], hop_lengths=None, windows=None) -> torch.Tensor:
    """multiscale fft power spectrogram
    uses torch.stft so it should be differentiable

    Args:
        audio : (batch) input audio tensor Shape: [(batch), n_samples]
        sizes : fft sizes. Defaults to [64, 128, 256, 512, 1024, 2048].
        overlap : overlap between windows. Defaults to 0.75.
    """
    specs = []
    if hop_lengths is None:
        overlap = 0.75
        hop_lengths = [int((1-overlap)*s) for s in sizes]
    if windows is None:
        windows = [torch.hann_window(s, device=audio.device) for s in sizes]
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    stft_params = zip(sizes, hop_lengths, windows)
    for n_fft, hl, win in stft_params:
        wl = win.shape[-1]
        stft = torch.stft(audio, n_fft, window=win, hop_length=hl, win_length=wl, center=False, return_complex=True)
        stft = torch.view_as_real(stft)
        specs.append(amp(stft))
    return specs

def spec_loudness(spec, a_weighting: torch.Tensor, range_db:float=DB_RANGE, ref_db:float=0.0):
    """
    Args:
        spec: Shape [..., freq_bins]
    """
    power = spec.real**2+spec.imag**2
    weighting = 10**(a_weighting/10) #db to linear
    weighted_power = power * weighting
    avg_power = torch.mean(weighted_power, dim=-1)
    # to db
    min_power = 10**-(range_db / 10.0)
    power = torch.clamp(avg_power, min=min_power)
    db = 10.0 * torch.log10(power)
    db -= ref_db
    db = torch.clamp(db, min=-range_db)
    return db

def A_weighting(frequencies, min_db=-80.0):
    # ported from librosa
    f_sq = np.asanyarray(frequencies) ** 2.0
    const = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights = 2.0 + 20.0 * (
        np.log10(const[0])
        + 2 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
        - 0.5 * np.log10(f_sq + const[3])
    )
    return weights if min_db is None else np.maximum(min_db, weights)

def fft_frequencies(*, sr=22050, n_fft=2048):
    # ported from librosa
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def compute_loudness(
    audio,
    sample_rate=16000,
    hop_length=512,
    n_fft=2048,
    range_db=DB_RANGE,
    ref_db=0.0,
    a_weighting=None,
    center=True,
):
    """Perceptual loudness in dB, relative to white noise, amplitude=1.

    Args:
        audio: tensor. Shape [batch_size, audio_length] or [audio_length].
        sample_rate: Audio sample rate in Hz.
        hop_length: FFT hop length
        n_fft: FFT window size.
        range_db: Sets the dynamic range of loudness in decibels. The minimum loudness (per a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by (A_weighting + 10 * log10(abs(stft(audio))**2.0).

    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    if is_1d:
        audio = audio[None, :]

    # Take STFT.
    s = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        center=center,
        window=torch.hann_window(n_fft, device=audio.device),
    )
    # batch, frequency_bins, n_frames
    s = s.permute(0, 2, 1)
    if a_weighting is None:
        frequencies = fft_frequencies(sr=sample_rate, n_fft=n_fft)
        a_weighting = A_weighting(frequencies+1e-8)
        a_weighting = torch.from_numpy(a_weighting.astype(np.float32)).to(audio.device)
    loudness = spec_loudness(s, a_weighting, range_db, ref_db)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    # n_secs = audio.shape[-1] / float(sample_rate)  # `n_secs` can have milliseconds
    # expected_len = int(n_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector
    # loudness = pad_or_trim_to_expected_length(loudness, expected_len, -range_db)
    return loudness


def loudness_loss(input_audio, target_audio, sr=16000):
    input_l = compute_loudness(input_audio, sr)
    target_l = compute_loudness(target_audio, sr)
    return F.l1_loss(torch.pow(10, input_l/10), torch.pow(10, target_l/10))
