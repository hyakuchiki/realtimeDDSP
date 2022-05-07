from librosa.core import fft
import torch
import torch.nn as nn
from diffsynth.spectral import multiscale_fft, compute_loudness, spectrogram
import librosa
from diffsynth.util import log_eps
import torch.nn.functional as F
import functools

def multispectrogram_loss(x_audio, target_audio, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_ls=None, windows=None, log_mag_w=1.0, mag_w=1.0, fro_w=0.0, reduce='mean'):
    x_specs = multiscale_fft(x_audio, fft_sizes, hop_ls, windows)
    target_specs = multiscale_fft(target_audio, fft_sizes, hop_ls, windows)
    spec_loss = {}
    log_spec_loss = {}
    fro_spec_loss = {}
    if reduce == 'mean':
        reduce_dims = (0,1,2)
    else:
        # do not reduce batch
        reduce_dims = (1,2)
    for n_fft, x_spec, target_spec in zip(fft_sizes, x_specs, target_specs):
        if mag_w > 0:
            spec_loss[n_fft] = mag_w * torch.mean(torch.abs(x_spec - target_spec), dim=reduce_dims)
        if log_mag_w > 0:
            log_spec_loss[n_fft] = log_mag_w * torch.mean(torch.abs(log_eps(x_spec) - log_eps(target_spec)), dim=reduce_dims)
        if fro_w > 0: # spectral convergence
            fro_loss = torch.linalg.norm(x_spec - target_spec, 'fro', dim=(1,2)) / torch.linalg.norm(target_spec, 'fro', dim=(1,2))
            fro_spec_loss[n_fft] = torch.mean(fro_loss) if reduce=='mean' else fro_loss
    return {'spec':spec_loss, 'logspec':log_spec_loss, 'fro': fro_spec_loss}

class SpectralLoss(nn.Module):
    """
    loss for reconstruction with multiscale spectrogram loss and waveform loss
    """
    def __init__(self, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_lengths=None, win_lengths=None, mag_w=1.0, log_mag_w=1.0, fro_w=0.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.mag_w = mag_w
        self.log_mag_w = log_mag_w
        self.fro_w = fro_w
        if win_lengths is not None:
            for wl in win_lengths:
                self.register_buffer(f'window_{wl}', torch.hann_window(wl))
        self.spec_loss = functools.partial(multispectrogram_loss, fft_sizes=fft_sizes, hop_ls=hop_lengths, log_mag_w=log_mag_w, mag_w=mag_w, fro_w=fro_w)
        
    def __call__(self, output_dict, target_dict, sum_losses=True):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        if self.win_lengths is not None:
            windows = [getattr(self, f'window_{wl}') for wl in self.win_lengths]
        else:
            windows = None
        spec_losses = self.spec_loss(x_audio, target_audio, windows=windows)
        if sum_losses:
            multi_spec_loss = sum(spec_losses['spec'].values()) + sum(spec_losses['logspec'].values()) + sum(spec_losses['fro'].values())
            multi_spec_loss /= (len(self.fft_sizes)*(self.mag_w + self.log_mag_w + self.fro_w))
            return multi_spec_loss
        else:
            multi_spec_losses = {k: sum(v.values()) for k, v in spec_losses.items()}
            return multi_spec_losses

class LoudnessLoss(nn.Module):
    def __init__(self, fft_size=1024, sr=16000, frame_rate=50, db=False) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.sr = sr
        self.frame_rate = frame_rate
        self.db = db
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=fft_size)
        a_weighting = librosa.A_weighting(frequencies)[None, :, None]
        self.register_buffer('a_weighting', torch.from_numpy(a_weighting).float())

    def __call__(self, output_dict, target_dict):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        x_loud = compute_loudness(x_audio, self.sr, self.frame_rate, a_weighting=self.a_weighting)
        target_loud = compute_loudness(target_audio, self.sr, self.frame_rate, a_weighting=self.a_weighting)
        l1_loss = F.l1_loss(torch.pow(10, x_loud/10), torch.pow(10, target_loud/10))
        if self.db:
            l1_loss = 10 * torch.log10(l1_loss)
        return l1_loss