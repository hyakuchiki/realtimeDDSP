import torch
import torch.nn.functional as F
import torch.nn as nn
from diffsynth.processor import Processor
import diffsynth.util as util
from diffsynth.f0 import yin_frame, FMIN, FMAX
from diffsynth.spectral import spec_loudness, A_weighting, fft_frequencies
from typing import Dict, Tuple
import numpy as np
from nnAudio.features import MFCC

class StatefulGRU(nn.Module):
    def __init__(self, gru):
        super().__init__()
        self.gru = gru
        self.hidden = None
    
    def forward(self, x):
        x, hidden = self.gru(x, self.hidden)
        self.hidden = hidden
        return x, hidden

class StreamHarmonic(Processor):
    def __init__(self, sample_rate:int=48000, normalize_below_nyquist:bool=True, name:str='harmonic', n_harmonics:int=256, freq_range:Tuple[float, float]=(FMIN, FMAX), batch_size:int=1):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.normalize_below_nyquist = normalize_below_nyquist
        self.n_harmonics = n_harmonics
        self.register_buffer('phase', torch.zeros(batch_size, 1, n_harmonics))
        self.register_buffer('prev_freqs', torch.ones(batch_size, 1, n_harmonics)*440)
        self.register_buffer('prev_harm', torch.zeros(batch_size, 1, n_harmonics))
        self.param_sizes = {'amplitudes': 1, 'harmonic_distribution': self.n_harmonics, 'f0_hz': 1}
        self.param_range = {'amplitudes': (0.0, 1.0), 'harmonic_distribution': (0.0, 1.0), 'f0_hz': freq_range}
        self.param_types = {'amplitudes': 'exp_sigmoid', 'harmonic_distribution': 'exp_sigmoid', 'f0_hz': 'freq_sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """Synthesize audio with additive synthesizer from controls.

        Args:
        amplitudes: Amplitude tensor of shape [batch, n_frames, 1].
        harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
        f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
            n_frames, 1].

        Returns:
        signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        amplitudes = params['amplitudes']
        harmonic_distribution = params['harmonic_distribution']
        f0_hz = params['f0_hz']
        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            harmonic_frequencies = util.get_harmonic_frequencies(f0_hz, self.n_harmonics)
            harmonic_distribution = util.remove_above_nyquist(harmonic_frequencies, harmonic_distribution, self.sample_rate)

        # Normalize
        harmonic_distribution /= torch.sum(harmonic_distribution, dim=-1, keepdim=True)

        harmonic_amplitudes = amplitudes * harmonic_distribution
        # interpolate with previous params
        harmonic_frequencies = util.get_harmonic_frequencies(f0_hz, self.n_harmonics)
        harmonic_freqs = torch.cat([self.prev_freqs, harmonic_frequencies], dim=1)
        frequency_envelopes = util.resample_frames(harmonic_freqs, n_samples) 
        harmonic_amps = torch.cat([self.prev_harm, harmonic_amplitudes], dim=1)
        amplitude_envelopes = util.resample_frames(harmonic_amps, n_samples)
        audio, last_phase = util.oscillator_bank_stream(frequency_envelopes, amplitude_envelopes, sample_rate=self.sample_rate, init_phase=self.phase)
        self.phase = last_phase
        self.prev_harm = harmonic_amplitudes[:, -1:]
        self.prev_freqs = harmonic_frequencies[:, -1:]
        return audio

class StreamFilteredNoise(Processor):
    def __init__(self, filter_size=257, name='noise', amplitude=1.0, batch_size=1):
        super().__init__(name=name)
        self.filter_size = int(filter_size)
        self.amplitude = amplitude
        self.register_buffer('cache', torch.zeros(batch_size, 1))
        self.param_sizes = {'freq_response': self.filter_size // 2 + 1}
        self.param_range = {'freq_response': (0.0, 2.0)}
        self.param_types = {'freq_response': 'exp_sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """generate Gaussian white noise through FIRfilter
        Args:
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        freq_response = params['freq_response']
        batch_size = freq_response.shape[0]
        # noise
        audio = (torch.rand(batch_size, n_samples) * 2.0 - 1.0).to(
            freq_response.device
        ) * self.amplitude
        filtered = util.fir_filter(
            audio, freq_response, self.filter_size, padding="same"
        )
        output = filtered[..., :n_samples]
        cache = F.pad(self.cache, pad=(0, n_samples-self.cache.shape[-1]))
        output = output + cache
        self.cache = filtered[..., n_samples:]
        return output

class StreamIRReverb(Processor):
    def __init__(self, ir, name='noise', batch_size=1):
        super().__init__(name=name)
        self.register_buffer('ir', torch.cat([torch.zeros(1), ir], dim=0)[None, :].expand(batch_size, -1))
        self.register_buffer('buffer', torch.zeros(batch_size, 16000))
        self.param_sizes = {'audio': 1, 'mix': 1}
        self.param_range = {'audio': (-1.0, 1.0), 'mix': (0.0, 1.0)}
        self.param_types = {'audio': 'raw', 'mix': 'sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """generate Gaussian white noise through FIRfilter
        Args:
            freq_response (torch.Tensor): frequency response (only magnitude) [batch, n_frames, filter_size // 2 + 1]

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        audio = params['audio']
        mix = params['mix']
        audio_len = audio.shape[-1]
        wet = util.fft_convolve(audio, self.ir, padding='valid', delay_compensation=0)
        # add output and buffer
        length = max(wet.shape[-1], self.buffer.shape[-1])
        buffer_wet = F.pad(self.buffer, (0, max(0, length-self.buffer.shape[-1]))) \
                        + F.pad(wet, (0, max(0, length-wet.shape[-1])))

        self.buffer = buffer_wet[..., audio_len:]
        out_wet = buffer_wet[..., :audio_len]
        output = mix*2*out_wet + (1-mix)*2*audio
        return output

def replace_modules(module):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            replace_modules(m)
        if isinstance(m, torch.nn.GRU):
            str_m = StatefulGRU(m)
            setattr(module, name, str_m)


class Spec2Mfcc(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_mfcc,
        hop_length,
        window_size,
    ):
        super().__init__()
        # load default mfcc
        mfcc = MFCC(
            sample_rate,
            n_mfcc,
            # center=True,
            hop_length=hop_length,
            n_fft=window_size,
            verbose=False,
        )
        self.top_db = mfcc.top_db
        self.amin = mfcc.amin
        self.ref = mfcc.ref
        self.n_mfcc = mfcc.n_mfcc
        self.mel_basis = mfcc.melspec_layer.mel_basis

    def _power_to_db(self, S):
        log_spec = 10.0 * torch.log10(torch.max(S, self.amin))
        log_spec -= 10.0 * torch.log10(torch.max(self.amin, self.ref))
        # make the dim same as log_spec so that it can be broadcasted
        batch_wise_max = log_spec.flatten(1).max(1)[0].unsqueeze(1).unsqueeze(1)
        log_spec = torch.max(log_spec, batch_wise_max - self.top_db)
        return log_spec

    def _dct(self, x):
        """
        Refer to https://github.com/zh217/torch-dct for the original implmentation.
        """
        x = x.permute(
            0, 2, 1
        )  # make freq the last axis, since dct applies to the frequency axis
        x_shape = x.shape
        N = x_shape[-1]

        v = torch.cat([x[:, :, ::2], x[:, :, 1::2].flip([2])], dim=2)
        Vc = torch.view_as_real(torch.fft.fft(v))

        # TODO: Can make the W_r and W_i trainable here
        k = (
            -torch.arange(N, dtype=x.dtype, device=x.device)[None, :]
            * torch.pi
            / (2 * N)
        )
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, :, 0] * W_r - Vc[:, :, :, 1] * W_i

        # if norm == "ortho":
        V[:, :, 0] /= torch.sqrt(torch.tensor([N])).item() * 2
        V[:, :, 1:] /= torch.sqrt(torch.tensor([N]) / 2).item() * 2

        V = 2 * V
        return V.permute(0, 2, 1)

    def forward(self, spec):
        power_spec = spec.real**2 + spec.imag**2
        melspec = torch.matmul(self.mel_basis, power_spec)
        melspec = self._power_to_db(melspec)
        mfcc = self._dct(melspec)[:, : self.n_mfcc, :]
        return mfcc


class CachedStreamEstimatorFLSynth(nn.Module):
    # harmonics plus noise model
    def __init__(
        self,
        estimator,
        synth,
        sample_rate,
        hop_size=512,
        pitch_min=50.0,
        pitch_max=2000.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate       
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        # caching
        self.offset = hop_size
        self.hop_size = hop_size
        self.window_size = 2048
        self.input_cache = torch.zeros(1, self.window_size - self.offset)
        self.output_cache = torch.zeros(1, self.hop_size)
        # loudness
        frequencies = fft_frequencies(sr=sample_rate, n_fft=self.window_size)
        a_weighting = A_weighting(frequencies+1e-8)
        self.register_buffer(
            "a_weighting",
            torch.from_numpy(a_weighting.astype(np.float32)),
            persistent=False,
        )
        self.prev_f0 = torch.ones(1)*440
        # Estimator
        self.estimator = estimator
        # Synth
        self.synth = synth
        self.register_buffer(
            "window", torch.hann_window(self.window_size), persistent=False
        )
        self.spec2mfcc = Spec2Mfcc(
            sample_rate=sample_rate,
            n_mfcc=20,
            hop_length=hop_size,
            window_size=self.window_size,
        )

    def forward(self, audio: torch.Tensor, f0_mult: torch.Tensor, param: Dict[str, torch.Tensor]):
        # audio (batch=1, n_samples=L)
        with torch.no_grad():
            orig_len = audio.shape[-1]
            # input cache
            audio = torch.cat([self.input_cache.to(audio.device), audio], dim=-1)
            # doesn't do anything
            windows = util.slice_windows(audio, self.window_size, self.hop_size, pad=False)
            windows = self.window[None, None, :] * windows

            self.offset = self.hop_size - ((orig_len - self.offset) % self.hop_size)
            self.input_cache = audio[:, -(self.window_size - self.offset):]

            f0 = yin_frame(windows, self.sample_rate, self.pitch_min, self.pitch_max)
            # loudness
            comp_spec = torch.fft.rfft(windows, dim=-1)
            mfcc = self.spec2mfcc(comp_spec.permute(0, 2, 1))
            loudness = spec_loudness(comp_spec, self.a_weighting)

            if f0[:, 0] == 0:
                # use previous f0 if noisy
                f0[:, 0] = self.prev_f0
                # also assume silent if noisy
                # loudness[:, 0] = 0
            for i in range(1, f0.shape[1]):
                if f0[:, i] == 0:
                    f0[:, i] = f0[:, i-1]
                    # loudness[:, i] = 0

            self.prev_f0 = f0[:, -1]
            # estimator
            f0 = f0_mult * f0
            x = {
                "f0": f0[:, :, None],
                "loud": loudness[:, :, None],
                "audio": audio,
                "mfcc": mfcc,
            }  # batch=1, n_frames=windows.shape[1], 1
            x.update(param)
            est_param = self.estimator(x)
            params_dict = self.synth.fill_params(est_param, x)
            render_length = windows.shape[1]*self.hop_size # last_of_prev_frame<->0th window<-...->last window
            resyn_audio, outputs = self.synth(params_dict, render_length)
            # output cache (delay)
            resyn_audio = torch.cat([self.output_cache.to(audio.device), resyn_audio], dim=-1)
            if resyn_audio.shape[-1] > orig_len:
                self.output_cache = resyn_audio[:, orig_len:]
                resyn_audio = resyn_audio[:, :orig_len]
            return resyn_audio
