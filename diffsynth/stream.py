import torch
import torch.nn.functional as F
import torch.nn as nn
import hydra
from diffsynth.modules.generators import FilteredNoise, Harmonic
from diffsynth.processor import Processor
from diffsynth.synthesizer import Synthesizer
import diffsynth.util as util
from diffsynth.f0 import yin_frame, FMIN, FMAX
from diffsynth.spectral import spec_loudness
import librosa
from typing import Dict
import numpy as np

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
    def __init__(self, sample_rate=48000, normalize_below_nyquist=True, name='harmonic', n_harmonics=256, freq_range=(FMIN, FMAX), batch_size=1):
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
        amplitudes: Amplitude tensor of shape [batch, n_frames=1, 1].
        harmonic_distribution: Tensor of shape [batch, n_frames=1, n_harmonics].
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
        self.prev_harm = harmonic_amplitudes
        self.prev_freqs = harmonic_frequencies
        return audio

class StreamFilteredNoise(Processor):
    def __init__(self, filter_size=257, scale_fn=util.exp_sigmoid, name='noise', initial_bias=-5.0, amplitude=1.0, batch_size=1):
        super().__init__(name=name)
        self.filter_size = int(filter_size)
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias
        self.amplitude = amplitude
        self.register_buffer('cache', torch.zeros(batch_size, 1))
        self.param_sizes = {'freq_response': self.filter_size // 2 + 1}
        self.param_range = {'freq_response': (1e-7, 2.0)}
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
        freq_response = self.scale_fn(freq_response + self.initial_bias)
        # noise
        audio = (torch.rand(batch_size, n_samples)*2.0-1.0).to(freq_response.device) * self.amplitude
        
        filtered = util.fir_filter(audio, freq_response, self.filter_size, padding='valid')
        output = filtered[..., :n_samples]
        cache = F.pad(self.cache, pad=(0, n_samples-self.cache.shape[-1]))
        output = output + cache
        self.cache = filtered[..., n_samples:]
        return output

def replace_modules(module):
    for name, m in module.named_children():
        if len(list(m.children())) > 0:
            replace_modules(m)
        if isinstance(m, torch.nn.GRU):
            str_m = StatefulGRU(m)
            setattr(module, name, str_m)

def construct_streaming_synth_from_conf(synth_conf):
    dag = []
    for module_name, v in synth_conf.dag.items():
        module = hydra.utils.instantiate(v.config, name=module_name)
        conn = v.connections
        if isinstance(module, Harmonic):
            module = StreamHarmonic(module.sample_rate, module.normalize_below_nyquist, module.name, module.n_harmonics, module.freq_range)
        if isinstance(module, FilteredNoise):
            module = StreamFilteredNoise(module.filter_size, module.scale_fn, module.name, module.initial_bias, module.amplitude)
        dag.append((module, conn))
    synth = Synthesizer(dag, conditioned=synth_conf.conditioned)
    return synth

class StreamEstimatorFLSynth(nn.Module):
    # harmonics plus noise model
    def __init__(self, estimator, synth_cfg, sample_rate, pitch_min=50.0, pitch_max=2000.0):
        super().__init__()
        self.sample_rate = sample_rate       
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        # loudness
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        a_weighting = librosa.A_weighting(frequencies)
        self.register_buffer('a_weighting', torch.from_numpy(a_weighting.astype(np.float32)))
        # Estimator
        self.estimator = estimator
        # Synth
        self.synth = construct_streaming_synth_from_conf(synth_cfg)

    def forward(self, audio: torch.Tensor):
        # audio (batch=1, n_samples=2048)
        # audio should be 2048 samples
        with torch.no_grad():
            audio_len = audio.shape[-1]
            assert audio_len == 2048
            # f0
            f0 = yin_frame(audio, self.sample_rate, self.pitch_min, self.pitch_max)[:, None]
            # loudness
            comp_spec = torch.fft.rfft(audio, dim=-1)
            loudness = spec_loudness(comp_spec, self.a_weighting)[:, None]
            # estimator
            x = {'f0': f0, 'loud': loudness} # batch=1, n_frames=1
            est_param = self.estimator(x)
            params_dict = self.synth.fill_params(est_param, x)
            resyn_audio, outputs = self.synth(params_dict, audio_len)
            return resyn_audio

class InputCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # caching
        self.offset = 0
        self.hop_size = hop_size # 960@48kHz = 50Hz
        self.window_size = 2048
        self.input_cache = torch.zeros(1, self.window_size)

class CachedStreamEstimatorFLSynth(nn.Module):
    # harmonics plus noise model
    def __init__(self, estimator, synth_cfg, sample_rate, hop_size=960, pitch_min=50.0, pitch_max=2000.0):
        super().__init__()
        self.sample_rate = sample_rate       
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        # caching
        self.offset = 0
        self.hop_size = hop_size # 960@48kHz = 50Hz
        self.window_size = 2048
        self.input_cache = torch.zeros(1, self.window_size)
        self.output_cache = torch.zeros(1, self.hop_size)
        # loudness
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.window_size)
        a_weighting = librosa.A_weighting(frequencies)
        self.register_buffer('a_weighting', torch.from_numpy(a_weighting.astype(np.float32)))
        # Estimator
        self.estimator = estimator
        # Synth
        self.synth = construct_streaming_synth_from_conf(synth_cfg)

    def forward(self, audio: torch.Tensor):
        # audio (batch=1, n_samples=L)
        with torch.no_grad():
            orig_len = audio.shape[-1]
            # input cache
            audio = torch.cat([self.input_cache.to(audio.device), audio], dim=-1)
            windows = util.slice_windows(audio, self.window_size, self.hop_size)
            assert windows.shape[1] == (audio.shape[-1] - self.offset) // self.hop_size + 1

            self.offset = self.hop_size - (audio.shape[-1] - self.offset) % self.hop_size
            self.input_cache = audio[-(self.hop_size - self.offset):]

            # f0
            f0 = yin_frame(windows, self.sample_rate, self.pitch_min, self.pitch_max)
            # loudness
            comp_spec = torch.fft.rfft(windows, dim=-1)
            loudness = spec_loudness(comp_spec, self.a_weighting)
            print(f0.shape, loudness.shape)
            # estimator
            x = {'f0': f0[:,:,None], 'loud': loudness[:,:,None]} # batch=1, n_frames=windows.shape[1], 2
            est_param = self.estimator(x)
            params_dict = self.synth.fill_params(est_param, x)
            render_length = (windows.shape[1]+1)*self.hop_size # accounting for previous frame
            resyn_audio, outputs = self.synth(params_dict, render_length)
            # output cache (delay)
            resyn_audio = torch.cat([self.output_cache.to(audio.device), resyn_audio], dim=-1)
            if resyn_audio.shape[-1] > orig_len:
                resyn_audio = resyn_audio[:, :orig_len]
                self.output_cache = resyn_audio[orig_len:]
            return resyn_audio