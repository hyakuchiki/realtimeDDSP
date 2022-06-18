import torch
import torch.nn as nn
from diffsynth.f0 import FMIN, FMAX
from diffsynth.processor import Processor
import diffsynth.util as util
import numpy as np
from typing import Dict

class Harmonic(Processor):
    """Synthesize audio with a bank of harmonic sinusoidal oscillators.
    code mostly borrowed from DDSP"""

    def __init__(self, sample_rate=16000, normalize_below_nyquist=True, name='harmonic', n_harmonics=256, freq_range=(FMIN, FMAX)):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.normalize_below_nyquist = normalize_below_nyquist
        self.n_harmonics = n_harmonics
        self.freq_range = freq_range
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
        if len(f0_hz.shape) < 3: # when given as a condition
            f0_hz = f0_hz[:, :, None]
        # Bandlimit the harmonic distribution.
        if self.normalize_below_nyquist:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = util.get_harmonic_frequencies(f0_hz, n_harmonics)
            harmonic_distribution = util.remove_above_nyquist(harmonic_frequencies, harmonic_distribution, self.sample_rate)

        # Normalize
        harmonic_distribution /= torch.sum(harmonic_distribution, axis=-1, keepdim=True)

        signal = util.harmonic_synthesis(frequencies=f0_hz, amplitudes=amplitudes, harmonic_distribution=harmonic_distribution, n_samples=n_samples, sample_rate=self.sample_rate)
        return signal

class FilteredNoise(Processor):
    """
    uses frequency sampling
    """
    
    def __init__(self, filter_size=257, name='noise', amplitude=1.0):
        super().__init__(name=name)
        self.filter_size = filter_size
        self.amplitude = amplitude
        self.param_sizes = {'freq_response': self.filter_size // 2 + 1}
        self.param_range = {'freq_response': (0, 2.0)}
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

        audio = (torch.rand(batch_size, n_samples)*2.0-1.0).to(freq_response.device) * self.amplitude
        filtered = util.fir_filter(audio, freq_response, self.filter_size)
        return filtered

class SawOscillator(Processor):
    """
    Synthesize audio from a saw oscillator
    """

    def __init__(self, sample_rate=16000, n_harms=512, name='saw', freq_range=(FMIN, FMAX)):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        # saw waveform
        # waveform = torch.roll(torch.linspace(1.0, -1.0, 64), 32) # aliasing?
        # Antialias in case of f=440Hz
        n_harms = min(int(sample_rate/440), n_harms)
        k = torch.arange(1, n_harms+1)
        harm_dist = (2 / np.pi * (1/k)).unsqueeze(-1) #(n_harms, 1)
        frequencies = torch.linspace(1.0, float(n_harms), n_harms).unsqueeze(1) #(n_harms, 1)
        phase = torch.linspace(0.0, np.pi*2, 512).unsqueeze(0) # (1, timestep)
        saw = torch.sum(harm_dist * torch.sin(frequencies * phase), dim=0)
        self.register_buffer('waveform', saw)
        self.param_sizes = {'amplitudes': 1, 'f0_hz': 1}
        self.param_range = {'amplitudes': (0.0, 1.0), 'f0_hz': freq_range}
        self.param_types = {'amplitudes': 'exp_sigmoid', 'f0_hz': 'freq_sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """forward pass of saw oscillator

        Args:
            amplitudes: (batch_size, n_frames, 1)
            f0_hz: frequency of oscillator at each frame (batch_size, n_frames, 1)

        Returns:
            signal: synthesized signal ([batch_size, n_samples])
        """
        amplitudes = params['amplitudes']
        f0_hz = params['f0_hz']
        signal = util.wavetable_synthesis(f0_hz, amplitudes, self.waveform, n_samples, self.sample_rate)
        return signal

class SineOscillator(Processor):
    """Synthesize audio from a sine oscillator
    """

    def __init__(self, sample_rate=16000, name='sin', freq_range=(FMIN, FMAX)):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.param_sizes = {'amplitudes': 1, 'f0_hz': 1}
        self.param_range = {'amplitudes': (0.0, 1.0), 'f0_hz': freq_range}
        self.param_types = {'amplitudes': 'exp_sigmoid', 'f0_hz': 'freq_sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """forward pass of sine oscillator

        Args:
            amplitudes: (batch_size, n_frames, 1)
            f0_hz: frequency of oscillator at each frame (batch_size, n_frames, 1)

        Returns:
            signal: synthesized signal ([batch_size, n_samples])
        """
        amplitudes = params['amplitudes']
        f0_hz = params['f0_hz']
        signal = util.sin_synthesis(f0_hz, amplitudes, n_samples, self.sample_rate)
        return signal