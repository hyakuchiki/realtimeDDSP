import torch
import torch.nn as nn
from diffsynth.processor import Processor
import diffsynth.util as util
import numpy as np
from typing import Dict

class IRReverb(Processor):
    """
    Learns an IR as model parameter (always applied)
    """
    def __init__(self, name='ir', ir_s=1.0, sr=48000):
        super().__init__(name)
        ir_length = int(ir_s*sr)
        noise = torch.rand(1, ir_length)*2-1 # [-1, 1)
        # initial value should be zero to mask dry signal
        self.register_buffer('zero', torch.zeros(1))
        time = torch.linspace(0.0, 1.0, ir_length-1)
        # initial ir = decaying white noise
        self.ir = nn.Parameter((torch.rand(ir_length-1)-0.5)*0.1 * torch.exp(-5.0 * time), requires_grad=True)
        self.ir_length = ir_length
        self.param_sizes = {'audio': 1}
        self.param_range = {'audio': (-1.0, 1.0)}
        self.param_types = {'audio': 'raw'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """
        audio: input audio (batch, n_samples)
        """
        audio = params['audio']
        ir = torch.cat([self.zero, self.ir], dim=0)[None, :].expand(audio.shape[0], -1)
        wet = util.fft_convolve(audio, ir, padding='same', delay_compensation=0)
        return audio+wet

class IRReverbMix(IRReverb):
    """
    Learns an IR as model parameter (always applied)
    """
    def __init__(self, name='ir', ir_s=1.0, sr=48000):
        super().__init__(name, ir_s, sr)
        self.param_sizes = {'audio': 1, 'mix': 1}
        self.param_range = {'audio': (-1.0, 1.0), 'mix': (0.0, 1.0)}
        self.param_types = {'audio': 'raw', 'mix': 'sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """
        audio: input audio (batch, n_samples)
        """
        audio = params['audio']
        mix = params['mix']
        ir = torch.cat([self.zero, self.ir], dim=0)[None, :].expand(audio.shape[0], -1)
        wet = util.fft_convolve(audio, ir, padding='same', delay_compensation=0)
        return (1-mix)*2*audio + mix*2*wet

class DecayReverb(Processor):
    """
    Reverb with exponential decay
    1. Make IR based on decay and gain
        - Exponentially decaying white noise
    2. convolve with IR
    3. Cut tail
    """

    def __init__(self, name='reverb', ir_length=16000):
        super().__init__(name)
        noise = torch.rand(1, ir_length)*2-1 # [-1, 1)
        noise[:, 0] = 0.0 # initial value should be zero to mask dry signal
        self.register_buffer('noise', noise)
        time = torch.linspace(0.0, 1.0, ir_length)[None, :]
        self.register_buffer('time', time)
        self.ir_length = ir_length
        self.param_sizes = {'audio': 1, 'gain': 1, 'decay': 1}
        self.param_range = {'audio': (-1.0, 1.0), 'gain': (0.0, 0.25), 'decay': (10.0, 25.0)}
        self.param_types = {'audio': 'raw', 'gain': 'exp_sigmoid', 'decay': 'sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        """
        gain: gain of reverb ir (batch, 1, 1)
        decay: decay rate - larger the faster (batch, 1, 1)
        """
        audio = params['audio']
        gain = params['gain']
        decay = params['decay']
        gain = gain.squeeze(1)
        decay = decay.squeeze(1)
        ir = gain * torch.exp(-decay * self.time) * self.noise # batch, time
        wet = util.fft_convolve(audio, ir, padding='same', delay_compensation=0)
        return audio+wet