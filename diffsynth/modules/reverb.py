import torch
import torch.nn as nn
from diffsynth.processor import Processor
import diffsynth.util as util
import numpy as np

# input_audio decay gain
# make ir 
# set ir_0=0 (cut dry signal)

class DecayReverb(Processor):
    """
    Reverb with exponential decay
    1. Make IR based on decay and gain
        - Exponentially decaying white noise -> avoids flanging?
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
        self.param_desc = {
                'audio':    {'size': 1, 'range': (-1, 1),       'type': 'raw'},
                'gain':     {'size': 1, 'range': (0, 0.25),      'type': 'exp_sigmoid'}, 
                'decay':    {'size': 1, 'range': (10.0, 25.0),    'type': 'sigmoid'}, 
                }

    def forward(self, audio, gain, decay):
        """
        gain: gain of reverb ir (batch, 1, 1)
        decay: decay rate - larger the faster (batch, 1, 1)
        """
        gain = gain.squeeze(1)
        decay = decay.squeeze(1)
        ir = gain * torch.exp(-decay * self.time) * self.noise # batch, time
        wet = util.fft_convolve(audio, ir, padding='same', delay_compensation=0)
        return audio+wet