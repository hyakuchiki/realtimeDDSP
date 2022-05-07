import numpy as np
import torch
import torch.nn as nn
import diffsynth.util as util
import math

SCALE_FNS = {
    'raw': lambda x, low, high: x,
    'sigmoid': lambda x, low, high: x*(high-low) + low,
    'freq_sigmoid': lambda x, low, high: util.unit_to_hz(x, low, high, clip=False),
    'exp_sigmoid': lambda x, low, high: util.exp_scale(x, math.log(10.0), high, 1e-7+low),
}

WIDE_RANGE = (32.5, 1980) # C1=24(32.7)~B6=95(1975.5) ok for crepe

class Processor(nn.Module):
    def __init__(self, name):
        """ Initialize as module """
        super().__init__()
        self.name = name
        self.param_desc = {}

    def process(self, scaled_params=[], **kwargs):
        # scaling each parameter according to the property
        # input is 0~1
        for k in kwargs.keys():
            if k not in self.param_desc or k in scaled_params:
                continue
            desc = self.param_desc[k]
            scale_fn = SCALE_FNS[desc['type']]
            p_range = desc['range']
            # if (kwargs[k] > 1).any():
            #     raise ValueError('parameter to be scaled is not 0~1')
            kwargs[k] = scale_fn(kwargs[k], p_range[0], p_range[1])
        return self(**kwargs)

    def forward(self):
        raise NotImplementedError

class Gen(Processor):
    def __init__(self, name):
        super().__init__(name)

class Add(Processor):
    def __init__(self, name='add'):
        super().__init__(name=name)
        self.param_desc = {
            'signal_a': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 'signal_b': {'size':1, 'range': (-1, 1), 'type': 'raw'}
            }

    def forward(self, signal_a, signal_b):
        # kinda sucks can only add two
        return signal_a+signal_b

class Mix(Processor):
    def __init__(self, name='mix'):
        super().__init__(name=name)
        self.param_desc = {
                'signal_a': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 
                'signal_b': {'size':1, 'range': (-1, 1), 'type': 'raw'},
                'mix_a': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                'mix_b': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                }

    def forward(self, signal_a, signal_b, mix_a, mix_b):
        n_samples = signal_a.shape[1]
        if mix_a.shape[1] != n_samples:
            mix_a = util.resample_frames(mix_a, n_samples)
        if mix_b.shape[1] != n_samples:
            mix_b = util.resample_frames(mix_b, n_samples)
        return mix_a[:, :, 0] * signal_a + mix_b[:, :, 0] * signal_b

class VCA(Processor):
    def __init__(self, name='vca'):
        super().__init__(name=name)
        self.param_desc = {
                'signal': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 
                'amp': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                }    

    def forward(self, signal, amp):
        # signal: batch, n_samples
        # amp: batch, n_frames, 1
        n_samples = signal.shape[1]
        if amp.shape[1] != n_samples:
            amp = util.resample_frames(amp, n_samples)
        return amp[:, :, 0]*signal

class Attenuate(Processor):
    def __init__(self, name='att'):
        super().__init__(name=name)
        self.param_desc = {
                'signal': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 
                'attenuate': {'size':1, 'range': (0, 1), 'type': 'sigmoid'},
                }    

    def forward(self, signal, attenuate):
        # signal: batch, n_samples
        # amp: batch, n_frames, 1
        n_samples = signal.shape[1]
        if attenuate.shape[1] != n_samples:
            attenuate = util.resample_frames(attenuate, n_samples)
        return (1 - attenuate[:, :, 0])*signal

class Diff(Processor):
    def __init__(self, name='diff'):
        super().__init__(name=name)
        self.param_desc = {
                'signal': {'size':1, 'range': (-np.inf, np.inf), 'type': 'raw'}, 
                }    

    def forward(self, signal):
        # signal: batch, n_frames, n_dim
        first_frame = torch.zeros_like(signal)[:, 0:1, :]
        return torch.diff(signal, n=1, dim=1, prepend=first_frame)