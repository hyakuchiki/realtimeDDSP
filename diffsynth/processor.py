import numpy as np
import torch
import torch.nn as nn
import diffsynth.util as util
import math
from typing import Dict, List, Tuple

class Processor(nn.Module):
    def __init__(self, name):
        """ Initialize as module """
        super().__init__()
        self.name = name
        self.param_sizes: Dict[str, int] = {}
        self.param_range: Dict[str, Tuple(float, float)] = {}
        self.param_types: Dict[str, str] = {}
        self.is_gen = False

    def scale_params(self, params: Dict[str, torch.Tensor], scaled_params: List[str]):
        # scaling each parameter according to the property
        # input is 0~1
        scaled: Dict[str, torch.Tensor] = {}
        for k in params.keys():
            if k not in self.param_sizes:
                raise ValueError(f'Specified non-existent parameter {self.name}: {k}')
            scale_type = self.param_types[k]
            p_range = self.param_range[k]
            if scale_type == 'raw' or k in scaled_params:
                scaled[k] = params[k]
            elif scale_type == 'sigmoid':
                scaled[k] = params[k] * (p_range[1] - p_range[0]) + p_range[0]
            elif scale_type == 'freq_sigmoid':
                scaled[k] = util.unit_to_hz(params[k], p_range[0], p_range[1])
            elif scale_type == 'exp_sigmoid':
                scaled[k] = util.exp_scale(params[k], math.log(10.0), p_range[1], 1e-7+p_range[0])
            else:
                raise ValueError(f'Incorrect parameter scaling settings for {self.name}: {k}')
        return scaled
        
    def process(self, params: Dict[str, torch.Tensor], n_samples: int, scaled_params: List[str]):
        """automatically scale inputs and run thru processor

        Args:
            params (Dict): {[parameter name]: [parameter value], ...}
            scaled_params (list): list of names of parameters that do not need scaling

        Returns:
            dict: Final output of processor
        """
        params = self.scale_params(params, scaled_params)
        return self(params, n_samples)

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        raise NotImplementedError

class Add(Processor):
    def __init__(self, name='add'):
        super().__init__(name=name)
        self.param_sizes = {'signal_a': 1, 'signal_b': 1}
        self.param_range = {'signal_a': (-1.0, 1.0), 'signal_b': (-1.0, 1.0)}
        self.param_types = {'signal_a': 'raw', 'signal_b': 'raw'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        return params['signal_a']+params['signal_b']

class Mix(Processor):
    def __init__(self, name='mix'):
        super().__init__(name=name)
        self.param_sizes = {'signal_a': 1, 'signal_b': 1, 'mix_a': 1, 'mix_b': 1}
        self.param_range = {'signal_a': (-1.0, 1.0), 'signal_b': (-1.0, 1.0), 'mix_a': (0.0, 1.0), 'mix_b': (0.0, 1.0)}
        self.param_types = {'signal_a': 'raw', 'signal_b': 'raw', 'mix_a': 'sigmoid', 'mix_b': 'sigmoid'}

    def forward(self, params: Dict[str, torch.Tensor], n_samples: int):
        signal_a = params['signal_a']        
        signal_b = params['signal_b']
        mix_a = params['mix_a']
        mix_b = params['mix_b']
        if mix_a.ndim > 1:
            if mix_a.shape[1] != n_samples:
                mix_a = util.resample_frames(mix_a, n_samples)
            if mix_b.shape[1] != n_samples:
                mix_b = util.resample_frames(mix_b, n_samples)
            return mix_a[:, :, 0] * signal_a + mix_b[:, :, 0] * signal_b
        else:
            return mix_a * signal_a + mix_b * signal_b

class VCA(Processor):
    def __init__(self, name='vca'):
        super().__init__(name=name)
        self.param_desc = {
                'signal': {'size':1, 'range': (-1, 1), 'type': 'raw'}, 
                'amp': {'size':1, 'range': (0, 10), 'type': 'sigmoid'},
                }    

    def forward(self, signal, amp):
        # signal: batch, n_samples
        # amp: batch, n_frames, 1
        if amp.ndim > 1:
            n_samples = signal.shape[1]
            if amp.shape[1] != n_samples:
                amp = util.resample_frames(amp, n_samples)
            return amp[:, :, 0]*signal
        else:
            return amp*signal