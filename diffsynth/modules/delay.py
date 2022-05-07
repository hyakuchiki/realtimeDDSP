import torch
import torch.nn as nn
from diffsynth.processor import Processor
import diffsynth.util as util
import numpy as np
import math

class ModulatedDelay(Processor):
    """
    Use with LFO to create flanger/vibrato/chorus
    """
    def __init__(self, name='chorus', sample_rate=16000):
        super().__init__(name)
        self.sr = sample_rate
        self.param_desc = {
            'audio':    {'size': 1,     'range': (-1, 1),       'type': 'raw'},
            'delay_ms': {'size': 1,     'range': (1, 10.0),     'type': 'sigmoid'}, #ms
            'phase':    {'size': 1,     'range': (-1.0, 1.0),   'type': 'sigmoid'},
            'depth':    {'size': 1,     'range': (0, 0.25),     'type': 'sigmoid'},
            'mix':      {'size': 1,     'range': (0, 1.0),      'type': 'sigmoid'}
            }

    def forward(self, audio, delay_ms, phase, depth, mix):
        """pass audio through chorus/flanger
        Args:
            audio (torch.Tensor): [batch_size, n_samples]
            # static parameters
            delay_ms (torch.Tensor): Average delay in ms [batch_size, 1, 1]
            phase (torch.Tensor): -1->delay_ms*(1-depth), 1->delay_ms*(1+depth) [batch_size, n_frames, 1]
            depth (torch.Tensor): Lfo depth relative to delay_ms (0~1) [batch_size, 1, 1]
            mix (torch.Tensor): wet/dry ratio (0: all dry, 1: all wet) [batch_size, n_samples or 1, 1]

        Returns:
            [torch.Tensor]: Mixed audio. Shape [batch, n_samples]
        """
        # delay: delay_ms*(1-depth) <-> delay_ms*(1+depth)

        delay_ms = delay_ms.squeeze(-1)
        depth = depth.squeeze(-1)
        mix = mix.squeeze(-1)
        phase = util.resample_frames(phase, audio.shape[1])
        phase = phase.squeeze(-1)

        max_delay = self.param_desc['delay_ms']['range'][1] * 2 / 1000.0 * self.sr # samples
        delay_center = delay_ms / 1000.0 * self.sr # samples
        delay_value = phase * (depth * delay_center) + delay_center
        delay_phase = delay_value / max_delay # 0-> no delay 1: max_delay
        delayed = util.variable_delay(delay_phase, audio, buf_size=math.ceil(max_delay))
        return mix * delayed + (1-mix)*audio

class ModulatedDelayDelta(Processor):
    """
    Specify the delta of phase not the phase itself
    input can be LFO and the output should be same sin'(x) = cos(x)
    """
    def __init__(self, name='chorus', sample_rate=16000):
        super().__init__(name)
        self.sr = sample_rate
        self.param_desc = {
            'audio':    {'size': 1,     'range': (-1, 1),       'type': 'raw'},
            'delay_ms': {'size': 1,     'range': (1, 10.0),     'type': 'sigmoid'}, #ms
            'phase_delta':    {'size': 1,     'range': (-1.0, 1.0),   'type': 'sigmoid'},
            'depth':    {'size': 1,     'range': (0, 0.25),     'type': 'sigmoid'},
            'mix':      {'size': 1,     'range': (0, 1.0),      'type': 'sigmoid'}
            }

    def forward(self, audio, delay_ms, phase_delta, depth, mix):
        """pass audio through chorus/flanger
        Args:
            audio (torch.Tensor): [batch_size, n_samples]
            # static parameters
            delay_ms (torch.Tensor): Average delay in ms [batch_size, 1, 1]
            phase_delta (torch.Tensor): how much the phase changes from the previous frame [batch_size, n_frames, 1]
            depth (torch.Tensor): Lfo depth relative to delay_ms (0~1) [batch_size, 1, 1]
            mix (torch.Tensor): wet/dry ratio (0: all dry, 1: all wet) [batch_size, n_samples or 1, 1]

        Returns:
            [torch.Tensor]: Mixed audio. Shape [batch, n_samples]
        """
        # delay: delay_ms*(1-depth) <-> delay_ms*(1+depth)

        delay_ms = delay_ms.squeeze(-1)
        depth = depth.squeeze(-1)
        mix = mix.squeeze(-1)
        phase = torch.cumsum(phase_delta, dim=1)
        phase = util.resample_frames(phase, audio.shape[1])
        phase = phase.squeeze(-1)

        max_delay = self.param_desc['delay_ms']['range'][1] * 2 / 1000.0 * self.sr # samples
        delay_center = delay_ms / 1000.0 * self.sr # samples
        delay_value = phase * (depth * delay_center) + delay_center
        delay_phase = delay_value / max_delay # 0-> no delay 1: max_delay
        delayed = util.variable_delay(delay_phase, audio, buf_size=math.ceil(max_delay))
        return mix * delayed + (1 - mix)*audio

class ChorusFlanger(Processor):
    """
    LFO modulated delay
    no feedback
    delay_ms:
    Flanger: 1ms~5ms
    Chorus: 5ms~
    """
    
    def __init__(self, name='chorus', sample_rate=16000, delay_range=(1.0, 40.0)):
        super().__init__(name)
        self.sr = sample_rate
        self.param_desc = {
            'audio':    {'size': 1,     'range': (-1, 1),       'type': 'raw'},
            'delay_ms':    {'size': 1, 'range': delay_range, 'type': 'sigmoid'}, #ms
            'rate':     {'size': 1, 'range': (0.1, 10.0), 'type': 'sigmoid'}, #Hz
            'depth':    {'size': 1, 'range': (0, 0.25), 'type': 'sigmoid'},
            'mix':      {'size': 1, 'range': (0, 0.5), 'type': 'sigmoid'}
            }

    def forward(self, audio, delay_ms, rate, depth, mix):
        """pass audio through chorus/flanger
        Args:
            audio (torch.Tensor): [batch_size, n_samples]
            # static parameters
            delay_ms (torch.Tensor): Average delay in ms [batch_size, 1, 1]
            rate (torch.Tensor): LFO rate in Hz [batch_size, 1, 1]
            depth (torch.Tensor): Lfo depth relative to delay_ms (0~1) [batch_size, 1, 1]
            mix (torch.Tensor): wet/dry ratio (0: all dry, 1: all wet) [batch_size, n_samples or 1, 1]

        Returns:
            [torch.Tensor]: Mixed audio. Shape [batch, n_samples]
        """
        # delay: delay_ms*(1-depth) <-> delay_ms*(1+depth)

        delay_ms = delay_ms.squeeze(-1)
        rate = rate.squeeze(-1)
        depth = depth.squeeze(-1)
        mix = mix.squeeze(-1)

        max_delay = self.param_desc['delay_ms']['range'][1] * 2 / 1000.0 * self.sr # samples
        delay_center = delay_ms / 1000.0 * self.sr # samples
        n_samples = audio.shape[1]
        delay_lfo = torch.sin(torch.linspace(0, n_samples/self.sr, n_samples, device=mix.device)[None, :]*math.pi*2*rate)
        delay_value = delay_lfo * (depth*delay_center) + delay_center
        delay_phase = delay_value / max_delay
        delayed = util.variable_delay(delay_phase, audio, buf_size=math.ceil(max_delay))
        return mix * delayed + (1-mix)*audio

        