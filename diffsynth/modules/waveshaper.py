import torch
import torch.nn as nn
import torch.nn.functional as F
from diffsynth.processor import Processor
import diffsynth.util as util
import math

class WaveShaper(Processor):
    """
    Based on FastNEWT from Neural Waveshaping Synthesis (ISMIR 2021) by Ben Hayes et al.
    https://github.com/ben-hayes/neural-waveshaping-synthesis

    Uses grid_sampler instead of lookup.
    """
    
    def __init__(self, name='waveshaper', n_functions=64, xy_lim=3.0, table_size=4096):
        super().__init__(name)
        self.xy_lim = xy_lim # waveshaper is defined between -xy_lim~xy_lim
        # initialize as identity function
        tables = torch.linspace(-xy_lim, xy_lim, table_size).unsqueeze(0).repeat(n_functions, 1)
        self.tables = nn.Parameter(tables)
        self.n_functions = n_functions
        self.param_desc = {
            'audio': {'size': 1, 'range': (-1, 1), 'type': 'raw'},
            'dist_index': {'size': n_functions, 'range': (-1, 1), 'type': 'sigmoid'},
            'dist_bias': {'size': n_functions, 'range': (-1, 1), 'type': 'sigmoid'},
            'wave_mix': {'size': n_functions, 'range': (0, 1), 'type': 'sigmoid'}
        }

    def waveshape(self, input_audio):
        batch_size, n_samples, n_funcs = input_audio.shape
        wave4d = self.tables[:, None, None, :] # (N=n_funcs, C=1, H=1, W=table_size)
        grid_x = input_audio.permute(2, 0, 1).unsqueeze(-1) # N=n_funcs, H=batch_size, W=n_samples, 1
        # input_audio can be -xy_lim~xy_lim but grid has to be -1~1
        grid_x = grid_x / self.xy_lim
        # Height/y is a dummy dimension
        grid_y = torch.zeros(n_funcs, batch_size, n_samples, 1, device=input_audio.device) # N=n_funcs, H=batch_size, W=n_samples, 1
        grid = torch.cat([grid_x, grid_y], dim=-1)
        # N=n_funcs, 1, H=batch_size, W=n_samples
        output = torch.nn.functional.grid_sample(wave4d, grid, mode='bilinear', align_corners=True) 
        return output.squeeze(1).permute(1, 2, 0)

    def forward(self, audio, dist_index, dist_bias, wave_mix):
        """pass audio through waveshaper
        Args:
            audio (torch.Tensor): [batch_size, n_samples]
            
            Distortion parameters below are [batch_size, frame_size, n_functions]
            dist_index (torch.Tensor): Multiplier for waveshaper input
            dist_bias (torch.Tensor): Bias for waveshaper input
            wave_mix (torch.Tensor): Mix for each waveshaper

        Returns:
            [torch.Tensor]: Mixed audio. Shape [batch, n_samples]
        """
        audio = audio.unsqueeze(-1).expand(-1, -1, self.n_functions)
        
        dist_index = util.resample_frames(dist_index, audio.shape[1])
        dist_bias = util.resample_frames(dist_bias, audio.shape[1])
        wave_mix = util.resample_frames(wave_mix, audio.shape[1])

        audio = audio * dist_index + dist_bias
        audio = self.waveshape(audio)
        return torch.sum(audio * wave_mix, dim=-1)