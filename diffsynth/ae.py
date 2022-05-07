import hydra
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from diffsynth.model import EstimatorSynth
from diffsynth.estimator import MelEstimator
from diffsynth.schedules import ParamSchedule
from diffsynth.spectral import Mfcc
from diffsynth.modelutils import construct_synth_from_conf

class MelEncoder(MelEstimator):
    def __init__(self, output_dim, n_mels=128, n_fft=1024, hop=256, sample_rate=16000, channels=64, kernel_size=7, strides=[2,2,2], num_layers=1, hidden_size=256, dropout_p=0.0, norm='batch'):
        super().__init__(output_dim, n_mels, n_fft, hop, sample_rate, channels, kernel_size, strides, num_layers, hidden_size, dropout_p, False, norm)

    def forward(self, audio):
        x = self.logmel(audio)
        x = self.norm(x)
        batch_size, n_mels, n_frames = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.n_mels).unsqueeze(1)
        # x: [batch_size*n_frames, 1, n_mels]
        for i, conv in enumerate(self.convs):
            x = conv(x)
        x = x.view(batch_size, n_frames, self.channels, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        D = 2 if self.bidirectional else 1
        output, _hidden = self.gru(x, torch.zeros(D * self.num_layers, batch_size, self.hidden_size, device=x.device))
        # output: [batch_size, n_frames, self.output_dim]
        return self.out(output) # no sigmoid on the output

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.gru = nn.GRUCell(input_dim, hidden_size)
        self.lin_out = nn.Linear(hidden_size, output_dim)

    def forward(self, z, n_frames):
        # z: (batch, input_dim)
        h = []
        h_t = torch.zeros(z.shape[0], self.hidden_size, device=z.device)
        for t in range(n_frames):
            h_t = self.gru(z, h_t)
            h.append(h_t)
        h = torch.stack(h, dim=1) # batch, n_frames, hidden_size
        return self.lin_out(h)

class AE(nn.Module):
    def __init__(self, encoder, decoder, encoder_dims, latent_dims):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.map_latent = nn.Linear(encoder_dims, latent_dims)

    def latent(self, enc_output):
        enc_output = enc_output[:, -1, :]
        z = self.map_latent(enc_output)
        z_params = {'loc': z, 'scale': torch.zeros_like(z)}
        #latent and latent loss
        return z, torch.zeros(1).to(z.device), z_params

    def forward(self, audio):
        h = self.encoder(audio)
        z, z_loss = self.latent(h)
        out = self.decoder(z)
        return out, z_loss