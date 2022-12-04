import torch
import torch.nn as nn
from diffsynth.layers import Normalize2d
from torchaudio.transforms import MFCC

class MFCCGRUEncoder(nn.Module):
    def __init__(self, latent_size, sample_rate, hidden_size=512, n_mels=128, n_mfcc=30, n_fft=2048, hop_length=960) -> None:
        super().__init__()
        self.mfcc = MFCC(sample_rate, n_mfcc=n_mfcc, norm='ortho', melkwargs=dict(hop_length=hop_length, n_fft=n_fft, n_mels=n_mels, center=True, f_min=20, f_max=sample_rate/2))
        self.norm = Normalize2d("instance")
        self.gru = nn.GRU(n_mfcc, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, latent_size)

    def forward(self, audio):
        # output latent variable
        x_mfcc = self.norm(self.mfcc(audio)) # batch, n_mfcc, time
        x_mfcc = x_mfcc.permute(0, 2, 1)
        z = self.linear(self.gru(x_mfcc)[0])
        return z