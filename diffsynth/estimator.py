import torch
import torch.nn as nn
from nnAudio.features import MFCC

from diffsynth.layers import MLP
from diffsynth.f0 import FMIN, FMAX
from diffsynth.spectral import DB_RANGE
from typing import Dict

class EstimatorFL(nn.Module):
    def __init__(self, output_dim, hidden_size=512):
        super().__init__()
        # control encoder
        self.mlp_0 = MLP(2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mlp_1 = MLP(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)
        self.output_dim = output_dim
        self.frange = (FMIN, FMAX)
        self.db_range = DB_RANGE

    def forward(self, conditioning: Dict[str, torch.Tensor]):
        # normalize input 0~1
        f0 = (conditioning['f0'] - self.frange[0]) / (self.frange[1]-self.frange[0])
        loud = (conditioning['loud'] / self.db_range) + 1.0
        x = torch.cat([f0, loud], dim=-1) # batch, n_frames, feat_dim=2
        x, _hidden = self.gru(self.mlp_0(x))
        out = self.out(self.mlp_1(x))
        return torch.sigmoid(out)


class EstimatorFLZ(nn.Module):
    def __init__(self, output_dim, sr, hidden_size=512, n_mfcc=20):
        super().__init__()
        # control encoder
        self.mfcc_norm = nn.BatchNorm1d(n_mfcc)
        self.mlp_0 = MLP(2 + n_mfcc, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mlp_1 = MLP(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)
        self.output_dim = output_dim
        self.frange = (FMIN, FMAX)
        self.db_range = DB_RANGE

    def forward(self, conditioning: Dict[str, torch.Tensor]):
        # normalize input 0~1
        mfcc = self.mfcc_norm(conditioning["mfcc"]).permute(0, 2, 1)
        f0 = (conditioning["f0"] - self.frange[0]) / (self.frange[1] - self.frange[0])
        loud = (conditioning["loud"] / self.db_range) + 1.0
        # mfcc = mfcc[:, : f0.shape[1], :]
        x = torch.cat([mfcc, f0, loud], dim=-1)  # batch, n_frames, feat_dim=2
        x, _hidden = self.gru(self.mlp_0(x))
        out = self.out(self.mlp_1(x))
        return torch.sigmoid(out)
