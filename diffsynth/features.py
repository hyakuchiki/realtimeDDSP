from typing import Any, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import math, warnings
import torch
import torch.nn as nn
import torchaudio
import nnAudio.features

from hdec.spectral import chroma_filterbank, melscale_fbanks
from diffsynth.util import center_pad, slice_windows
from hdec.modules import yingram

EVAL_MAX_BATCH_SIZE = 2


class Feature(ABC, nn.Module):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        center: str = "half-hop",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        if hop_size is None:
            self.frame_rate = frame_rate
            self.hop_size = int(sample_rate // frame_rate)
        else:
            self.hop_size = hop_size
            self.frame_rate = int(sample_rate // hop_size)
        self.center = center
        if center == "half-hop":  # first window centered around half of hop size
            cache_size = (window_size - self.hop_size) // 2
        elif center == "half-window":
            cache_size = 0
        elif center == "zero":
            cache_size = window_size // 2
        # for streaming
        self.register_buffer("cache", torch.zeros(EVAL_MAX_BATCH_SIZE, 1, cache_size))
        self.streaming = False

    @abstractmethod
    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, time)
        # output: (batch_size, n_frames, feat_dim)
        pass

    def get_n_frames(self, input_length: int) -> float:
        return float()

    def stream(self, mode: bool = True):
        self.streaming = mode

    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: Optional[int] = None,
    ):
        """
        audio: ((batch_size, (n_channels)), time)

        Output: ((batch_size, (n_channels)), n_frames, feat_dim)
        """
        input_ndim = audio.ndim
        if input_ndim == 1:
            audio = audio[None, None, :]
        elif input_ndim == 2:
            audio = audio[None, :, :]  # TODO: fix BUG with torchscript export
        if self.streaming:
            if self.cache.numel() > 0:
                if self.cache.shape[0] < audio.shape[0]:
                    self.cache = self.cache.repeat(
                        audio.shape[0] // self.cache.shape[0], 1, 1
                    )
                x = torch.cat([self.cache[: audio.shape[0]], audio], dim=-1)
            else:
                x = audio
            n_frames = (x.shape[-1] - self.window_size) // self.hop_size + 1
            # starting position of frame that wasn't calculated
            next_pos = self.hop_size * n_frames
            # save as new cache
            self.cache = x[:, :, next_pos:].clone()
        else:
            if sample_rate is not None and sample_rate != sample_rate:
                # resample
                audio = torchaudio.functional.resample(
                    audio, sample_rate, self.sample_rate
                )
            if self.center != "half-window":
                x = center_pad(
                    audio,
                    self.window_size,
                    self.hop_size,
                    pad_last=True,
                    center_type=self.center,
                )
            else:
                x = audio
        # features are calculated as (Batch/channel, Time)
        batch, channel, time = audio.shape
        feat = self.compute_feature(x.flatten(0, 1))
        feat = feat.view(batch, channel, feat.shape[-2], feat.shape[-1])
        if input_ndim == 1:
            feat = feat[0, 0, :]
        elif input_ndim == 2:
            feat = feat[0]
        return feat


class FeatureProcessor(nn.Module):
    def __init__(self, features: Dict[str, Feature]) -> None:
        super().__init__()
        self.features = nn.ModuleDict(features)
        # make sure all frame_rates are the same so feats line up
        fpss = [feat.frame_rate for feat in self.features.values()]
        assert all(x == fpss[0] for x in fpss)
        self.resamples: Dict[Tuple[int, int], nn.Module] = {}
        if len(self.features) > 0:
            print("calculating features:", list(self.features.keys()))

    @torch.jit.unused
    def resample(
        self,
        sample_rate: int,
        target_sr: int,
        audio: torch.Tensor,
        inputs: Dict[int, torch.Tensor],
    ):
        # needs resampling
        if (sample_rate, target_sr) not in self.resamples:
            # make resampling kernel only once
            self.resamples[(sample_rate, target_sr)] = torchaudio.transforms.Resample(
                sample_rate, target_sr
            )
        x_resamp = self.resamples[(sample_rate, target_sr)](audio)
        # save resampled audio for other features to maybe use
        inputs[target_sr] = x_resamp

    def forward(self, audio: torch.Tensor, sample_rate: int) -> Dict[str, torch.Tensor]:
        """
        audio: ((batch_size), (n_channels), time)

        Output: ((batch_size), (n_channels), n_frames, feat_dim)
        """
        inputs = {sample_rate: audio}
        feature_data: Dict[str, torch.Tensor] = {}
        for feat_name, feat_mod in self.features.items():
            target_sr = feat_mod.sample_rate
            if target_sr not in inputs:
                self.resample(sample_rate, feat_mod.sample_rate, audio, inputs)
            feature_data[feat_name] = feat_mod(inputs[target_sr])
        return feature_data


class SpectralCentroid(Feature):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        center: bool = "half-hop",
        n_fft: Optional[int] = None,
        window: str = "hann",
    ):
        super().__init__(sample_rate, window_size, hop_size, frame_rate, center)
        n_fft = n_fft if n_fft else window_size
        self.spec = nnAudio.features.STFT(
            n_fft=window_size,
            hop_length=self.hop_size,
            center=False,
            verbose=False,
            output_format="Magnitude",
            window=window,
        )
        freqs = torch.fft.rfftfreq(self.window_size, 1 / self.sample_rate)[
            None, :, None
        ]
        self.register_buffer("freqs", freqs, persistent=False)

    def process_spec(self, spec: torch.Tensor):
        cent = (self.freqs * spec).sum(dim=-2) / (spec.sum(dim=-2) + 1e-5)
        return cent.unsqueeze(-1)  # batch, n_frames, 1

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.spec(x)
        cent = (self.freqs * spec).sum(dim=-2) / (spec.sum(dim=-2) + 1e-5)
        return cent.unsqueeze(-1)  # batch, n_frames, 1


class Volume(Feature):
    # not loudness, just energy
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        center: bool = "half-hop",
    ):
        super().__init__(sample_rate, window_size, hop_size, frame_rate, center)
        window = torch.hann_window(window_size)
        self.register_buffer("window", window)

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        x_sqr = x**2
        a2_win = slice_windows(
            x_sqr, self.window_size, self.hop_size, "none", pad=False
        )
        rms = a2_win.mean(dim=-1).sqrt()
        return rms.unsqueeze(-1)  # batch, n_frames, 1


class CREPEF0(Feature):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        f0_range: List[float] = [31, 1984],
        center: bool = "half-hop",
        viterbi: bool = True,
        device: str = "cuda",
    ):
        import torchcrepe

        super().__init__(sample_rate, window_size, hop_size, frame_rate, center)
        self.device = device
        assert (
            self.sample_rate == 16000
        ), f"CREPE only works in 16k sampling rate but got sr={sample_rate}"
        assert (
            self.window_size == 1024
        ), f"CREPE window_size is only 1024 but got {window_size}"
        self.f0_range = f0_range
        self.f0_dec = torchcrepe.decode.viterbi if viterbi else torchcrepe.decode.argmax

    def compute_feature(self, x: torch.Tensor) -> torch.Tensor:
        import torchcrepe
        from hdec.f0 import median_pool_1d, process_f0

        # x: (1, time)
        with torch.no_grad():
            f0_hz, periodicity = torchcrepe.predict(
                x,
                self.sample_rate,
                hop_length=self.hop_size,
                fmin=self.f0_range[0],
                fmax=self.f0_range[1],
                model="full",
                decoder=self.f0_dec,
                device=self.device,
                batch_size=64,
                return_periodicity=True,
                pad=False,
            )
        periodicity = median_pool_1d(periodicity, 4)
        f0_hz = process_f0(f0_hz, periodicity)
        return f0_hz.unsqueeze(-1).cpu()  # batch, n_frames, 1



class MelSpectrogram(Feature):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,  # = n_fft
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        center: bool = "half-hop",
        n_mels: int = 256,
        trainable_mel: bool = True,
        window: str = "hann",
        n_fft: Optional[int] = None,
        f_min: Optional[int] = None,
        f_max: Optional[int] = None,
        trainable_stft: bool = False,
        logstft: bool = False,
    ):
        super().__init__(sample_rate, window_size, hop_size, frame_rate, center)
        self.logstft = logstft
        if logstft:
            self.stft = nnAudio.features.STFT(
                n_fft=window_size if n_fft is None else n_fft,
                hop_length=self.hop_size,
                center=False,
                trainable=trainable_stft,
                window=window,
                verbose=False,
                freq_scale="log",
                sr=sample_rate,
                output_format="Magnitude",
            )
            fb = melscale_fbanks(
                torch.tensor(self.stft.bins2freq, dtype=torch.float),
                f_min,
                f_max,
                n_mels=n_mels,
                mel_scale="htk",
                norm="slaney",
            )
            self.register_buffer("fb", fb, persistent=False)
        else:
            self.melgram = nnAudio.features.MelSpectrogram(
                sample_rate,
                n_fft=window_size if n_fft is None else n_fft,
                win_length=window_size,
                n_mels=n_mels,
                hop_length=self.hop_size,
                center=False,
                trainable_mel=trainable_mel,
                window=window,
                verbose=False,
                trainable_STFT=trainable_stft,
                fmin=f_min,
                fmax=f_max,
            )

    def compute_feature(self, x: torch.Tensor):
        if self.logstft:
            specgram = self.stft(x)
            melgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(
                -1, -2
            )
            return melgram
        else:
            return self.melgram(x)


class Spectrogram(Feature):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,  # = n_fft
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        center: bool = "half-hop",
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        n_fft: Optional[int] = None,
        trainable_stft: bool = True,
        freq_scale: str = "no",
        window: str = "hann",
        output_format: str = "Magnitude",
    ):
        super().__init__(sample_rate, window_size, hop_size, frame_rate, center)
        self.specgram = nnAudio.features.STFT(
            n_fft=window_size if n_fft is None else n_fft,
            hop_length=self.hop_size,
            center=False,
            trainable=trainable_stft,
            window=window,
            fmin=fmin,
            fmax=fmax,
            verbose=False,
            freq_scale=freq_scale,
            sr=sample_rate,
            output_format=output_format,
        )

    def compute_feature(self, x: torch.Tensor):
        return self.specgram(x)



class SpectralRolloffs(Feature):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,  # = n_fft
        hop_size: Optional[int] = None,
        frame_rate: Optional[int] = None,
        center: bool = "half-hop",
        n_fft: Optional[int] = None,
        window: str = "hann",
        roll_percent=[0.1, 0.9],
    ):
        super().__init__(sample_rate, window_size, hop_size, frame_rate, center)
        n_fft = window_size if n_fft is None else n_fft
        self.specgram = nnAudio.features.STFT(
            n_fft=window_size if n_fft is None else n_fft,
            hop_length=self.hop_size,
            center=False,
            window=window,
            verbose=False,
            output_format="Magnitude",
        )
        self.roll_percent = roll_percent
        self.register_buffer(
            "fft_freqs",
            torch.fft.fftfreq(n=n_fft, d=1 / sample_rate)[: n_fft // 2],
            persistent=False,
        )

    def compute_feature(self, x: torch.Tensor):
        S = self.specgram(x)
        total_energy = torch.cumsum(S, dim=-2)
        threshold_lo = self.roll_percent[0] * total_energy[..., -1:, :]
        threshold_hi = self.roll_percent[1] * total_energy[..., -1:, :]
        mask = torch.arange(S.shape[-2] + 1, 1, step=-1, device=x.device)[:, None]
        idx_lo = torch.argmax(mask * (total_energy > threshold_lo).long(), dim=-2)
        idx_hi = torch.argmax(mask * (total_energy > threshold_hi).long(), dim=-2)
        rolloff = torch.stack([self.fft_freqs[idx_lo], self.fft_freqs[idx_hi]], dim=-1)
        return rolloff  # batch, num_frames, 2
