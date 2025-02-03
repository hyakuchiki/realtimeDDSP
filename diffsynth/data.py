import os, glob, pickle, itertools, warnings
from tqdm.auto import tqdm
import torchaudio
import torch
import lmdb
from nnAudio.features import MFCC

from torch.utils.data import Dataset
import torch.nn.functional as F
from diffsynth.f0 import compute_f0, FMIN, FMAX
from diffsynth.spectral import compute_loudness

class SliceDataset(Dataset):
    # slice length [s] sections from longer audio files like urmp
    # some of LMDB code borrowed from UDLS
    # https://github.com/caillonantoine/UDLS/tree/7a99c503eb02ca60852626ca0542ddc1117295ac (A. Caillon, MIT License)
    # and https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py
    def __init__(
        self,
        raw_dir,
        db_path,
        sample_rate=48000,
        length=1.0,
        frame_rate=None,
        hop_length=512,
        window_size=2048,
        f0_range=(FMIN, FMAX),
        n_mfcc=20,
    ):
        self.raw_dir = raw_dir
        self.sample_rate = sample_rate
        self.length = length
        self.f0_range = f0_range
        self.window_size = window_size
        if frame_rate is not None:
            hop_length = sample_rate // frame_rate
            assert sample_rate % frame_rate == 0
        self.hop_length = hop_length
        self.mfcc = MFCC(
            sample_rate,
            n_mfcc,
            center=True,
            hop_length=hop_length,
            n_fft=window_size,
            verbose=False,
        )
        os.makedirs(db_path, exist_ok=True)
        # max of ~100GB
        self.lmdb_env = lmdb.open(db_path, map_size=int(1e11), lock=False)
        # check if lmdb database exists
        with self.lmdb_env.begin(write=False) as txn:
            lmdblength = txn.get("length".encode("utf-8"))
        self.len = int(lmdblength) if lmdblength is not None else 0
        if self.len == 0: # database not made or empty
            self.preprocess()
        if self.len == 0:
            raise Exception(f'No data found @ {raw_dir}')

    def calculate_features(self, audio):
        # calculate f0 and loudness
        # pad=True->center=True
        f0, periodicity = compute_f0(
            audio,
            self.sample_rate,
            hop_length=self.hop_length,
            f0_range=self.f0_range,
        )
        loudness = compute_loudness(
            audio,
            self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.window_size,
            center=True,
        )
        mfcc = self.mfcc(audio)
        return f0, periodicity, loudness, mfcc

    def preprocess(self):
        self.raw_files = sorted(
            list(
                itertools.chain(
                    *(
                        glob.glob(
                            os.path.join(self.raw_dir, f"**/*.{ext}"), recursive=True
                        )
                        for ext in ["mp3", "wav", "flac", "MP3", "WAV", "FLAC"]
                    )
                )
            )
        )
        # load audio
        idx = 0
        resample = {}
        for audio_file in tqdm(self.raw_files):
            try:
                audio, orig_sr = torchaudio.load(audio_file)
                audio = audio.mean(dim=0) # force mono
            except RuntimeError:
                warnings.warn('Falling back to librosa because torchaudio loading (sox) failed.')
                import librosa
                audio, orig_sr = librosa.load(audio_file, sr=None, mono=True)
                audio = torch.from_numpy(audio)
            # resample
            if orig_sr != self.sample_rate:
                if orig_sr not in resample:
                    # save kernel
                    resample[orig_sr] = torchaudio.transforms.Resample(
                        orig_sr, self.sample_rate, lowpass_filter_width=64, rolloff=0.99
                    )
                audio = resample[orig_sr](audio)
            # pad so that it can be evenly sliced
            len_audio_chunk = int(self.sample_rate*self.length)
            pad_audio = (len_audio_chunk - (audio.shape[-1] % len_audio_chunk)) % len_audio_chunk
            audio = F.pad(audio, (0, pad_audio))
            # split
            audios = torch.split(audio, len_audio_chunk)

            for x in audios:
                # calculate features after slicing
                if max(abs(x)) < 1e-2:
                    # only includes silence
                    continue
                f0, periodicity, loudness, mfcc = self.calculate_features(
                    x
                )  # (1, n_frames)
                if periodicity is not None and (periodicity < 1e-3).all():
                    # too noisy to use for data
                    continue
                data = (x.numpy(), f0.numpy(), loudness.numpy(), mfcc.numpy())
                with self.lmdb_env.begin(write=True) as txn:
                    txn.put(f'{idx:08d}'.encode('utf-8'), pickle.dumps(data))
                idx+=1
        # write dataset length to database
        with self.lmdb_env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), f'{idx:08d}'.encode('utf-8'))
        self.len = idx

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            x, f0, loudness, mfcc = pickle.loads(txn.get(f"{idx:08d}".encode("utf-8")))
        return {
            "audio": x,
            "f0": f0[:, None],
            "loud": loudness[:, None],
            "mfcc": mfcc[0],
        }
