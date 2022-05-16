import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.callbacks import Callback

def plot_spec(y, ax, sr=16000):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.label_outer()

def plot_recons(x, x_tilde, plot_dir, name=None, epochs=None, sr=16000, num=6, save=True):
    """Plot spectrograms/waveforms of original/reconstructed audio

    Args:
        x (numpy array): [batch, n_samples]
        x_tilde (numpy array): [batch, n_samples]
        sr (int, optional): sample rate. Defaults to 16000.
        dir (str): plot directory.
        name (str, optional): file name.
        epochs (int, optional): no. of epochs.
        num (int, optional): number of spectrograms to plot. Defaults to 6.
    """
    fig, axes = plt.subplots(num, 4, figsize=(15, 30), squeeze=False)
    for i in range(num):
        plot_spec(x[i], axes[i, 0], sr)
        plot_spec(x_tilde[i], axes[i, 1], sr)
        axes[i, 2].plot(x[i])
        axes[i, 2].set_ylim(-1,1)
        axes[i, 3].plot(x_tilde[i])
        axes[i, 3].set_ylim(-1,1)
    if save:
        if epochs:
            fig.savefig(os.path.join(plot_dir, 'epoch{:0>3}_recons.png'.format(epochs)))
            plt.close(fig)
        else:
            fig.savefig(os.path.join(plot_dir, name+'.png'))
            plt.close(fig)
    else:
        return fig

def save_to_board(i, name, writer, orig_audio, resyn_audio, plot_num=4, sr=16000):
    orig_audio = orig_audio.detach().cpu()
    resyn_audio = resyn_audio.detach().cpu()
    for j in range(plot_num):
        writer.add_audio('{0}_orig/{1}'.format(name, j), orig_audio[j].unsqueeze(0), i, sample_rate=sr)
        writer.add_audio('{0}_resyn/{1}'.format(name, j), resyn_audio[j].unsqueeze(0), i, sample_rate=sr)
    fig = plot_recons(orig_audio.detach().cpu().numpy(), resyn_audio.detach().cpu().numpy(), '', sr=sr, num=plot_num, save=False)
    writer.add_figure('plot_recon_{0}'.format(name), fig, i)

class AudioLogger(Callback):
    def __init__(self, batch_frequency=1000, sr=16000):
        super().__init__()
        self.batch_freq = batch_frequency
        self.sr = sr

    @rank_zero_only
    def log_local(self, writer, name, current_epoch, orig_audio, resyn_audio):
        save_to_board(current_epoch, name, writer, orig_audio, resyn_audio, plot_num=6, sr=self.sr)

    def log_audio(self, pl_module, batch, batch_idx, name="train"):
        if batch_idx % self.batch_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            # get audio
            with torch.no_grad():
                resyn_audio, _outputs = pl_module(batch)
            resyn_audio = torch.clamp(resyn_audio.detach().cpu(), -1, 1)
            orig_audio = torch.clamp(batch['audio'].detach().cpu(), -1, 1)
            
            self.log_local(pl_module.logger.experiment, name, pl_module.current_epoch, orig_audio, resyn_audio)

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_audio(pl_module, batch, batch_idx, name="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_audio(pl_module, batch, batch_idx, name="val_"+str(dataloader_idx))

class SaveEvery(Callback):
    def __init__(self, every_n=50):
        super().__init__()
        self.every_n = every_n

    def on_validation_epoch_end(self, trainer, pl_module):
        cur_epoch = pl_module.current_epoch
        if (cur_epoch+1) % self.every_n == 0:
            trainer.save_checkpoint(f"epoch_{cur_epoch}.ckpt")