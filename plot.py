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
    def __init__(self, batch_frequency=1000):
        super().__init__()
        self.batch_freq = batch_frequency

    @rank_zero_only
    def log_local(self, writer, name, current_epoch, orig_audio, resyn_audio):
        save_to_board(current_epoch, name, writer, orig_audio, resyn_audio)

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
            
            self.log_local(pl_module.logger.experiment[0], name, pl_module.current_epoch, orig_audio, resyn_audio)

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

    # def on_fit_end(self, trainer, pl_module):
    #     trainer.save_checkpoint(f"fit_end.ckpt")

def save_to_board_mel(i, writer, orig_mel, recon_mel, plot_num=8):
    orig_mel = orig_mel.detach().cpu()
    recon_mel = recon_mel.detach().cpu()

    fig, axes = plt.subplots(2, plot_num, figsize=(30, 8))
    for j in range(plot_num):
        axes[0, j].imshow(orig_mel[j], aspect=0.25)
        axes[1, j].imshow(recon_mel[j], aspect=0.25)
    fig.tight_layout()
    writer.add_figure('plot_recon', fig, i)

def plot_param_dist(param_stats):
    """
    violin plot of parameter values
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    labels = param_stats.keys()
    parts = ax.violinplot(param_stats.values(), showmeans=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_ylim(0, 1)
    return fig

def plot_envelope(target_audio, amp_env, name):
    fig, ax = plt.subplots()
    ax.plot(target_audio, zorder=0)
    ax.plot(amp_env[:, 0], c='r', zorder=10)
    if amp_env.shape[-1] > 1:
        ax.plot(amp_env[:, 1], c='orange', zorder=5)
    ax.axis('off')
    fig.savefig('{0}.png'.format(name))
    plt.close(fig)