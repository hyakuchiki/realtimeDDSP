import torch
import torch.nn.functional as F
from diffsynth.spectral import compute_lsd, loudness_loss, Mfcc, spectral_convergence
import pytorch_lightning as pl
from diffsynth.modelutils import construct_synth_from_conf
from diffsynth.schedules import ParamSchedule
import hydra
from itertools import chain

class EstimatorSynth(pl.LightningModule):
    def __init__(self, model_cfg, synth_cfg, losses_cfg):
        super().__init__()
        self.synth = construct_synth_from_conf(synth_cfg)
        self.estimator = hydra.utils.instantiate(model_cfg.estimator, output_dim=self.synth.ext_param_size)
        self.losses = hydra.utils.instantiate(losses_cfg.losses)
        self.loss_w_sched = ParamSchedule(losses_cfg.sched) # loss weighting
        # assert all([(loss_name in self.loss_w_sched.sched) for loss_name in self.losses])
        self.sr = model_cfg.sample_rate
        self.lr = model_cfg.lr
        self.mfcc = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20, sample_rate=self.sr)
        self.save_hyperparameters()

    def estimate_param(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: estimated parameters in Tensor ranged 0~1
        """
        return self.estimator(conditioning)

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)

        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, outputs

    def train_losses(self, output, target, loss_w=None):
        loss_dict = {}
        for k, loss in self.losses.items():
            weight = 1.0 if loss_w is None else loss_w[k]
            if weight > 0.0:
                loss_dict[k] = weight * loss(output, target)
            else:
                loss_dict[k] = 0.0
        return loss_dict

    def monitor_losses(self, output, target):
        mon_losses = {}
        # Audio losses
        target_audio = target['audio']
        resyn_audio = output['output']
        # losses not used for training
        ## audio losses
        mon_losses['lsd'] = compute_lsd(resyn_audio, target_audio)
        mon_losses['sc'] = spectral_convergence(resyn_audio, target_audio)
        mon_losses['loud'] = loudness_loss(resyn_audio, target_audio, self.sr)
        mon_losses['mfcc_l1'] = F.l1_loss(self.mfcc(resyn_audio)[:, 1:], self.mfcc(target_audio)[:, 1:])
        return mon_losses

    def training_step(self, batch_dict, batch_idx):
        # get loss weights
        loss_weights = self.loss_w_sched.get_parameters(self.global_step)
        self.log_dict({'lw/'+k: v for k, v in loss_weights.items()}, on_epoch=True, on_step=False)
        # render audio
        resyn_audio, output_dict = self(batch_dict)
        losses = self.train_losses(output_dict, batch_dict, loss_weights)
        self.log_dict({'train/'+k: v for k, v in losses.items()}, on_epoch=True, on_step=False)
        batch_loss = sum(losses.values())
        self.log('train/total', batch_loss, prog_bar=True, on_epoch=True, on_step=False)
        return batch_loss

    def validation_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(outputs, batch_dict)
        eval_losses = self.monitor_losses(outputs, batch_dict)
        losses.update(eval_losses)
        losses = {'val_{0}/{1}'.format(dataloader_idx, k): v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    def test_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(outputs, batch_dict)
        eval_losses = self.monitor_losses(outputs, batch_dict)
        losses.update(eval_losses)
        losses = {'val_{0}/{1}'.format(dataloader_idx, k): v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.estimator.parameters(), self.synth.parameters()), self.lr)
        # optimizer = torch.optim.Adam(self.estimator.parameters(), self.lr)
        return {
        "optimizer": optimizer,
        }