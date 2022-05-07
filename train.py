import hydra

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    import os
    from omegaconf import open_dict
    import pytorch_lightning as pl
    import torch
    from plot import AudioLogger
    import warnings
    from torch.utils.data import DataLoader, random_split
    from pytorch_lightning.callbacks import ModelCheckpoint
    from diffsynth.model import EstimatorSynth
    from diffsynth.data import SliceDataset
    pl.seed_everything(cfg.seed, workers=True)
    warnings.simplefilter('ignore', RuntimeWarning)
    # load model
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.loss)
    # loggers setup
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs", "", default_hp_metric=False, version='')
    # load dataset
    dataset = hydra.utils.instantiate(cfg.data)
    print(f'loaded {len(dataset)} samples')
    train_set, valid_set = random_split(dataset, [len(dataset)*0.9, len(dataset)*0.1], generator=torch.Generator().manual_seed(cfg.seed))
    train_dl = DataLoader(train_set, cfg.batch_size, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_set, cfg.batch_size, shuffle=False)
    # trainer setup
    # keep every checkpoint_every epochs and best epoch
    checkpoint_callback = ModelCheckpoint(dirpath=os.getcwd(), monitor=cfg.monitor, save_top_k=-1, save_last=False, every_n_epochs=cfg.checkpoint_every)
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'), AudioLogger(), checkpoint_callback]
    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=tb_logger)
    # log hyperparameters
    with open_dict(cfg):
        cfg.model.total_params = sum(p.numel() for p in model.parameters())
    # make model
    trainer.fit(model, train_dl, valid_dl, ckpt_path=cfg.ckpt)
    # torch.save(datamodule, os.path.join(os.getcwd(), 'datamodule.pt'))
    # return value used for optuna
    return trainer.callback_metrics[cfg.monitor]

if __name__ == "__main__":
    main()
