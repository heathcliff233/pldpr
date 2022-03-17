import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from mydpr.model.biencoder import MyEncoder
from mydpr.dataset.cath35 import UniclustDataModule


def configure_callbacks(cfg: DictConfig):
    return ModelCheckpoint(
        monitor=cfg.callback.monitor,
        dirpath=cfg.callback.dirpath,
        filename=cfg.callback.filename,
        mode=cfg.callback.mode,
        save_top_k=cfg.callback.save_top_k,
        save_on_train_epoch_end=cfg.callback.save_on_train_epoch_end,
    )

@hydra.main(config_path="conf", config_name="train_conf")
def main(cfg: DictConfig):

    os.environ["MASTER_PORT"] = cfg.trainer.port
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.trainer.devices

    if cfg.logger.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=cfg.logger.project, log_model=cfg.logger.log_model)
    else:
        logger = True

    pl.seed_everything(cfg.trainer.seed)
    model = MyEncoder()
    if cfg.model.resume: 
        model.load_from_checkpoint(
        checkpoint_path=cfg.model.ckpt_path,
    )

    dm = UniclustDataModule(cfg.trainer.cath_dir, cfg.trainer.cath_dir, cfg.trainer.batch_size, model.alphabet)

    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus, 
        accelerator=cfg.trainer.accelerator, 
        accumulate_grad_batches=cfg.trainer.acc_step, 
        precision=cfg.trainer.precision, 
        #gradient_clip_val=0.5,
        logger=logger,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=configure_callbacks(cfg),
        fast_dev_run=False,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
