import os
import torch
import esm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.biencoder import MyEncoder
from dataset.cath35 import UniclustDataModule

os.environ["MASTER_PORT"] = "39524"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_dir = "../model_uni/"
use_wandb = False
batch_sz = 224
cath_dir = "/share/hongliang/fastMSA-uc/dataset/"
if use_wandb:
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='uniclust_pl', log_model=False)
else:
    logger = True
#TODO: add arg parser

callback_checkpoint = ModelCheckpoint(
    monitor="val_acc",
    dirpath=model_dir,
    filename="pl_biencoder-{epoch:03d}-{val_acc:.4f}",
    mode="max",
    save_top_k=5,
    save_on_train_epoch_end=True,
)

def main():
    pl.seed_everything(1234)
    model = MyEncoder().load_from_checkpoint(
        #checkpoint_path="../model-prem/pl_biencoder-epoch=012-val_acc=0.7992.ckpt"
        checkpoint_path="../model_test/pl_biencoder-epoch=585-val_acc=0.8470.ckpt"
    )
    dm = UniclustDataModule(cath_dir, cath_dir, batch_sz, model.alphabet)
    trainer = pl.Trainer(
        gpus = 0,
        #gpus=[0,1,2,3], 
        #accelerator='ddp', 
        #accumulate_grad_batches=4, 
        #precision=16, 
        #replace_sampler_ddp=False, 
        #gradient_clip_val=0.5,
        logger=logger,
        log_every_n_steps=10,
        max_epochs=100,
        callbacks=callback_checkpoint,
        fast_dev_run=False,
        #resume_from_checkpoint="../model-prem/pl_biencoder-epoch=012-val_acc=0.7992.ckpt"
    )
    #trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
