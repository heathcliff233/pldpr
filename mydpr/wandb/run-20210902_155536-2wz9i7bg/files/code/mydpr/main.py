import os
import torch
import esm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.biencoder import MyEncoder
from dataset.cath35 import Cath35DataModule

os.environ["MASTER_PORT"] = "39522"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,6,7"
model_dir = "../model96/"
use_wandb = True
batch_sz = 24
cath_dir = "/share/wangsheng/train_test_data/cath35_20201021/cath35_a3m/"
cath_cfg = "../../mydpr/split/"
if use_wandb:
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(project='pl_ft', log_model=True)
else:
    logger = True
#TODO: add arg parser

callback_checkpoint = ModelCheckpoint(
    monitor="val_acc",
    dirpath=model_dir,
    filename="pl_biencoder-{epoch:03d}-{val_acc:.2f}",
    mode="max",
    save_top_k=3,
    save_on_train_epoch_end=True,
)

def main():
    pl.seed_everything(1234)
    #encoder, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    ##################################
    #model = MyEncoder(encoder)
    #prev = torch.load('../../mydpr/continue_train/59.pth')
    #later = dict((k[7:], v) for (k,v) in prev.items())
    #model.load_state_dict(later)
    #model.cuda()
    ##################################
    #model = MyEncoder.load_from_checkpoint("/user/hongliang/pldpr/model/pl_biencoder-epoch=004-val_acc=0.99.ckpt")
    model = MyEncoder()
    dm = Cath35DataModule(cath_dir, cath_cfg, batch_sz, model.alphabet)
    trainer = pl.Trainer(
        gpus=[0,1,2,3], 
        accelerator='ddp', 
        accumulate_grad_batches=4, 
        precision=16, 
        replace_sampler_ddp=False, 
        gradient_clip_val=0.5,
        logger=logger,
        log_every_n_steps=20,
        max_epochs=100,
        callbacks=callback_checkpoint,
        fast_dev_run=False,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()