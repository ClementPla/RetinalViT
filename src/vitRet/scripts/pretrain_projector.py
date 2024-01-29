import os

import torch
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from vitRet.data.fundus import EyePACSDataModule
from vitRet.models.pretraining import PretrainModule

torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(True)



def train():
    config = Config("configs/config.yaml")
    config['logger']['project'] = 'Projector Pretraining - Superpixels'
    
    config['training'] = {'lr':1e-3}
    seed_everything(1234, workers=True)

    eyepacs_datamodule = EyePACSDataModule(**config["data"])
    model = PretrainModule(**config["training"])

    wandb_logger = WandbLogger(**config["logger"], config=config.tracked_params)
    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = wandb_logger.experiment.name

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", os.environ["WANDB_RUN_NAME"]),
    )

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=eyepacs_datamodule)
    trainer.test(model, datamodule=eyepacs_datamodule, ckpt_path="best")

if __name__ == "__main__":
    train()