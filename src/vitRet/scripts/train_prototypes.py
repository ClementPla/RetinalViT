import os

import torch
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from vitRet.data.segmentation import MaplesDR
from vitRet.models.prototypes_vit.prototype_trainer import LogValidationPredictedPrototypeMap, PrototypeTrainer

torch.set_float32_matmul_precision("medium")


def train():
    config = Config("configs/config_prototype.yaml")

    seed_everything(1234, workers=True)

    maplesdr_datamodule = MaplesDR(**config["data"])
    model = PrototypeTrainer(**config['model'], **config["training"])

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
    logcallback = LogValidationPredictedPrototypeMap(wandb_logger=wandb_logger)
    config["trainer"]["strategy"] = "ddp_find_unused_parameters_true"
    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            logcallback,
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=maplesdr_datamodule)
    trainer.test(model, datamodule=maplesdr_datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
