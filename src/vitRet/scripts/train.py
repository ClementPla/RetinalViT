import os

import torch
from nntools.utils import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from vitRet.data.fundus import DDRDataModule, EyePACSDataModule
from vitRet.my_lightning_module import LogValidationAttentionMap, TrainerModule

torch.set_float32_matmul_precision("medium")

def train():
    config = Config("configs/config.yaml")
    seed_everything(1234, workers=True)

    datamodule = EyePACSDataModule(**config["data"])
    # ddr_datamodule = DDRDataModule(**config["data"])
    datamodule.setup('fit')
    datamodule.setup('validate')
    datamodule.setup('test')
    model = TrainerModule(config["model"], config["training"])

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
    log_image_callback = LogValidationAttentionMap(wandb_logger, frequency=2)
    # profiler = AdvancedProfiler(dirpath=".", filename="profiler.txt")
    trainer = Trainer(
        **config["trainer"],
        # profiler=profiler,
        # fast_dev_run=4,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            log_image_callback,
            # EarlyStopping(monitor="val_loss", patience=10),
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

