from typing import Any

import pytorch_lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.data.mixup import Mixup

from vitRet.models.model_factory import create_model
from vitRet.utils import lr_decay as lrd


class TrainerModule(pytorch_lightning.LightningModule):
    def __init__(self, network_config: dict, training_config: dict) -> None:
        super().__init__()
        self.network_config = network_config
        self.training_config = training_config
        self.model = create_model(**network_config)
        self.n_classes = network_config["num_classes"]
        self.loss = nn.MSELoss()
        self.metrics = torchmetrics.MetricCollection(
            {
                "Accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=network_config["num_classes"]
                ),
                "Quadratic Kappa": torchmetrics.CohenKappa(
                    num_classes=network_config["num_classes"],
                    task="multiclass",
                    weights="quadratic",
                ),
            }
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "Test accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=network_config["num_classes"]
                ),
                "Test Quadratic Kappa": torchmetrics.CohenKappa(
                    num_classes=network_config["num_classes"],
                    task="multiclass",
                    weights="quadratic",
                ),
            }
        )
        mixup_config = training_config.get("mixup", None)
        if mixup_config is not None and any([mixup_config['mixup_alpha'] > 0, 
                                             mixup_config['cutmix_alpha'] > 0,
                                             mixup_config['cutmix_minmax'] is not None]):
            self.mixup = Mixup(**mixup_config)
        else:
            self.mixup = None
            
    def training_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        if self.mixup is not None:
            image, gt = self.mixup(image, gt)
            
        pred = self.model(image, return_attention=False)
        pred = pred.flatten()
        loss = self.loss(pred, gt.float())
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True, 
                 prog_bar=True)
        return loss

    def validation_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        pred, attn = self.model(image, return_attention=True)
        pred = pred.flatten()
        attn = (attn - torch.amin(attn, (1, 2), keepdim=True)) / (
            torch.amax(attn, (1, 2), keepdim=True)
            - torch.amin(attn, (1, 2), keepdim=True)
        )
        loss = self.loss(pred, gt.float())
        
        pred = torch.round(pred).clamp(0, self.n_classes - 1)
        self.metrics.update(pred, gt)
        self.log_dict(self.metrics, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return {"pred": pred, "attn": attn, "gt": gt}

    def test_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        gt = data["label"]
        pred = self.model(image, return_attention=False)
        pred = pred.flatten()
        pred = torch.round(pred).clamp(0, self.n_classes - 1)
        self.test_metrics.update(pred, gt)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self) -> Any:
        param_groups = lrd.param_groups_lrd(self.model, 
                                            self.training_config["optimizer"]["weight_decay"],
        no_weight_decay_list=self.model.no_weight_decay,
        layer_decay=self.training_config.get("layer_decay", 0.))
        
        
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.training_config["lr"], 
            **self.training_config["optimizer"]
        )
        return [optimizer], [
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer,
                    max_lr=self.training_config["lr"],
                    total_steps=self.trainer.estimated_stepping_batches,
                    pct_start=self.training_config["pct_start"])
                ,
                "interval": "step",
            }
        ]

class LogValidationAttentionMap(pl.Callback):
    def __init__(self, wandb_logger, n_images=4, frequency=2):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        self.frequency = frequency
        self.__call = 0

        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if batch_idx < 1 and trainer.is_global_zero:
            n = self.n_images
            x = batch["image"][:n].float()
            attn = outputs["attn"][:n]
            columns = ["image", "attention_map"]
            data = [
                [wandb.Image(x_i), wandb.Image(attn_i / attn_i.max())]
                for x_i, attn_i in list(zip(x, attn))
            ]
            self.wandb_logger.log_table(
                data=data, key=f"Validation First Batch", columns=columns
            )
            self.__call += 1
