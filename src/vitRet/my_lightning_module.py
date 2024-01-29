from typing import Any

import pytorch_lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.data.mixup import Mixup

import wandb
from vitRet.models.model_factory import create_model
from vitRet.utils import lr_decay as lrd
from vitRet.utils.images import spatial_batch_normalization


class TrainerModule(pytorch_lightning.LightningModule):
    def __init__(self, network_config: dict, training_config: dict) -> None:
        super().__init__()

        self.as_regression = training_config.get("as_regression", False)
        self.n_classes = network_config["num_classes"]

        if self.as_regression:
            network_config["num_classes"] = 1

        self.network_config = network_config
        self.training_config = training_config

        self.model = create_model(**network_config)

        if self.as_regression:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.metrics = torchmetrics.MetricCollection(
            {
                "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes),
                "Quadratic Kappa": torchmetrics.CohenKappa(
                    num_classes=self.n_classes,
                    task="multiclass",
                    weights="quadratic",
                ),
            }
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "Test accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes),
                "Test Quadratic Kappa": torchmetrics.CohenKappa(
                    num_classes=self.n_classes,
                    task="multiclass",
                    weights="quadratic",
                ),
            }
        )
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.n_classes, task="multiclass")
        mixup_config = training_config.get("mixup", None)
        if mixup_config is not None and any(
            [
                mixup_config["mixup_alpha"] > 0,
                mixup_config["cutmix_alpha"] > 0,
                mixup_config["cutmix_minmax"] is not None,
            ]
        ):
            self.mixup = Mixup(**mixup_config)
        else:
            self.mixup = None

    def training_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        segments = data["segments"]
        gt = data["label"]
        if self.mixup is not None:
            image, gt = self.mixup(image, gt)

        logits = self.model(image, segments, return_attention=False)
        loss = self.get_loss(logits, gt)
        self.log("train_loss", loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def get_pred(self, logits):
        if self.as_regression:
            return torch.round(logits).clamp(0, self.n_classes - 1).squeeze(1)
        else:
            return torch.argmax(logits, dim=1)

    def get_loss(self, logits, gt):
        if self.as_regression:
            return self.loss(logits.flatten(), gt.float())
        else:
            return self.loss(logits, gt.long())

    def validation_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        segments = data["segments"]
        gt = data["label"]
        logits, attn = self.model(image, segments, return_attention=True)
        loss = self.get_loss(logits, gt)
        pred = self.get_pred(logits)
        attn = spatial_batch_normalization(attn)
        self.metrics.update(pred, gt)
        self.log_dict(self.metrics, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return {"pred": pred, "attn": attn, "gt": gt}

    def test_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        segments = data["segments"]
        gt = data["label"]
        logits = self.model(image, segments, return_attention=False)
        pred = self.get_pred(logits)
        self.test_metrics.update(pred, gt)
        self.confusion_matrix.update(pred, gt)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False, sync_dist=True)
        
    def on_test_epoch_end(self) -> None:
        confmat = self.confusion_matrix.compute()
        self.confusion_matrix.reset()
        print(confmat)        
        
    def configure_optimizers_old(self) -> Any:
        param_groups = lrd.param_groups_lrd(
            self.model,
            self.training_config["optimizer"]["weight_decay"],
            no_weight_decay_list=self.model.no_weight_decay,
            layer_decay=self.training_config.get("layer_decay", 0.0),
        )
        optimizer = torch.optim.AdamW(param_groups, lr=self.training_config["lr"], **self.training_config["optimizer"])
        return [optimizer], [
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.training_config["lr"],
                    total_steps=self.trainer.estimated_stepping_batches,
                    pct_start=self.training_config["pct_start"],
                ),
                "interval": "step",
            }
        ]

    def configure_optimizers(self) -> Any:
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if name in self.model.no_weight_decay:
                no_decay.append(param)
            else:
                decay.append(param)

        params = [
            {"params": decay, "weight_decay": self.training_config["optimizer"]["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(params, lr=self.training_config["lr"], **self.training_config["optimizer"])
        return [optimizer], [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode="min"),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]


class LogValidationAttentionMap(pl.Callback):
    def __init__(self, wandb_logger, n_images=8, frequency=2):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        self.frequency = frequency

        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 1 and trainer.is_global_zero:
            n = self.n_images
            x = batch["image"][:n].float()
            attn = outputs["attn"][:n]

            mixed = x * F.interpolate(attn.unsqueeze(1), size=x.shape[-2:], mode="nearest")
            pred = outputs["pred"][:n]
            gt = outputs["gt"][:n]
            columns = ["image", "attention_map", 'mixed', "prediction", "groundtruth"]

            data = [
                [wandb.Image(x_i), wandb.Image(attn_i / attn_i.max()), wandb.Image(mixed_i), p_i, gt_i]
                for x_i, attn_i, mixed_i, p_i, gt_i in list(zip(x, attn, mixed, pred, gt))
            ]

            self.wandb_logger.log_table(data=data, key="Validation First Batch", columns=columns)
