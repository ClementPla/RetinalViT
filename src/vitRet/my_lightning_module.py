from typing import Any

import pytorch_lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from kornia.morphology import gradient
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.data.mixup import Mixup
from timm.scheduler.scheduler_factory import create_scheduler_v2

import wandb
from vitRet.models.model_factory import create_model
from vitRet.utils import lr_decay as lrd


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
        logits, segments, align_loss = self.model(image, segments, return_attention=False)
        self.log("align_loss", align_loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        classif_loss = self.get_loss(logits, gt)
        self.log("train_loss", classif_loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return classif_loss + align_loss

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
        gt = data["label"]
        logits, segments, align_loss, attn = self.model(image, data["segments"], return_attention=True)
        self.log("val_align_loss", align_loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        loss = self.get_loss(logits, gt)
        pred = self.get_pred(logits)
        self.metrics.update(pred, gt)
        self.log_dict(self.metrics, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return {"pred": pred, "attn": attn, "gt": gt, "last_segment": segments}

    def test_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        segments = data["segments"]
        gt = data["label"]
        logits, segments, _ = self.model(image, segments, return_attention=False)
        pred = self.get_pred(logits)
        self.test_metrics.update(pred, gt)
        self.confusion_matrix.update(pred, gt)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        confmat = self.confusion_matrix.compute()
        self.confusion_matrix.reset()
        print(confmat)

    def configure_optimizers(self) -> Any:
        param_groups = lrd.param_groups_lrd(
            self.model,
            self.training_config["optimizer"]["weight_decay"],
            no_weight_decay_list=self.model.no_weight_decay,
            layer_decay=self.training_config.get("layer_decay", 0.1),
        )
        optimizer = torch.optim.AdamW(param_groups, lr=self.training_config["lr"], **self.training_config["optimizer"])
        num_epochs = self.trainer.max_epochs
        num_batches = self.trainer.num_training_batches / self.trainer.accumulate_grad_batches
        scheduler_interval = self.training_config["scheduler"].pop("interval", "step")
        step_on_epochs = scheduler_interval == "step"
        # scheduler, _ = create_scheduler_v2(optimizer,
        #                                 num_epochs=num_epochs,
        #                                 updates_per_epoch=num_batches,
        #                                 step_on_epochs=step_on_epochs,
        #                                 **self.training_config["scheduler"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches, **self.training_config["scheduler"]
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": scheduler_interval,
            }
        ]


class LogValidationAttentionMap(pl.Callback):
    def __init__(self, wandb_logger, n_images=8, frequency=2):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        self.frequency = frequency
        self.labels = [
            "non_roi",
            "retina",
            "bright_uncertains",
            "cottonWoolSpots",
            "drusens",
            "exudates",
            "hemorrhages",
            "macula",
            "microaneurysms",
            "neovascularization",
            "vessels",
            "optic_cup",
            "optic_disc",
            "red_uncertains",
        ]
        self.classes_labels = {i: label for i, label in enumerate(self.labels)}
        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 1 and trainer.is_global_zero:
            n = self.n_images
            x = batch["image"][:n].float()
            segments = batch["segments"][:n]

            kernel = segments.new_ones(3, 3)
            border = gradient(segments, kernel) > 0
            border = border.squeeze(1).long().cpu().numpy()

            last_segment = outputs["last_segment"][:n]
            if last_segment.ndim == 3:
                last_segment = last_segment.unsqueeze(1)

            last_border = gradient(last_segment, kernel) > 0
            last_border = last_border.squeeze(1).long().cpu().numpy()

            attn = [a[:n] for a in outputs["attn"]]

            attn = [F.interpolate(a.unsqueeze(1), size=x.shape[-2:], mode="nearest").squeeze(1).cpu().numpy() for a in attn]

            pred = outputs["pred"][:n]
            gt = outputs["gt"][:n]
            columns = ["image", "prediction", "groundtruth"]
            def get_prediction_dict(batch_idx):
                return {f"Prediction scale {i}": {"mask_data": attn[i][batch_idx], "class_labels": self.classes_labels} 
                        for i in range(len(attn))}
            data = [
                [
                    wandb.Image(
                        x[i],
                        masks={
                            "Last superpixels": {
                                "mask_data": last_border[i],
                                "class_labels": {1: "border", 0: "non-border"},
                            },
                            "Superpixels": {
                                "mask_data": border[i], 
                                "class_labels": {1: "border", 0: "non-border"}},
                            **get_prediction_dict(i)
                        },
                    ),
                    pred[i],
                    gt[i],
                ]
                for i in range(x.shape[0])
            ]

            self.wandb_logger.log_table(data=data, key="Validation First Batch", columns=columns)
