from typing import Any, Mapping

import pytorch_lightning
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer

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

        self.lr = self.training_config["lr"]

        self.training_step_can_skip = True

        self.lmbda = 0.01

    def training_step(self, data, batch_index) -> STEP_OUTPUT:
        image = data["image"]
        segments = data["segments"]
        gt = data["label"]
        # kmeans_only = False
        # if self.model.use_kmeans and self.current_epoch == 0:
        #     self.model.set_update_kmeans(True)
        #     kmeans_only = True
        #     with torch.no_grad():
        #         logits, segments, align_loss = self.model(image, segments, return_attention=False)

        # else:
        #     self.model.set_update_kmeans(False)
        #     self.model.use_kmeans = False

        # if kmeans_only:
        #     return torch.tensor(0.0, requires_grad=True)  # Skip training step by returning a dummy loss

        logits, segments, align_loss = self.model(image, segments, return_attention=False)

        self.log("align_loss", align_loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        classif_loss = self.get_loss(logits, gt)
        self.log("train_loss", classif_loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        return classif_loss + self.lmbda * align_loss

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
        logits, segments, align_loss, attn, all_segments = self.model(
            image, data["segments"], return_attention=True, return_all_segments=True
        )
        self.log("val_align_loss", align_loss, on_epoch=True, on_step=True, sync_dist=True, prog_bar=True)
        loss = self.get_loss(logits, gt)
        pred = self.get_pred(logits)
        self.metrics.update(pred, gt)
        self.log_dict(self.metrics, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return {"pred": pred, "attn": attn, "gt": gt, "last_segment": segments}

    def test_step(self, data, batch_index, dataloader_idx=0) -> STEP_OUTPUT:
        self.model.set_update_kmeans(False)
        image = data["image"]
        segments = data["segments"]
        gt = data["label"]
        logits, segments, _ = self.model(image, segments, return_attention=False)
        pred = self.get_pred(logits)
        self.test_metrics.update(pred, gt)
        self.confusion_matrix.update(pred, gt)
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False, sync_dist=True, add_dataloader_idx=False)

    def on_test_epoch_end(self) -> None:
        confmat = self.confusion_matrix.compute()
        self.confusion_matrix.reset()
        self.test_metrics.reset()
        print(confmat)

    def configure_optimizers(self) -> Any:
        param_groups = lrd.param_groups_lrd(
            self.model,
            initial_lr=self.lr,
            weight_decay=self.training_config["optimizer"]["weight_decay"],
            no_weight_decay_list=self.model.no_weight_decay,
            layer_decay=self.training_config.get("layer_decay", 0.99),
        )

        optimizer = torch.optim.AdamW(param_groups, lr=self.training_config["lr"], **self.training_config["optimizer"])
        scheduler_interval = self.training_config["scheduler"].pop("interval", "step")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches, **self.training_config["scheduler"]
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": scheduler_interval,
            }
        ]


class DataDebugger(pytorch_lightning.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        x = self.model(x)
        return x.mean()

    def training_step(self, data, batch_index):
        img = data["image"]

        return self.forward(img)

    def validation_step(self, data, batch_index) -> torch.Tensor | Mapping[str, Any] | None:
        img = data["image"]
        return self.forward(img)

    def test_step(self, data, batch_index) -> torch.Tensor | Mapping[str, Any] | None:
        img = data["image"]
        return self.forward(img)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
