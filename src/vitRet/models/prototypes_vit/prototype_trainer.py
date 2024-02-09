import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import functional as F
from torch_scatter import scatter, scatter_add

import wandb


class PrototypeTrainer(LightningModule):
    def __init__(self, n_protos, features_extractor: str, weight=None, **config_training):
        super().__init__()
        self.n_protos = n_protos
        self.features_extractor = timm.create_model(features_extractor, pretrained=True, features_only=True)
        # self.features_extractor.requires_grad_(False)
        self.layer_extracted = -2
        f = self.features_extractor.feature_info.channels()[self.layer_extracted]
        self.prototypes = torch.nn.Parameter(
            torch.randn(n_protos, f)
        )
        self.loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.lr = config_training.get("lr", 1e-3)
        self.wc = config_training.get("weight_decay", 1e-4)

    def get_superpixels_groundtruth(self, gt_mask: torch.Tensor, superpixels_segment):
        """
        superpixels_segment: (B, H, W)
        gt_mask: (B, n, H, W)
        """
        
        # We get the sum of each groundtruth overlapping each segment
        output = scatter_add(gt_mask.flatten(-2), superpixels_segment.unsqueeze(1).flatten(2), dim=-1)
        gt_segment = output.argmax(1, keepdim=True)
        gt_segment = torch.zeros_like(output).scatter_(1, gt_segment, 1)
        return gt_segment.long()
    
    def forward(self, x, superpixels_segment, gt_mask, return_pred=False):
        self.features_extractor.eval()
        features = self.features_extractor(x)[self.layer_extracted]
        # features.detach_()
        features = F.interpolate(features, size=x.shape[-2:], mode="bilinear")
        superpixels_segment = F.interpolate(
            superpixels_segment.unsqueeze(1).float(), size=features.shape[-2:], mode="nearest"
        ).long()
        _, superpixels_segment = torch.unique(superpixels_segment, return_inverse=True)
        gt_mask = F.interpolate(gt_mask.float(), size=features.shape[-2:], mode="nearest").long()
        superpixels_gt = self.get_superpixels_groundtruth(gt_mask, superpixels_segment)
        superfeatures = scatter(
            features.flatten(-2), superpixels_segment.flatten(-2), dim=-1, reduce="mean"
        )  # B x C x N
        superfeatures = F.normalize(superfeatures, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        compatibility_matrix = torch.matmul(prototypes, superfeatures)
        return compatibility_matrix, superpixels_gt, superpixels_segment

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        superpixels_segment = batch["segments"]
        gts = batch["label"]
        cmat, gt, segment = self(image, superpixels_segment, gts)
        loss = self.loss(cmat, gt.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        superpixels_segment = batch["segments"]
        gts = batch["label"]
        cmat, gt, segment = self(image, superpixels_segment, gts)
        loss = self.loss(cmat, gt.float())
        output = {"loss": loss}

        if batch_idx < 1:
            binary_proba = torch.sigmoid(cmat)
            bg_proba = 1-torch.amax(binary_proba, 1, keepdim=True)
            cmat = torch.cat((bg_proba, binary_proba), 1)
            pred = torch.gather(cmat.argmax(1)-1, 1, segment.flatten(1)).view(-1, *segment.shape[-2:])
            output["pred"] = pred
        self.log("val_loss", loss)
        return output

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wc)
        return optim


class LogValidationPredictedPrototypeMap(pl.Callback):
    def __init__(self, wandb_logger, n_images=2, frequency=2):
        self.n_images = n_images
        self.wandb_logger = wandb_logger

        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 1 and trainer.is_global_zero:
            n = self.n_images
            x = batch["image"][:n].float()
            x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear")
            gt = batch["label"][:n]
            gt = torch.nn.functional.interpolate(gt.float(), size=(512, 512), mode="nearest").long()
            bg = torch.amax(gt, 1, keepdim=True)==0
            gt = torch.cat([bg, gt], 1)
            gt = gt.argmax(1)
            pred = outputs["pred"][:n]
            pred = (
                torch.nn.functional.interpolate(pred.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest")
                .long()
                .squeeze(1)
            )
            columns = ["image"]

            labels = [
                'background',
                'bright_uncertains',
                'cottonWoolSpots',
                'drusens',
                'exudates',
                'hemorrhages',
                'macula',
                'microaneurysms',
                'neovascularization',
                'vessels',
                'red_uncertains',
                'optic_cup',
                'optic_disc',
            ]
            labels = {i: l for i, l in enumerate(labels)}

            data = [
                [
                    wandb.Image(
                        x_i,
                        masks={
                            "Prediction": {
                                "mask_data": p_i.cpu().numpy().astype("uint8"),
                                "class_labels": labels,
                            },
                            "Groundtruth": {
                                "mask_data": gt_i.cpu().numpy().astype("uint8"),
                                "class_labels": labels,
                            },
                        },
                    ),
                ]
                for x_i, gt_i, p_i in list(zip(x, gt, pred))
            ]

            self.wandb_logger.log_table(data=data, key="Validation First Batch", columns=columns)


if __name__ == "__main__":
    from vitRet.data.segmentation import MaplesDR

    data_dir = "/usagers/clpla/data/Maples-DR/"
    datamodule = MaplesDR(
        data_dir=data_dir,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=2,
        num_workers=4,
        use_superpixels=True,
        superpixels_nb=2048,
        superpixels_filter_black=True,
        superpixels_num_threads=1,
    )

    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()
    model = PrototypeTrainer(n_protos=512, features_extractor="resnet34")
    for batch in dataloader:
        loss, _ = model.validation_step(batch, 0)
        print(loss)
        break
