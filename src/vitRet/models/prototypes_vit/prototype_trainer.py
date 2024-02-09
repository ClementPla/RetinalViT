import kornia as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import timm
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import functional as F
from torch_scatter import scatter, scatter_add

import wandb


class PrototypeTrainer(LightningModule):
    def __init__(
        self,
        n_protos,
        n_head: int = 24,
        features_extractor: str = "resnet34",
        detach_features: bool = False,
        weight=None,
        **config_training,
    ):
        super().__init__()
        self.n_protos = n_protos
        self.detach_features = detach_features
        self.features_extractor = timm.create_model(features_extractor, pretrained=True, features_only=True)
        self.features_extractor.requires_grad_(not self.detach_features)
        self.layer_extracted = -1
        f = self.features_extractor.feature_info.channels()[self.layer_extracted]
        self.n_head = n_head
        self.prototypes = torch.nn.Parameter(torch.randn(self.n_head, n_protos, f))
        self.ce_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.dice_loss = DiceLoss(mode="multilabel", from_logits=False)
        self.lr = config_training.get("lr", 1e-3)
        self.wc = config_training.get("weight_decay", 1e-4)

    def get_superpixels_groundtruth(self, gt_mask: torch.Tensor, superpixels_segment):
        """
        superpixels_segment: (B, H, W)
        gt_mask: (B, C, H, W)
        """
        # We get the sum of each groundtruth overlapping each segment
        output = scatter_add(gt_mask.flatten(-2), superpixels_segment.unsqueeze(1).flatten(2), dim=-1)
        mask = output.max(1, keepdim=True).values > 20
        gt_segment = output.argmax(1, keepdim=True)
        gt_segment = torch.zeros_like(output).scatter_(1, gt_segment, 1) * mask
        gt_segment[:, :, 0] = 0
        return gt_segment.long()
    
    def loss(self, y, gt):
        loss = 0.05*self.ce_loss(y, gt) + self.dice_loss(y, gt)
        return loss

    def forward(self, x, superpixels_segment, gt_mask, return_pred=False):
        self.features_extractor.eval()
        features = self.features_extractor(x)[self.layer_extracted]
        if self.detach_features:
            features = features.detach()
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
        B, _, N = superfeatures.shape
        superfeatures = F.normalize(superfeatures, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        compatibility_matrix = torch.matmul(prototypes.flatten(0, 1), superfeatures)
        compatibility_matrix = compatibility_matrix.view(B, self.n_head, self.n_protos, -1)
        compatibility_matrix = (compatibility_matrix + 1) / 2
        compatibility_matrix = compatibility_matrix.mean(1)
        return compatibility_matrix, superpixels_gt, superpixels_segment

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        superpixels_segment = batch["segments"]
        gts = batch["label"]
        cmat, gt, segment = self(image, superpixels_segment, gts)
        loss = self.loss(cmat, gt.float())
        self.log("train_loss", loss, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def from_segment_to_image(self, x, segment):
        return torch.gather(x, 1, segment.flatten(1)).view(-1, *segment.shape[-2:])

    def multilabel_to_multiclass(self, multilabel):
        if multilabel.dtype == torch.long:
            bg = torch.amax(multilabel, 1, keepdim=True) == 0
            fg = torch.cat([bg, multilabel], 1)
            multiclass = fg.argmax(1)
            return multiclass
        else:
            binary_proba = multilabel
            bg_proba = 1 - torch.amax(binary_proba, 1, keepdim=True)
            multiclass = torch.cat([bg_proba, binary_proba], 1)
            return multiclass

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        superpixels_segment = batch["segments"]
        gts = batch["label"].long()
        cmat, gt, segment = self(image, superpixels_segment, gts)

        # gt_multiclass = self.multilabel_to_multiclass(gt)
        # gt_img = self.from_segment_to_image(gt_multiclass, segment)
        # kernel = gts.new_ones(3,3)
        # gradient = K.morphology.gradient(superpixels_segment.float().unsqueeze(1), kernel=kernel) > 0
        # fig, axs = plt.subplots(1, 2)
        # axs[1].imshow(gradient[0].cpu().numpy().squeeze(), vmin=0, vmax=1, cmap="gray")
        # axs[1].imshow(gt_img[0].cpu().numpy(), vmin=0, vmax=13, cmap="RdYlGn", alpha=0.75)
        # axs[0].imshow(self.multilabel_to_multiclass(gts)[0].squeeze().cpu(), vmin=0, vmax=13, cmap="RdYlGn")
        # plt.show()

        loss = self.loss(cmat, gt.float())
        output = {"loss": loss}

        if batch_idx < 1:
            cmat = self.multilabel_to_multiclass(cmat)
            pred = self.from_segment_to_image(cmat.argmax(1), segment)
            gt_multiclass = self.multilabel_to_multiclass(gt)
            gt_segment = self.from_segment_to_image(gt_multiclass, segment)
            output["pred"] = pred
            output["gt_segment"] = gt_segment

        self.log("val_loss", loss, sync_dist=True)
        return output

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wc, fused="fused")
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

            gt = outputs["gt_segment"][:n]
            gt = (
                torch.nn.functional.interpolate(gt.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest")
                .long()
                .squeeze(1)
            )
            pred = outputs["pred"][:n]
            pred = (
                torch.nn.functional.interpolate(pred.unsqueeze(1).float(), size=x.shape[-2:], mode="nearest")
                .long()
                .squeeze(1)
            )

            columns = ["image"]

            labels = [
                "background",
                "bright_uncertains",
                "cottonWoolSpots",
                "drusens",
                "exudates",
                "hemorrhages",
                "macula",
                "microaneurysms",
                "neovascularization",
                "vessels",
                "red_uncertains",
                "optic_cup",
                "optic_disc",
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
    data_dir = "/home/clement/Documents/data/Maples-DR/"
    datamodule = MaplesDR(
        data_dir=data_dir,
        img_size=(1024, 1024),
        valid_size=0.1,
        batch_size=3,
        num_workers=4,
        use_superpixels=True,
        superpixels_nb=8192,
        superpixels_filter_black=True,
        superpixels_num_threads=1,
    )

    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()
    model = PrototypeTrainer(n_protos=12, n_head=24, detach_features=True, features_extractor="resnet34").cuda()
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model.validation_step(batch, 0)
        break
