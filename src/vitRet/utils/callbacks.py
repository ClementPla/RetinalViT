import pytorch_lightning as pl
from kornia.morphology import gradient
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import functional as F

import wandb


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
            segments = batch["segments"][:n]
            if segments.ndim == 3:
                segments = segments.unsqueeze(1)

            kernel = segments.new_ones(3, 3)
            border = gradient(segments, kernel) > 0
            border = border.squeeze(1).long().cpu().numpy()

            last_segment = outputs["last_segment"][:n]
            if last_segment.ndim == 3:
                last_segment = last_segment.unsqueeze(1)

            last_border = gradient(last_segment, kernel) > 0
            last_border = last_border.squeeze(1).long().cpu().numpy()

            attn = [a[:n] for a in outputs["attn"]]

            attn = [
                F.interpolate(a.unsqueeze(1), size=x.shape[-2:], mode="nearest").squeeze(1).cpu().numpy() for a in attn
            ]

            pred = outputs["pred"][:n]
            gt = outputs["gt"][:n]
            columns = ["image", "prediction", "groundtruth"]

            def get_prediction_dict(batch_idx):
                return {f"Prediction scale {i}": {"mask_data": attn[i][batch_idx]} for i in range(len(attn))}

            data = [
                [
                    wandb.Image(
                        x[i],
                        masks={
                            "Last superpixels": {
                                "mask_data": last_border[i],
                                "class_labels": {1: "border", 0: "non-border"},
                            },
                            "Superpixels": {"mask_data": border[i], "class_labels": {1: "border", 0: "non-border"}},
                            **get_prediction_dict(i),
                        },
                    ),
                    pred[i],
                    gt[i],
                ]
                for i in range(x.shape[0])
            ]

            self.wandb_logger.log_table(data=data, key="Validation First Batch", columns=columns)
