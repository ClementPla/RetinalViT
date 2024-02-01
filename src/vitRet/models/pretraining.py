from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from vitRet.models.stochastic_attention.tokenization import MultiScaleTokenization, Projector
from vitRet.utils.ckpts import ModelCkpt


class TokenizerWithProjector(torch.nn.Module):
    def __init__(
        self,
        in_chans=3,
        img_size=512,
        padding: int = 0,
        first_embedding_dim=768,
        single_cls_token: bool = True,
        embed_dim: int = 768,
        kernel_size: int = 16,
        projection_stride: int = 16,
        scales=1,
        global_pool=False,
    ):
        super().__init__()
        self.projector = Projector(
            in_chans=in_chans,
            img_size=img_size,
            kernel_size=kernel_size,
            stride=projection_stride,
            padding=padding,
            out_chans=first_embedding_dim,
        )

        self.tokenizer = MultiScaleTokenization(
            input_dim=first_embedding_dim,
            embed_dim=embed_dim,
            scales=scales,
            single_cls_token=single_cls_token,
            global_pool=global_pool,
        )

    def forward(self, img, segments):
        x = self.projector(img)
        seqs, segments = self.tokenizer(x, segments)
        return seqs[0]


class PretrainModule(LightningModule):
    def __init__(self, lr=0.003) -> None:
        super().__init__()

        self.reference_projector = TokenizerWithProjector(projection_stride=4, 
                                                          img_size=1024)

        ckpt_path = ModelCkpt.STANDARD
        state_dict = torch.load(ckpt_path)["state_dict"]
        state_dict["model.tokenizer.cls_pos_embed"] 
        state_dict["model.tokenizer.cls_token"].squeeze_(1)
        for k in list(state_dict.keys()):
            state_dict[k.replace("model.", '')] = state_dict.pop(k, None)
        _ = self.reference_projector.load_state_dict(state_dict, strict=False)
        self.trained_projector = TokenizerWithProjector(first_embedding_dim=32, projection_stride=4, img_size=1024)
        self.loss = nn.MSELoss()
        self.lr = lr
    
    def forward(self, img, segments):
        with torch.no_grad():
            gts_seq = self.reference_projector(img, segments)
        pred_seq = self.trained_projector(img, segments)
        return pred_seq, gts_seq
        
    def training_step(self, batch, batch_idx):
        img = batch["image"]
        segments = batch["segments"]
        pred_seq, gts_seq = self(img, segments)
        loss = self.loss(pred_seq, gts_seq)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        segments = batch["segments"]
        pred_seq, gts_seq = self(img, segments)
        loss = self.loss(pred_seq, gts_seq)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        parameters = list(self.trained_projector.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        return optimizer
