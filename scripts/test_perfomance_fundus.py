import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from vitRet.data.fundus import AptosDataModule, EyePACSDataModule
from vitRet.utils.ckpts import ModelCkpt, ProjectorCkpt
from vitRet.vit_pl_training_module import TrainerModule

torch.set_float32_matmul_precision("high")


def get_datamodule(img_size, scales, n_segments, aptos=False, batch_size=8):
    if aptos:
        data_dir = "/home/tmp/clpla/data/aptos/"
        module = AptosDataModule
    else:
        data_dir = "/usagers/clpla/data/eyepacs/"
        module = EyePACSDataModule

    datamodule = module(
        data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=12,
        superpixels_scales=scales,
        superpixels_max_nb=6000,
        superpixels_min_nb=n_segments,
        superpixels_filter_black=True,
        valid_size=4000,
    )
    if aptos:
        datamodule.setup("all")
    else:
        datamodule.setup("fit")
        datamodule.setup("test")

    return datamodule


def get_model(ckpt_path, **kwargs):
    model = TrainerModule(**kwargs)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    pos_embed = state_dict["model.projector.pos_embed"]
    state_dict["model.projector.pos_embed"] = F.interpolate(
        pos_embed, size=model.model.projector.pos_embed.shape[-2:], mode="bilinear"
    )
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model


def compressed_embedded_model(ckpt_path, **kwargs):
    model = TrainerModule(**kwargs)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    projector_path = ProjectorCkpt.DEPTH_32
    projector_dict = torch.load(projector_path, map_location="cpu")["state_dict"]
    only_projector = {}
    for k, v in projector_dict.items():
        if k.startswith("trained_projector."):
            only_projector[k.replace("trained_projector", "model")] = v
    state_dict.update(only_projector)
    pos_embed = state_dict["model.projector.pos_embed"]
    state_dict["model.projector.pos_embed"] = F.interpolate(
        pos_embed, size=model.model.projector.pos_embed.shape[-2:], mode="bilinear"
    )

    inc_keys = model.load_state_dict(state_dict=state_dict, strict=True)
    print(inc_keys)
    return model


def test():
    seed_everything(1234, workers=True)
    img_size = (1024, 1024)
    scales = 1
    ntokens = 4096
    datamodule = get_datamodule(img_size, scales, ntokens, aptos=True, batch_size=16)

    ckpt_path = "checkpoints/efficient-wood-282/epoch=3-step=2596.ckpt"
    network_config = {
        "arch": "svt_16_base",
        "img_size": img_size,
        "first_embedding_dim": 32,
        "num_classes": 5,
        "max_tokens": 4096,
        "scales": scales,
        "projection_stride": 1,
        "qkv_bias": True,
        "drop_path": 0.2,
        "global_pool": False,
        "single_cls_token": True,
        "padding": 0,
    }
    training_config = {
        "as_regression": True,
    }
    model = get_model(ckpt_path, network_config=network_config, training_config=training_config)
    model.eval()
    trainer = Trainer(accelerator="auto", devices=1)
    trainer.test(model, dataloaders=datamodule.val_dataloader())

    scores = model.test_metrics.compute()
    confMat = model.confusion_matrix.compute()

    save_dict = {
        "scores": scores,
        "confMat": confMat,
    }
    torch.save(save_dict, f"results/scores_{ntokens}_res{img_size[0]}.pt")


if __name__ == "__main__":
    test()
