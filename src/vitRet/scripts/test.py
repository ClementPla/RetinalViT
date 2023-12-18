from nntools.utils import Config
from trainer import TrainerModule
from retina_data.datamodules import EyePACSDataModule
import torchmetrics
from tqdm import tqdm
import torch
import os
import torchvision
from pytorch_lightning import seed_everything


if __name__ == "__main__":
    seed_everything(1234, workers=True)

    config = Config("config.yaml")
    # img_size = [512, 512]
    # config["data"]["img_size"] = img_size
    # config["model"]["img_size"] = img_size
    # config["model"]["scales"] = 5
    # config['data']['data_dir'] = '/usagers/clpla/data/eyepacs'
    
    eyepacs_datamodule = EyePACSDataModule(**config["data"])
    eyepacs_datamodule.persistent_workers = False
    # model = TrainerModule(config["model"], config["training"])

    config["model"]["max_tokens"] = 1024
    model = TrainerModule.load_from_checkpoint(
        "checkpoints/dazzling-darkness-72/epoch=65-step=14322.ckpt",
        network_config=config["model"],
        training_config=config["training"],
    )

    model.eval()
    model = model.cuda()
    eyepacs_datamodule.setup("test")
    valid_dataloader = eyepacs_datamodule.test_dataloader()

    model.model.scales = 6
    kappa = torchmetrics.CohenKappa(
        num_classes=config["model"]["num_classes"],
        task="multiclass",
        weights="quadratic",
    ).cuda()
    os.makedirs("_tmp_images", exist_ok=True)
    os.makedirs("_tmp_attn", exist_ok=True)
    os.makedirs("_tmp_mixed", exist_ok=True)

    min_grade = 3
    for i, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
        with torch.inference_mode():
            batch = eyepacs_datamodule.transfer_batch_to_device(batch, "cuda", 0)
            # batch['image'] = batch['image'].repeat(4,1,1,1)
            # batch['label'] = batch['label'].repeat(4)
            
            results = model.validation_step(batch, i)
            pred = results["pred"]
            gt = results["gt"]
            attn = results["attn"]
            kappa.update(pred, gt)
            # images = batch["image"]
            # images = images[gt >= min_grade]
            # attn = attn[gt >= min_grade]
            # attn = torch.nn.functional.interpolate(
            #     attn.unsqueeze(1), size=(images.shape[2], images.shape[3])
            # )
            # if images.shape[0] > 0:
            #     torchvision.utils.save_image(
            #         images,
            #         f"_tmp_images/batch_{i}.png",
            #         nrow=2,
            #         normalize=True,
            #         scale_each=True,
            #     )
            #     torchvision.utils.save_image(
            #         attn,
            #         f"_tmp_attn/batch_{i}.png",
            #         nrow=2,
            #         normalize=True,
            #         scale_each=True,
            #     )
            #     mixed = images*torch.clamp(attn, 0, 1)
            #     torchvision.utils.save_image(
            #         mixed,
            #         f"_tmp_mixed/batch_{i}.png",
            #         nrow=2,
            #         normalize=True,
            #         scale_each=True,
            #     )
            # if i > 25:
            #     break

    print(kappa.compute())
