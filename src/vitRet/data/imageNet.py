from nntools.dataset import (
    ClassificationDataset,
    random_split,
    Composition,
    nntools_wrapper,
)
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from timm.data.constants import IMAGENET_INCEPTION_STD, IMAGENET_INCEPTION_MEAN
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)


@nntools_wrapper
def debugger(image, label):
    print(image.shape, label)
    return {"image": image, "label": label}


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        synset_mapping,
        csv_file,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=64,
        num_workers=32,
    ):
        super(ImageNetDataModule, self).__init__()
        self.img_size = img_size
        self.root_img = data_dir
        self.csv_file = csv_file
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.train = self.val = self.test = None

        with open(synset_mapping) as f:
            lines = f.readlines()
        self.labels = {
            f.split(" ")[0]: (", ".join(f.split(" ")[1:]).replace("\n", ""))
            for f in lines
        }
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            dataset = ClassificationDataset(
                os.path.join(self.root_img, "train/"),
                shape=self.img_size,
                keep_size_ratio=True,
            )

            val_length = int(len(dataset) * self.valid_size)
            train_length = len(dataset) - val_length
            self.train, self.val = random_split(dataset, [train_length, val_length])
            self.train.composer = Composition()
            self.val.composer = Composition()
            transforms = [
                self.img_size_ops(),
                self.data_augmentation_ops(),
                self.normalize_ops(),
            ]
            self.train.composer.add(*transforms)

            transforms = [self.img_size_ops(), self.normalize_ops()]
            self.val.composer.add(*transforms)

            print(f"Train set: {len(self.train)}")
            print(f"Val set: {len(self.val)}")

        if stage == "test":
            self.test = ClassificationDataset(
                os.path.join(self.root_img, "val/"),
                shape=self.img_size,
                keep_size_ratio=True,
                file_column="ImageId",
                gt_column="GtClassif",
                csv_filepath=self.csv_file,
            )
            self.test.remap("GtClassif", "label")
            self.test.composer = Composition()

            transforms = [self.img_size_ops(), self.normalize_ops()]
            self.test.composer.add(*transforms)

    @staticmethod
    def data_augmentation_ops():
        return A.Compose([A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(p=0.5)])

    @staticmethod
    def normalize_ops():
        return A.Compose(
            [
                A.Normalize(
                    always_apply=True,
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
                ToTensorV2(),
            ]
        )

    def img_size_ops(self):
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(self.img_size), always_apply=True),
                A.PadIfNeeded(
                    min_height=self.img_size[0],
                    min_width=self.img_size[1],
                    always_apply=True,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def human_readable_label(self, i: int):
        return self.labels[self.train.map_class[int(i)]]

    def human_readable_labels(self, labels):
        return [self.human_readable_label(label) for label in labels]
