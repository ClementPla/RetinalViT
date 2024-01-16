import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from nntools.dataset import (
    ClassificationDataset,
    Composition,
    nntools_wrapper,
    random_split,
)
from torch.utils.data import DataLoader

from vitRet.data.base import BaseDataModule


@nntools_wrapper
def debugger(image, label):
    print(image.shape, label)
    return {"image": image, "label": label}


class ImageNetDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        synset_mapping,
        csv_file,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=64,
        num_workers=32,
        use_cache=False,
        use_superpixels=True,
        superpixels_scales=4,
        superpixels_max_nb=2048):
        
        super(ImageNetDataModule, self).__init__(img_size,
            valid_size,
            batch_size,
            num_workers,
            use_cache,
            use_superpixels,
            superpixels_scales,
            superpixels_max_nb,)
        
        self.root_img = data_dir
        self.csv_file = csv_file
        self.train = self.val = self.test = None

        with open(synset_mapping) as f:
            lines = f.readlines()
        self.labels = {f.split(" ")[0]: (", ".join(f.split(" ")[1:]).replace("\n", "")) for f in lines}

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
                *self.img_size_ops(),
                self.data_augmentation_ops(),
                *self.normalize_and_cast_op(),
            ]
            self.train.composer.add(*transforms)

            transforms = [*self.img_size_ops(), *self.normalize_and_cast_op()]
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
                label_filepath=self.csv_file,
            )
            self.test.remap("GtClassif", "label")
            self.test.composer = Composition()

            transforms = [*self.img_size_ops(), *self.normalize_and_cast_op()]
            self.test.composer.add(*transforms)

    @staticmethod
    def data_augmentation_ops():
        return A.Compose([A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(p=0.5)])

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

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def human_readable_label(self, i: int):
        return self.labels[self.train.map_class['label'][int(i)]]

    def human_readable_labels(self, labels):
        return [self.human_readable_label(label) for label in labels]
