import os
from typing import List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from nntools.dataset import ClassificationDataset, Composition, nntools_wrapper, random_split
from nntools.dataset.utils import class_weighting
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from vitRet.data.base import BaseDataModule


def filter_name(name: str):
    return name.split(".png")[0]


@nntools_wrapper
def fundus_autocrop(image: np.ndarray):
    r_img = image[:, :, 0]
    _, mask = cv2.threshold(r_img, 25, 1, cv2.THRESH_BINARY)
    not_null_pixels = cv2.findNonZero(mask)
    mask = mask.astype(np.uint8)
    if not_null_pixels is None:
        return {"image": image, "mask": mask}
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
        return {"image": image, "mask": mask}
    return {
        "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
        "mask": mask[y_range[0] : y_range[1], x_range[0] : x_range[1]],
    }


class FundusDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=64,
        num_workers=32,
        use_cache=False,
        use_superpixels=True,
        superpixels_scales=4,
        superpixels_max_nb=2048,
        superpixels_min_nb=32,
        superpixels_filter_black=True,
        superpixels_num_threads=1,
        superpixels_compactness=10,
    ):
        super(FundusDataModule, self).__init__(
            img_size,
            valid_size,
            batch_size,
            num_workers,
            use_cache,
            use_superpixels,
            superpixels_scales,
            superpixels_max_nb,
            superpixels_min_nb=superpixels_min_nb,
            superpixels_filter_black=superpixels_filter_black,
            superpixels_num_threads=superpixels_num_threads,
            superpixels_compactness=superpixels_compactness
        )
        self.root_img = data_dir

    def setup(self, stage: str):
        test_composer = Composition()
        test_composer.add(fundus_autocrop, *self.img_size_ops(), *self.normalize_and_cast_op())
        train_composer = Composition()
        train_composer.add(fundus_autocrop, *self.img_size_ops(), *self.data_aug_ops(), *self.normalize_and_cast_op())
        if self.train:
            self.train.composer = train_composer
        if self.val:
            self.val.composer = test_composer
        if self.test:
            self.test.composer = test_composer

    @property
    def weights(self):
        return torch.Tensor(class_weighting(self.train.get_class_count()))

    def data_aug_ops(self) -> List[A.Compose]:
        return [
            A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.25),
                    A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, scale_limit=0, rotate_limit=10, p=0.25),
                    A.HueSaturationValue(val_shift_limit=10, hue_shift_limit=10, sat_shift_limit=10),
                    A.Blur(blur_limit=3, p=0.1),
                ]
            )
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self, shuffle=True, persistent_workers=True):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and persistent_workers and self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )


class DDRDataModule(FundusDataModule):
    def setup(self, stage: str):
        if stage == "fit":
            df = pd.read_csv(os.path.join(self.root_img, "train.txt"), sep=" ", names=["image", "label"])
            df = df[df["label"] != 5]
            dataset = ClassificationDataset(
                os.path.join(self.root_img, "train/"),
                label_dataframe=df,
                shape=self.img_size,
                keep_size_ratio=True,
                use_cache=False,
                auto_pad=True,
            )
            self.train = dataset
            if self.use_cache:
                self.train.use_cache = True
                self.train.init_cache()
        if stage == "validate":
            df = pd.read_csv(os.path.join(self.root_img, "valid.txt"), sep=" ", names=["image", "label"])
            df = df[df["label"] != 5]

            dataset = ClassificationDataset(
                os.path.join(self.root_img, "valid/"),
                label_dataframe=df,
                shape=self.img_size,
                keep_size_ratio=True,
                use_cache=False,
                auto_pad=True,
            )
            self.val = dataset
            if self.use_cache:
                self.val.use_cache = True
                self.val.init_cache()
        if stage == "test":
            df = pd.read_csv(os.path.join(self.root_img, "test.txt"), sep=" ", names=["image", "label"])
            df = df[df["label"] != 5]

            dataset = ClassificationDataset(
                os.path.join(self.root_img, "test/"),
                label_dataframe=df,
                shape=self.img_size,
                keep_size_ratio=True,
                use_cache=False,
                auto_pad=True,
            )
            self.test = dataset
            if self.use_cache:
                self.test.use_cache = True
                self.test.init_cache()
        super().setup(stage)


class IDRiDDataModule(FundusDataModule):
    def setup(self, stage: str):
        if stage in ["fit", "validate"]:
            label_filepath = os.path.join(self.root_img, "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
            dataset = ClassificationDataset(
                os.path.join(self.root_img, "1. Original Images/a. Training Set/"),
                label_filepath=label_filepath,
                file_column="Image name",
                gt_column="Retinopathy grade",
                shape=self.img_size,
                keep_size_ratio=True,
                use_cache=False,
                auto_pad=True,
            )
            if isinstance(self.valid_size, float):
                self.valid_size = int(len(dataset) * self.valid_size)

            val_length = self.valid_size
            train_length = len(dataset) - val_length
            self.train, self.val = random_split(dataset, [train_length, val_length])
            self.train.remap("Retinopathy grade", "label")
            self.val.remap("Retinopathy grade", "label")
        if stage == "test":
            label_filepath = os.path.join(self.root_img, "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
            self.test = ClassificationDataset(
                os.path.join(self.root_img, "1. Original Images/b. Testing Set/"),
                shape=self.img_size,
                keep_size_ratio=True,
                file_column="Image name",
                gt_column="Retinopathy grade",
                label_filepath=label_filepath,
            )
            self.test.remap("Retinopathy grade", "label")
            self.test.composer = None
        super().setup(stage)


class EyePACSDataModule(FundusDataModule):
    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            dataset = ClassificationDataset(
                os.path.join(self.root_img, "train/images/"),
                label_filepath=os.path.join(self.root_img, "trainLabels.csv"),
                file_column="image",
                gt_column="level",
                shape=self.img_size,
                keep_size_ratio=True,
                use_cache=False,
                auto_pad=True,
                extract_image_id_function=filter_name,
            )

            if isinstance(self.valid_size, float):
                self.valid_size = int(len(dataset) * self.valid_size)

            val_length = self.valid_size
            train_length = len(dataset) - val_length
            self.train, self.val = random_split(dataset, [train_length, val_length])
            self.train.remap("level", "label")
            self.val.remap("level", "label")

            if self.use_cache:
                self.train.use_cache = True
                self.train.init_cache()

        if stage == "test":
            self.test = ClassificationDataset(
                os.path.join(self.root_img, "test/images/"),
                shape=self.img_size,
                keep_size_ratio=True,
                file_column="image",
                gt_column="level",
                label_filepath=os.path.join(self.root_img, "testLabels.csv"),
                extract_image_id_function=filter_name,
            )
            self.test.remap("level", "label")

        super().setup(stage)


class AptosDataModule(FundusDataModule):
    def setup(self, stage: str) -> None:
        dataset = ClassificationDataset(
            os.path.join(self.root_img, "train/"),
            label_filepath=os.path.join(self.root_img, "train.csv"),
            file_column="id_code",
            gt_column="diagnosis",
            shape=self.img_size,
            keep_size_ratio=True,
            auto_pad=True,
        )
        dataset.remap("diagnosis", "label")
        if stage == "all":
            dataset.composer = Composition()
            self.train = dataset
            self.test = dataset
            self.val = dataset
            super().setup("test")
            return

        
        fold = StratifiedKFold(5, shuffle=True, random_state=2)
        list_index = np.arange(len(dataset))
        list_labels = dataset.gts["label"]
        train_index, test_index = next(fold.split(list_index, list_labels))
        if stage == "fit" or stage == "validate":
            dataset.subset(np.asarray(train_index))
            val_length = int(len(dataset) * self.valid_size)
            train_length = len(dataset) - val_length
            self.train, self.val = random_split(dataset, [train_length, val_length])
            self.train.composer = Composition()
            self.val.composer = Composition()

        if stage == "test":
            dataset.subset(np.asarray(test_index))
            self.test = dataset
            self.test.composer = Composition()
        super().setup(stage)


if __name__ == "__main__":
    datamodule = EyePACSDataModule("/usagers/clpla/data/eyepacs/")
    datamodule.setup("validate")

    dataset = datamodule.val
    print(dataset.composer)
    print(dataset[0].keys())