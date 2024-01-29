import os
from typing import List

import albumentations as A
import cv2
import numpy as np
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
    _, mask = cv2.threshold(r_img, 15, 1, cv2.THRESH_BINARY)

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
        )
        self.root_img = data_dir

    def setup(self, stage: str):
        test_composer = Composition()
        test_composer.add(fundus_autocrop, *self.img_size_ops(), *self.normalize_and_cast_op())

        if stage == "fit" or stage == "validate":
            train_composer = Composition()

            train_composer.add(
                fundus_autocrop, *self.img_size_ops(), *self.data_aug_ops(), *self.normalize_and_cast_op()
            )
            self.val.composer = test_composer
            self.train.composer = train_composer
        elif stage == "test":
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
                    A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT),
                    A.HueSaturationValue(),
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
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def val_dataloader(self, shuffle=True, persistent_workers=True):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and persistent_workers,
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


class EyePACSDataModule(FundusDataModule):
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
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
            self.test.composer = None

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
        if stage == 'all':
            dataset.remap("diagnosis", "label")
            dataset.composer = Composition()
            self.train = dataset
            self.test = dataset
            self.val = dataset
            super().setup('test')
            super().setup('test')
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
            self.train.remap("diagnosis", "label")
            self.val.remap("diagnosis", "label")

        if stage == "test":
            dataset.subset(np.asarray(test_index))
            self.test = dataset
            self.test.composer = Composition()
            self.test.remap("diagnosis", "label")
        super().setup(stage)


if __name__ == "__main__":
    datamodule = EyePACSDataModule("/usagers/clpla/data/eyepacs/")
    datamodule.setup("fit")

    dataset = datamodule.train
    print(len(dataset))
