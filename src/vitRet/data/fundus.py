import os

import albumentations as A
import cv2
import numpy as np
import torch
from nntools.dataset import ClassificationDataset, random_split, Composition, nntools_wrapper
from nntools.dataset.utils import class_weighting
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def filter_name(name):
    return name.split('.png')[0]


@nntools_wrapper
def fundus_autocrop(image):
    r_img = image[:, :, 0]
    _, mask = cv2.threshold(r_img, 15, 1, cv2.THRESH_BINARY)

    not_null_pixels = cv2.findNonZero(mask)
    mask = mask.astype(np.uint8)
    if not_null_pixels is None:
        return {'image': image, 'mask': mask}
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
        return {'image': image, 'mask': mask}
    return {'image': image[y_range[0]:y_range[1], x_range[0]:x_range[1]],
            'mask': mask[y_range[0]:y_range[1], x_range[0]:x_range[1]]}


class FundusDataModule(LightningDataModule):
    def __init__(self,
                 data_dir,
                 img_size=(512, 512),
                 valid_size=0.1,
                 batch_size=64,
                 num_workers=32,
                 use_cache=False):
        super(FundusDataModule, self).__init__()
        self.img_size = img_size
        self.root_img = data_dir
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.train = self.val = self.test = None
        if num_workers == 'auto':
            self.num_workers = os.cpu_count() // torch.cuda.device_count()
        else:
            self.num_workers = num_workers
        self.use_cache = use_cache
        self.persistent_workers = True
    def setup(self, stage:str):
        test_composer = Composition()
        test_composer.add(fundus_autocrop, *self.img_size_ops(), *self.normalize_and_cast_op())

        if stage == 'fit' or stage == 'validate':
            train_composer = Composition()

            train_composer.add(fundus_autocrop, *self.img_size_ops(),
                               *self.data_aug_ops(),
                               *self.normalize_and_cast_op())
            self.val.composer = test_composer
            self.train.composer = train_composer
        elif stage == 'test':
            self.test.composer = test_composer


    @property
    def weights(self):
        return torch.Tensor(class_weighting(self.train.get_class_count()))

    def img_size_ops(self):
        return [A.LongestMaxSize(max_size=max(self.img_size), always_apply=True),
                A.PadIfNeeded(min_height=self.img_size[0], min_width=self.img_size[1],
                              always_apply=True, border_mode=cv2.BORDER_CONSTANT)]

    def normalize_and_cast_op(self):
        return [A.Compose([A.Normalize(mean = IMAGENET_INCEPTION_MEAN,
                                      std = IMAGENET_INCEPTION_STD,
                                       always_apply=True),
                           ToTensorV2(always_apply=True)])]

    def data_aug_ops(self):
        return [
            A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=.25),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT),
                A.HueSaturationValue(),
                A.Blur(blur_limit=3, p=0.1),
            ])
        ]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          pin_memory=True)

    def val_dataloader(self, shuffle=True, persistent_workers=True):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=shuffle, 
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers 
                          and persistent_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=False,
                          pin_memory=True)


class EyePACSDataModule(FundusDataModule):
    def setup(self, stage: str) -> None:

        if stage == 'fit' or stage == 'validate':
            dataset = ClassificationDataset(os.path.join(self.root_img, 'train/images/'),
                                            label_filepath = os.path.join(self.root_img, 'trainLabels.csv'),
                                            file_column='image', gt_column='level',
                                            shape=self.img_size, keep_size_ratio=True,
                                            use_cache=False,
                                            auto_pad=True,
                                            extract_image_id_function=filter_name)
            dataset.remap('level', 'label')
            val_length = self.valid_size
            train_length = len(dataset) - val_length
            self.train, self.val = random_split(dataset, [train_length, val_length])
            self.train.composer = None
            self.val.composer = None
            if self.use_cache:
                self.train.use_cache = True
                self.train.init_cache()

        if stage == 'test':
            self.test = ClassificationDataset(os.path.join(self.root_img, 'test/images/'),
                                              shape=self.img_size, keep_size_ratio=True,
                                              file_column='image', gt_column='level',
                                              label_filepath=os.path.join(self.root_img, 'testLabels.csv'),
                                              extract_image_id_function=filter_name
                                              )
            self.test.remap('level', 'label')
            self.test.composer = None

        super().setup(stage)


class AptosDataModule(FundusDataModule):
    def setup(self, stage: str) -> None:

        fold = StratifiedKFold(5, shuffle=True, random_state=2)
        dataset = ClassificationDataset(os.path.join(self.root_img, 'train/'),
                                        csv_filepath=os.path.join(self.root_img, 'train.csv'),
                                        file_column='id_code', gt_column='diagnosis',
                                        shape=self.img_size, keep_size_ratio=True)

        dataset.remap('diagnos', 'label')
        list_index = np.arange(len(dataset))
        list_labels = dataset.gts['label']
        train_index, test_index = next(fold.split(list_index, list_labels))
        if stage == 'fit' or stage == 'validate':
            dataset.subset(np.asarray(train_index))
            val_length = int(len(dataset) * self.valid_size)
            train_length = len(dataset) - val_length
            self.train, self.val = random_split(dataset, [train_length, val_length])
            self.train.composer = Composition()
            self.val.composer = Composition()

        if stage == 'test':
            dataset.subset(np.asarray(test_index))
            self.test = dataset
            self.test.composer = Composition()
        super().setup(stage)
