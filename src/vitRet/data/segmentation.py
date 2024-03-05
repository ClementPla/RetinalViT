import logging
import os
from typing import Callable, List, Union

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from fast_slic.avx2 import SlicAvx2 as Slic
from nntools.dataset import Composition, SegmentationDataset, nntools_wrapper, random_split
from pytorch_lightning import LightningDataModule
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader


@nntools_wrapper
def concat_mask(
    bright_uncertains,
    cottonWoolSpots,
    drusens,
    exudates,
    hemorrhages,
    macula,
    microaneurysms,
    neovascularization,
    optic_cup,
    optic_disc,
    red_uncertains,
    vessels,
):
    masks = [
        bright_uncertains,
        cottonWoolSpots,
        drusens,
        exudates,
        hemorrhages,
        macula,
        microaneurysms,
        neovascularization,
        vessels,
        optic_cup,
        optic_disc,
        red_uncertains,
    ]
    label = np.asarray(masks).transpose(1, 2, 0)
    return {"label": label}

@nntools_wrapper
def fundus_autocrop(image: np.ndarray, label: np.ndarray):
    r_img = image[:, :, 0]
    _, mask = cv2.threshold(r_img, 25, 1, cv2.THRESH_BINARY)
    not_null_pixels = cv2.findNonZero(mask)
    mask = mask.astype(np.uint8)
    if not_null_pixels is None:
        return {"image": image, "mask": mask}
    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))
    if (x_range[0] == x_range[1]) or (y_range[0] == y_range[1]):
        return {"image": image, "mask": mask, "label": label}
    return {
        "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
        "mask": mask[y_range[0] : y_range[1], x_range[0] : x_range[1]],
        "label": label[y_range[0] : y_range[1], x_range[0] : x_range[1]],
    }
    
    
class MaplesDR(LightningDataModule):
    def __init__(
        self,
        data_dir,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=64,
        num_workers=32,
        use_superpixels=True,
        superpixels_nb=2048,
        superpixels_filter_black=True,
        superpixels_num_threads=1,
        superpixels_convert_to_lab=True,
        superpixels_manhattan_spatial_dist=False,
        superpixels_compactness=1,
        superpixels_min_size_factor=0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.use_superpixels = use_superpixels
        self.superpixels_filter_black = superpixels_filter_black
        self.persistent_workers = True
        
        
        self.num_workers = num_workers
        
        self.slic = Slic(
                num_components=superpixels_nb,
                convert_to_lab=superpixels_convert_to_lab,
                min_size_factor=superpixels_min_size_factor,
                manhattan_spatial_dist=superpixels_manhattan_spatial_dist,
                compactness=superpixels_compactness,
                num_threads=superpixels_num_threads,
            )
        

        self.labels = [
            "bright_uncertains",
            "cottonWoolSpots",
            "drusens",
            "exudates",
            "hemorrhages",
            "macula",
            "microaneurysms",
            "neovascularization",
            "vessels",
            "optic_cup",
            "optic_disc",
            "red_uncertains",
        ]

    def setup(self, stage: str) -> None:
        
        if stage in ("fit", "validate"):
            path = os.path.join(self.data_dir, "train", "fundus")
            mask_root = {label: os.path.join(self.data_dir, "train", label) for label in self.labels}
            ops = [concat_mask, fundus_autocrop, *self.img_size_ops(), *self.normalize_and_cast_op()]
            dataset = SegmentationDataset(
                img_root=path,
                mask_root=mask_root,
                shape=self.img_size,
                binarize_mask=True,
            )
            composer = Composition()
            composer.add(*ops)
            dataset.composer = composer
            self.train, self.val = random_split(dataset, [1 - self.valid_size, self.valid_size])
            
        elif stage == "test":
            path = os.path.join(self.data_dir, "test", "fundus")
            mask_root = {label: os.path.join(self.data_dir, "test", label) for label in self.labels}
            ops = [concat_mask, fundus_autocrop, *self.img_size_ops(), *self.normalize_and_cast_op()]
            dataset = SegmentationDataset(
                img_root=path,
                mask_root=mask_root,
                shape=self.img_size,
                binarize_mask=True,
            )
            composer = Composition()
            composer.add(*ops)
            dataset.composer = composer
            self.test = dataset
    
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
        
    def get_superpixels_decomposition(self):
        @nntools_wrapper
        def get_superpixels(image, mask=None):
            output = {"image": image}
            image = cv2.medianBlur(image, 9)
            segments = self.slic.iterate(image)
            if mask is not None:
                if self.superpixels_filter_black:
                    segments = mask * segments
                output["mask"] = mask
            output["segments"] = segments.astype(int)
            return output
        

        return get_superpixels
    
    def img_size_ops(self) -> List[A.Compose]:
        return [
            A.Compose(
                [
                    A.LongestMaxSize(max_size=max(self.img_size), always_apply=True),
                    A.PadIfNeeded(
                        min_height=self.img_size[0],
                        min_width=self.img_size[1],
                        always_apply=True,
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                ],
                additional_targets={"mask": "mask", "label": "mask"},
            )
        ]
    
    def normalize_and_cast_op(self) -> List[Union[A.Compose, Callable]]:
        ops = []
        additional_targets = None
        if self.use_superpixels:
            ops.append(self.get_superpixels_decomposition())
            additional_targets = {"segments": "mask", "label": "mask"}
        ops.append(
            A.Compose(
                [
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, always_apply=True),
                    ToTensorV2(always_apply=True, transpose_mask=True),
                ],
                additional_targets=additional_targets,
            )
        )
        return ops
    

if __name__=='__main__':
    
    data_dir = '/home/tmp/clpla/data/Maples-DR/'
    datamodule = MaplesDR(data_dir=data_dir, 
                          img_size=(512, 512), 
                          valid_size=0.1, batch_size=64, 
                          num_workers=32, 
                          use_superpixels=True, 
                          superpixels_nb=2048, 
                          superpixels_filter_black=True, superpixels_num_threads=1)
    
    datamodule.setup('fit')
    b = datamodule.train[0]
    print(b['image'].shape, b['mask'].shape, b['label'].shape, b['segments'].shape)