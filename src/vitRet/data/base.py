import os
from typing import Callable, List, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from fast_slic.avx2 import SlicAvx2 as Slic
from nntools.dataset import nntools_wrapper
from pytorch_lightning import LightningDataModule
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=64,
        num_workers=32,
        use_cache=False,
        use_superpixels=True,
        superpixels_scales=4,
        superpixels_max_nb=2048,
        superpixels_min_nb=32,
        superpixels_filter_black=True
    ):
        super().__init__()
        self.img_size = img_size
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.train = self.val = self.test = None
        if num_workers == "auto":
            self.num_workers = os.cpu_count() // torch.cuda.device_count()
        else:
            self.num_workers = num_workers
        self.use_cache = use_cache
        self.persistent_workers = True

        self.use_superpixels = use_superpixels
        self.superpixels_scales = superpixels_scales
        self.superpixels_max_nb = superpixels_max_nb
        self.superpixels_min_nb = superpixels_min_nb
        self.superpixels_filter_black = superpixels_filter_black

        segments_nb = np.logspace(
            start=np.log(self.superpixels_min_nb) / np.log(2),
            stop=np.log(self.superpixels_max_nb) / np.log(2),
            num=self.superpixels_scales,
            base=2,
        ).astype(int)
        self.slics = [
            Slic(
                num_components=seg_nb,
                convert_to_lab=True,
                min_size_factor=0,
                manhattan_spatial_dist=False,
                compactness=10,
            )
            for seg_nb in segments_nb
        ]

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
            )
        ]

    def normalize_and_cast_op(self) -> List[Union[A.Compose, Callable]]:
        ops = []
        additional_targets = None
        if self.use_superpixels:
            ops.append(self.get_superpixels_decomposition())
            additional_targets = {"segments": "mask"}

        ops.append(
            A.Compose(
                [
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, always_apply=True),
                    ToTensorV2(always_apply=True),
                ],
                is_check_shapes=False,
                additional_targets=additional_targets,
            )
        )
        if self.use_superpixels:

            @nntools_wrapper
            def permute_segments(segments):
                return {"segments": segments.permute(2, 0, 1)}

            ops.append(permute_segments)
        return ops

    def get_superpixels_decomposition(self):
        @nntools_wrapper
        def get_superpixels(image, mask=None):
            output = {"image": image}
            image = cv2.medianBlur(image, 9)
            list_segments = [slic.iterate(image) for slic in self.slics]
            
            segments = np.stack(list_segments, axis=-1)
            if mask is not None:
                if self.superpixels_filter_black:
                    segments = np.expand_dims(mask, 2) * segments
                output["mask"] = mask
            output["segments"] = segments.astype(int)
            return output
        
        @nntools_wrapper
        def get_debug_superpixels(image, mask=None):
            output = {"image": image}
            h, w = image.shape[:2]
            segments = np.arange(0, 4096).reshape((64, 64))
            segments = cv2.resize(segments, (h, w), interpolation=cv2.INTER_NEAREST)
            output["segments"] = np.expand_dims(segments.astype(int), 2)
            output['mask'] = mask
            return output

        return get_superpixels
