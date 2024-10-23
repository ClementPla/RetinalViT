import os
from typing import Tuple, Union

import timm
import torch


def get_feature_extractor(hub_path, out_indices: Union[int, Tuple[int]] = (4,)):
    if not isinstance(out_indices, tuple):
        out_indices = (out_indices,)

    try:
        model = timm.create_model(
            f"{hub_path}",
            pretrained=True,
            img_size=(1024, 1024),
            num_classes=1,
            features_only=True,
        )
    except TypeError:
        try:
            model = timm.create_model(
                f"{hub_path}",
                pretrained=True,
                num_classes=1,
                output_stride=8,
                features_only=True,
            )
        except TypeError:
            model = timm.create_model(
                f"{hub_path}",
                pretrained=True,
                num_classes=1,
                features_only=True,
            )

    return model
