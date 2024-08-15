import os
from typing import Tuple, Union

import timm
import torch


def get_feature_extractor(hub_path, out_indices: Union[int, Tuple[int]] = (4,)):
    if not isinstance(out_indices, tuple):
        out_indices = (out_indices,)

    try:
        model = timm.create_model(
            f"hf-hub:{hub_path}",
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
            img_size=(1024, 1024),
            num_classes=1,
        )
    except Exception as e:
        model = timm.create_model(
            f"hf-hub:{hub_path}",
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
            num_classes=1,
            output_stride=4,
        )

    return model


if __name__ == "__main__":
    import torch

    model = get_feature_extractor("ClementP/FundusDRGrading-mobilenetv3_small_100", out_indices=-2)
    x = torch.randn(1, 3, 1024, 1024)
    out = model(x)
    print(out[-1].shape)
