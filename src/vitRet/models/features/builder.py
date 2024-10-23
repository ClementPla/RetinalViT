from enum import Enum
from typing import Tuple, Union

import torch.nn as nn

from vitRet.models.superpixels.topo import filter_small_segments


class AvailableExtractor(str, Enum):
    SEGMENT_EMBEDDING = "SEGMENT_EMBEDDING"
    DINO_SEGMENT_EMBEDDING = "DINO_SEGMENT_EMBEDDING"
    CNN_SCATTERING = "CNN_SCATTERING"
    HANDCRAFTED_FEATURES = "HANDCRAFTED_FEATURES"
    SPITVIT = "SPITVIT"
    BBOX_CNN = "BBOX_CNN"
    SUPERPATCH = "SUPERPATCH"
    SEGMENT_SCATTER = "SEGMENT_SCATTER"


class FeaturesExtractor(nn.Module):
    def __init__(
        self,
        model_type: AvailableExtractor,
        in_chans=3,
        n_conv_stem=4,
        inter_dim: int = 128,
        embed_dim: int = 768,
        img_size: Union[int, Tuple[int, int]] = 1024,
        hub_model="",
        f_index=4,
        output_stride=8,
        block_size=(512, 512),
        block_stride=(512, 512),
        downsample_segments=False,
        cat_input=False,
        radius: int = 0.1,
        out_dim=256,
        eps=1e-6,
        beta=8,
        include_texture_descriptor=False,
        concat_positional_embedding=False,
        isolate_superpixels=False,
        minimum_segment_size=0,
        optimized=True,
        kernel_size=16,
        **kwargs,
    ):
        super().__init__()
        self.minimum_segment_size = minimum_segment_size
        match model_type:
            case AvailableExtractor.SEGMENT_EMBEDDING:
                from vitRet.models.features.segment_embedding import SegmentEmbed

                self.extractor = SegmentEmbed(in_chans, n_conv_stem, inter_dim, embed_dim, kernel_size=kernel_size)
            case AvailableExtractor.DINO_SEGMENT_EMBEDDING:
                from vitRet.models.features.dino import DinoSegmentEmbedding

                self.extractor = DinoSegmentEmbedding()
            case AvailableExtractor.CNN_SCATTERING:
                raise NotImplementedError("CNN_SCATTERING is not implemented yet")
            case AvailableExtractor.HANDCRAFTED_FEATURES:
                from vitRet.models.features import HandcraftedFeatures

                self.extractor = HandcraftedFeatures(radius, out_dim, eps)
            case AvailableExtractor.SPITVIT:
                from vitRet.models.features.spitvit import SPITViTFeatures

                self.extractor = SPITViTFeatures(
                    img_size,
                    beta,
                    embed_size=embed_dim,
                    include_texture_descriptor=include_texture_descriptor,
                    concat_positional_embedding=concat_positional_embedding,
                )
            case AvailableExtractor.SUPERPATCH | AvailableExtractor.BBOX_CNN:
                from vitRet.models.features.cnn import SuperPatchCNN

                self.extractor = SuperPatchCNN(
                    hub_model, f_index, beta=beta, isolate_superpixels=isolate_superpixels, output_features=embed_dim
                )
            case AvailableExtractor.SEGMENT_SCATTER:
                from vitRet.models.features.cnn import SegmentScatteringCNN

                self.extractor = SegmentScatteringCNN(
                    hub_model, f_index, output_features=embed_dim, cat_input=cat_input, optimized=optimized
                )
            case _:
                raise ValueError(f"Unknown model type: {model_type}")

    def init_weights(self):
        pass

    def forward(self, x, segment, *args, **kwargs):
        if self.minimum_segment_size:
            segment = filter_small_segments(segment.long(), self.minimum_segment_size).long()

        return self.extractor(x, segment, *args, **kwargs)

    @property
    def embed_size(self):
        return self.extractor.embed_size


if __name__ == "__main__":
    from vitRet.data.data_factory import get_test_sample

    batch = get_test_sample()
    image = batch["image"].cuda()
    segments = batch["segments"].cuda()
    print(image.shape, segments.shape)

    fmodel = FeaturesExtractor(
        AvailableExtractor.SUPERPATCH,
        hub_model="ClementP/FundusDRGrading-mobilenetv3_small_100",
        beta=32,
        f_index=-1,
    ).cuda()

    output, segment = fmodel(image, segments)

    print(output.shape, segment.shape)
