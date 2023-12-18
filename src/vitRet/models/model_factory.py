import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from vitRet.models.stochastic_attention.stochastic_vit import StochasticVisionTransformer


def load_weights_from_timm(timm_model: nn.Module, model: nn.Module):
    for i, block in enumerate(timm_model.blocks):
        keys = model.blocks.blocks[i].load_state_dict(block.state_dict(), strict=False)

    msg = model.projector.load_state_dict(timm_model.patch_embed.state_dict(), strict=False)
    print(f"For projector, missing keys: {msg.missing_keys}, unexpected keys: {msg.unexpected_keys}")
    model.fc_norm.load_state_dict(timm_model.fc_norm.state_dict(), strict=False)
    try:
        model.head.load_state_dict(timm_model.head.state_dict(), strict=False)
    except RuntimeError:
        print("Not loading head, incompatible shapes")

    current_pos_embed = model.projector.pos_embed
    _, _, N1, _ = current_pos_embed.shape
    pos_embed = timm_model.pos_embed
    cls_token = timm_model.cls_token

    _, N2, C = pos_embed.shape

    cls_pos_embed = pos_embed[:, 0]
    pos_embed = pos_embed[:, 1:]
    pos_embed = pos_embed.permute(0, 2, 1).view(1, C, int(N2**0.5), int(N2**0.5))

    model.projector.pos_embed = torch.nn.Parameter(
        F.interpolate(pos_embed, size=(N1, N1), mode="bilinear", align_corners=False)
    )
    model.tokenizer.load_cls_pos_embed(cls_token)
    model.tokenizer.load_cls_token(cls_pos_embed)

    return model


def svt_32_tiny(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=192, depth=12, num_heads=3, kernel_size=32, *args, **kwargs
    )
    if pretrained:
        timm_model = timm.create_model("vit_tiny_patch32_384", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_16_tiny(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=192, depth=12, num_heads=3, kernel_size=16, *args, **kwargs
    )
    if pretrained:
        timm_model = timm.create_model("vit_tiny_patch16_384", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_32_small(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, kernel_size=32, *args, **kwargs
    )
    if pretrained:
        timm_model = timm.create_model("vit_small_patch32_384", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_16_small(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, kernel_size=16, *args, **kwargs
    )
    if pretrained:
        timm_model = timm.create_model("vit_small_patch16_384", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_16_base(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=768, depth=12, num_heads=12, kernel_size=16, *args, **kwargs
    )
    if pretrained:
        timm_model = timm.create_model("vit_base_patch16_384", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_32_base(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=768, depth=12, num_heads=12, kernel_size=32, *args, **kwargs
    )
    if pretrained:
        timm_model = timm.create_model("vit_base_patch32_384", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_16_large(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        kernel_size=16,
        *args,
        **kwargs,
    )
    if pretrained:
        timm_model = timm.create_model("vit_large_patch16_224", pretrained=True)
        model = load_weights_from_timm(timm_model, model)
    return model


def svt_retfound(num_classes: int, *args, pretrained=True, **kwargs):
    from timm.models.layers import trunc_normal_
    model = StochasticVisionTransformer(
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        kernel_size=16,
        *args,
        **kwargs,
    )
    timm_model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=1)
    if pretrained:
        path = "pretrained_weights/RETFound_cfp_weights.pth"
        checkpoint = torch.load(path, map_location="cpu")
        msg = timm_model.load_state_dict(checkpoint["model"], strict=False)
        print(msg.missing_keys)
    model = load_weights_from_timm(timm_model, model)
    trunc_normal_(model.head.weight, std=2e-5)

    return model


def svt_custom(num_classes, *args, **kwargs):
    return StochasticVisionTransformer(num_classes=num_classes, *args, **kwargs)


models = {
    "svt_32_tiny": svt_32_tiny,
    "svt_16_tiny": svt_16_tiny,
    "svt_32_small": svt_32_small,
    "svt_16_small": svt_16_small,
    "svt_32_base": svt_32_base,
    "svt_16_base": svt_16_base,
    "svt_16_large": svt_16_large,
    "svt_custom": svt_custom,
    "svt_retfound": svt_retfound,
}


def create_model(arch: str, num_classes=1000, *args, **kwargs):
    return models[arch](num_classes, *args, **kwargs)


if __name__ == "__main__":
    import torch

    model = create_model(
        "svt_retfound",
        num_classes=1000,
        img_size=512,
        max_tokens=128,
        projection_stride=4,
        pretrained=True,
    )
    print("Model created")
    foo = torch.randn(2, 3, 512, 512)

    print(model)
