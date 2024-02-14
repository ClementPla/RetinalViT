import logging

import timm
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from vitRet.models.prototypes_vit import DynViT
from vitRet.models.stochastic_attention.stochastic_vit import StochasticVisionTransformer
from vitRet.utils.ckpts import ModelCkpt, ProjectorCkpt


def load_weights_from_timm(timm_model: VisionTransformer, model: StochasticVisionTransformer):
    
    model.blocks.load_state_dict(timm_model.blocks.state_dict(), strict=True)
    try:
        msg = model.projector.load_state_dict(timm_model.patch_embed.state_dict(), strict=False)
        logging.debug(f"For projector, missing keys: {msg.missing_keys}, unexpected keys: {msg.unexpected_keys}")
    except RuntimeError as e:
        pass

    model.norm.load_state_dict(timm_model.norm.state_dict(), strict=True)
    model.fc_norm.load_state_dict(timm_model.fc_norm.state_dict(), strict=True)
    try:
        model.head.load_state_dict(timm_model.head.state_dict(), strict=False)
    except RuntimeError:
        logging.warn("Not loading head, incompatible shapes")

    if model.is_compressed:
        ckpt = ProjectorCkpt.DEPTH_32
        try:
            projector_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
        except FileNotFoundError:
            return model
            
        only_projector = {}
        for k, v in projector_dict.items():
            if k.startswith('trained_projector.'):
                only_projector[k.replace('trained_projector.', '')] = v
        
        pos_embed = only_projector["projector.pos_embed"]
        if not pos_embed.shape[1] == model.projector.pos_embed.shape[2]:
            return model
        only_projector["projector.pos_embed"] = F.interpolate(pos_embed, 
                                                        size=model.projector.pos_embed.shape[-2:], 
                                                        mode="bilinear")
        inc_keys =  model.load_state_dict(only_projector, strict=False)
        print(inc_keys.unexpected_keys)
        return model

    if model.projector.pos_embed.shape[1] == timm_model.pos_embed.shape[2]:
        current_pos_embed = model.projector.pos_embed
        _, _, N1, _ = current_pos_embed.shape
        pos_embed = timm_model.pos_embed
        if not model.global_pool:
            cls_token = timm_model.cls_token
            cls_pos_embed = pos_embed[:, :1]
            model.tokenizer.load_cls_pos_embed(cls_pos_embed)
            model.tokenizer.load_cls_token(cls_token)

        pos_embed = pos_embed[:, 1:]
        _, N2, C = pos_embed.shape
        pos_embed = pos_embed.permute(0, 2, 1).view(1, C, int(N2**0.5), int(N2**0.5))

        if N1 != int(N2**0.5):
            pos_embed = F.interpolate(pos_embed, size=(N1, N1), mode="bilinear", align_corners=False)
        model.projector.pos_embed = torch.nn.Parameter(pos_embed)
    return model

def dyn_vit(num_classes: int, *args, **kwargs):    
    return DynViT(num_classes=num_classes, *args, **kwargs)

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

def svt_16_base_fundus(num_classes: int, *args, pretrained=True, **kwargs):
    model = StochasticVisionTransformer(
        num_classes=num_classes, embed_dim=768, depth=12, num_heads=12, kernel_size=16, *args, **kwargs
    )
    ckpt_path = ModelCkpt.STANDARD
    projector_ckpt_path = ProjectorCkpt.DEPTH_32
    state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    for k in list(state_dict.keys()):
        state_dict[k.replace('model.', '')] = state_dict.pop(k)
    state_dict['tokenizer.cls_token'].squeeze_(1)

    projector_dict = torch.load(projector_ckpt_path, map_location='cpu')['state_dict']
    for k, v in projector_dict.items():
        if k.startswith('trained_projector.'):
            state_dict[k.replace('trained_projector.', '')] = v

    state_dict['projector.pos_embed'] = F.interpolate(state_dict['projector.pos_embed'], 
                                                    size=model.projector.pos_embed.shape[-2:], 
                                                    mode='bilinear')
    model.load_state_dict(state_dict=state_dict, strict=True)
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
    global_pool = kwargs.get("global_pool", False)
    if global_pool:
        timm_model = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=num_classes, class_token=False, global_pool="avg"
        )
    else:
        timm_model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=num_classes)

    if pretrained:
        path = "pretrained_weights/RETFound_cfp_weights.pth"
        checkpoint = torch.load(path, map_location="cpu")
        for key in list(checkpoint["model"].keys()):
            if "decoder" in key:
                del checkpoint["model"][key]

        if global_pool:
            """
            RetFOUND builds a cls token but discards it at inference time. 
            We just remove it from the checkpoint 
            (including associated positional embedding)
            """
            checkpoint["model"]["pos_embed"] = checkpoint["model"]["pos_embed"][:, 1:]
        msg = timm_model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Loading RetFound in timm model: missing keys {msg.missing_keys}, unexpected_keys {msg.unexpected_keys}")
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
    "svt_16_base_fundus": svt_16_base_fundus,
    "dyn_vit": dyn_vit
}


def create_model(arch: str, num_classes=1000, *args, **kwargs):
    return models[arch](num_classes, *args, **kwargs)


if __name__ == "__main__":
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
