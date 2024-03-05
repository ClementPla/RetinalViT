from vitRet.models.prototypes_vit import DynViT


def dyn_vit(num_classes: int, *args, **kwargs):    
    return DynViT(num_classes=num_classes, *args, **kwargs)


models = {
    "dyn_vit": dyn_vit
}


def create_model(arch: str, num_classes=1000, *args, **kwargs):
    return models[arch](num_classes, *args, **kwargs)

