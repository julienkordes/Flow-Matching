from .UNet import UNet
from .DiT import DiT

model_dict = {
    "unet": UNet,
    "dit": DiT,
}

def get_model(name: str, *args, **kwargs):
    name = name.lower()
    try:
        return model_dict[name](*args, **kwargs)
    except KeyError:
        raise ValueError(f"Model '{name}' not found. Available: {list(model_dict.keys())}")