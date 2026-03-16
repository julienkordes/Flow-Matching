import torch
from config.argparser import load_opts
from utils import show_metrics
from models import get_model


@torch.no_grad()
def eval(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    model_configs = {
        "dit": {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "depth": args.depth,
            "patch_size": args.patch_size,
            "img_size": args.img_size,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
        },
        "unet": {
            "in_channels": args.in_channels,
            "channels": args.channels,
            "network_depth": args.network_depth,
            "num_res_block": args.num_res_block,
            "attention_resolution": args.attention_resolution,
            "image_size": args.img_size,
            "time_emb_dim": args.time_emb_dim,
        }
    }

    ema_model = get_model(args.model, **model_configs[args.model]).to(device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    ema_model.load_state_dict(ckpt["ema_model_state_dict"])
    print(f"Modèle chargé - Epoch : {ckpt['epoch']} - Loss : {ckpt['loss']}")

    show_metrics(ema_model, args)

if __name__ == "__main__":
    args = load_opts()
    eval(args)
