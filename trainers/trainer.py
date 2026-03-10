import os
import torch
from tqdm import tqdm
from models import get_model
from utils import get_dataloader

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Données
    dataloader = get_dataloader(args)

    model = get_model(args.model, 
            args.in_channels, 
            args.channels, 
            args.network_depth,
            args.num_res_block, 
            args.attention_resolution, 
            args.image_size, 
            args.time_emb_dim
            ).to(device)
    
    start_epoch = 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(args.checkpoint_path.split("_")[-1].replace(".pt", "")) 
    
    
    return