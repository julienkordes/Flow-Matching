import os
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import get_model
from utils import get_dataloader, get_scheduler, flow_matching_loss, update_ema, sample_and_save
from torch.optim import AdamW

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Données
    dataloader = get_dataloader(args)

    model = get_model(args.model, embed_dim=args.embed_dim, num_heads=args.num_heads, depth=args.depth, 
            in_channels=args.in_channels, img_size=args.img_size, patch_size=args.patch_size, num_classes=args.num_classes
            ).to(device)
    
    start_epoch = 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(args.checkpoint_path.split("_")[-1].replace(".pt", "")) 

    ema_model = copy.deepcopy(model)      
    ema_model.requires_grad_(False)      
    ema_decay = 0.9999
    
    num_epochs   = args.num_epochs 
    total_steps  = num_epochs * len(dataloader)
    warmup_steps = int(0.01 * total_steps)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    # Si on a loadé un checkpoint on fait avancer le scheduler manuellement jusqu'à la bonne valeur
    if start_epoch > 0:
        for _ in range(start_epoch):
            lr_scheduler.step()
        
    losses = []
    global_step = 0
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")

        for batch_idx, (x, class_label) in enumerate(pbar):
            x, class_label = x.to(device), class_label.to(device)
            optimizer.zero_grad()
            drop_mask = torch.rand(x.shape[0], device=device) < 0.1  
            class_label[drop_mask] = args.num_classes  # null token = index 10
            loss = flow_matching_loss(model, x, class_label=class_label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            update_ema(ema_model, model, ema_decay, global_step) 
            epoch_loss += loss.item() 
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch} — Loss moyenne : {avg_loss:.4f} — LR : {lr_scheduler.get_last_lr()[0]:.6f}")

        if epoch % args.sample_every == 0:
            sample_and_save(ema_model, args, epoch)

        # Sauvegarde checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"FM_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt_path)
            print(f"  Checkpoint sauvegardé : {ckpt_path}")

    # Courbe de loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Courbe d'entraînement DDPM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"))
    plt.close()
    print("Entraînement terminé.")
    return