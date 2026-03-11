import math
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader




def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif args.dataset == "celeba":
        dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {args.dataset} non supporté.")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

def get_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps          # warmup linéaire
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def flow_matching_loss(model, x1, class_label=None):
    B, _, _, _ = x1.shape
    x0 = torch.randn_like(x1)
    t  = torch.rand(B)
    xt = (1 - t) * x0 + t * x1
    ut = x1 - x0
    loss = F.mse_loss(model(xt, t, class_label=class_label), ut)
    return loss

def update_ema(ema_model, model, decay, step):
    # Les premiers steps, on copie directement sans lissage
    actual_decay = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(actual_decay).add_(param.data, alpha=1 - actual_decay)