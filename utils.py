import math
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw



def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
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
    t  = torch.rand(B, device=x1.device)
    t_ = t.view(B, 1, 1, 1)
    xt = (1 - t_) * x0 + t_ * x1
    ut = x1 - x0
    loss = F.mse_loss(model(xt, t, class_label=class_label), ut)
    return loss

def update_ema(ema_model, model, decay, step):
    # Les premiers steps, on copie directement sans lissage
    actual_decay = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(actual_decay).add_(param.data, alpha=1 - actual_decay)

@torch.no_grad()
def sample(model, args, shape, class_label=None):
    B, _, _, _ = shape
    def rk4_step(x, t, dt):
        t_ = t * torch.ones(B, device=x.device)
        k1 = model(x,              t_, class_label)
        k2 = model(x + dt/2 * k1, t_ + dt/2, class_label)
        k3 = model(x + dt/2 * k2, t_ + dt/2, class_label)
        k4 = model(x + dt   * k3, t_ + dt, class_label)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    x = torch.randn(shape).to(args.device)
    dt = 1.0 / args.sample_steps
    for _, t in enumerate(torch.linspace(0, 1 - dt, args.sample_steps)):
        x = rk4_step(x, t.item(), dt)
    return x


CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

@torch.no_grad()
def sample_and_save(model, args, epoch, class_label=None):
    model.eval()
    if class_label is None:
        class_labels = torch.randint(0, args.num_classes, (args.num_samples,), device=args.device)
    else:
        class_labels = torch.tensor([class_label] * args.num_samples, device=args.device)

    shape = (args.num_samples, args.in_channels, args.img_size, args.img_size)
    x = sample(model, args, shape, class_label=class_label)
    samples = (x.clamp(-1, 1) + 1) / 2

    LABEL_HEIGHT = 12  # pixels de hauteur pour le texte
    annotated = []
    for img, label_idx in zip(samples, class_labels):
        pil_img = TF.to_pil_image(img)
        annotated_img = Image.new("RGB", (pil_img.width, pil_img.height + LABEL_HEIGHT), (255, 255, 255))
        annotated_img.paste(pil_img, (0, 0))
        draw = ImageDraw.Draw(annotated_img)
        label_name = CIFAR10_CLASSES[label_idx.item()]
        text_x = pil_img.width // 2 - len(label_name) * 3
        draw.text((text_x, pil_img.height + 1), label_name, fill=(0, 0, 0))
        annotated.append(TF.to_tensor(annotated_img))

    annotated = torch.stack(annotated)
    grid = make_grid(annotated, nrow=4)
    path = os.path.join(args.output_dir, f"samples_epoch_{epoch:04d}.png")
    save_image(grid, path)
    model.train()
    print(f"  Samples sauvegardés : {path}")