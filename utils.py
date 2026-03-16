import math
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch_fidelity import calculate_metrics
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

    def cfg_vector_field(model, x, t, class_label, guidance_scale):
        t = min(t, 1.0 - 1e-5) 
        t_ = t * torch.ones(B, device=x.device)

        if class_label is None or guidance_scale == 1.0:
            return model(x, t_, class_label)  # pas de CFG

        v_cond = model(x, t_, class_label)
        null_label = torch.full_like(class_label, model.num_classes)
        v_uncond = model(x, t_, null_label)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def rk4_step_cfg(x, t, guidance_scale):
        k1 = cfg_vector_field(model, x, t, class_label, guidance_scale)
        k2 = cfg_vector_field(model, x + dt/2 * k1, t + dt/2, class_label, guidance_scale)
        k3 = cfg_vector_field(model, x + dt/2 * k2, t + dt/2, class_label, guidance_scale)
        k4 = cfg_vector_field(model, x + dt * k3, t + dt, class_label, guidance_scale)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    x = torch.randn(shape).to(args.device)
    dt = 1.0 / args.sample_steps
    for t in torch.linspace(0, 1 - dt, args.sample_steps):
        x = rk4_step_cfg(x, t.item(), args.guidance_scale)  
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
    x = sample(model, args, shape, class_label=class_labels)
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


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        # tensors : (N, 3, 32, 32) dans [0, 255] uint8
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, i):
        return self.tensors[i]


def show_metrics(model, args):
    """ Calcule la FID et l'IS """
    model.eval()
    all_samples = []

    for _ in tqdm(range(0, args.num_samples_FID, args.batch_size)):
        class_labels = torch.randint(0, args.num_classes, (args.batch_size,), device=args.device)
        shape = (args.batch_size, args.in_channels, args.img_size, args.img_size)
        x = sample(model, args, shape, class_label=class_labels)
        samples = (x.clamp(-1, 1) + 1) / 2

        samples = (samples * 255).byte().cpu()
        all_samples.append(samples)

    generated = torch.cat(all_samples, dim=0)
    dataset = GeneratedDataset(generated)
    metrics = calculate_metrics(input1=dataset, input2="cifar10-train", fid=True, isc=True)

    fid = metrics['frechet_inception_distance']
    isc = metrics['inception_score_mean']
    isc_std = metrics['inception_score_std']

    steps_info = f"{args.sample_steps} steps"

    col_width = 30
    sep = "+" + "-" * col_width + "+" + "-" * 20 + "+\n"

    def row(label, value):
        return f"| {label:<{col_width-2}} | {value:<18} |\n"

    table = "\n"
    table += sep
    table += row("Metric", "Value")
    table += sep
    table += row("Steps", steps_info)
    table += row("Guidance scale", f"{args.guidance_scale:.1f}")
    table += row("Num samples", str(args.num_samples_FID))
    table += sep
    table += row("FID ↓ (lower is better)", f"{fid:.2f}")
    table += row("IS  ↑ (higher is better)", f"{isc:.2f} ± {isc_std:.2f}")
    table += sep

    # Sauvegarde
    filename = f"metrics_flow_matching"
    filename += ".txt"

    output_path = os.path.join(args.output_dir, filename)
    with open(output_path, "w") as f:
        f.write(table)
    print(f"Métriques sauvegardées : {output_path}")