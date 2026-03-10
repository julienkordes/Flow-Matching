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
