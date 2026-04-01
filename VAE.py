"""
Latent KNN Coupling for CIFAR-10 — Class-Conditional
======================================================
1. Encode tout CIFAR-10 avec le VAE de Stable Diffusion
2. Stocke les labels associés à chaque latent
3. Construit un index FAISS PAR CLASSE (10 index)
4. Fournit un DataLoader qui retourne (z0, z1, class_label)
   où z1 est le k-NN de z0 DANS LA MÊME CLASSE

Usage:
    from latent_knn_coupling import build_latent_index
    vae, latents, labels, class_indices, class_masks, loader = build_latent_index()
    for z0, z1, y in loader:
        # z0: bruit, z1: k-NN intra-classe, y: label (0-9)
        loss = flow_matching_loss(model, z0, z1, y)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
import faiss
import os

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

VAE_MODEL_ID   = "stabilityai/sd-vae-ft-mse"
LATENT_SCALE   = 0.18215
VAE_INPUT_SIZE = 256
NUM_CLASSES    = 10


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement du VAE
# ─────────────────────────────────────────────────────────────────────────────

def load_vae(device="cuda"):
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ─────────────────────────────────────────────────────────────────────────────
# 2. Encoding de CIFAR-10 avec labels
# ─────────────────────────────────────────────────────────────────────────────

def encode_cifar(vae, cifar_root="./data", batch_size=64, device="cuda",
                 cache_path="./cifar10_latents.pt"):
    """
    Encode tout CIFAR-10 train.
    Retourne:
        latents : (N, 4, 32, 32)
        labels  : (N,) int64
    """
    if os.path.exists(cache_path):
        print(f"[VAE] Chargement depuis {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["latents"], data["labels"]

    print("[VAE] Encodage de CIFAR-10...")

    transform = transforms.Compose([
        transforms.Resize(VAE_INPUT_SIZE,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.CIFAR10(root=cifar_root, train=True,
                                download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    all_latents, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Encoding CIFAR-10"):
            imgs = imgs.to(device)
            z    = vae.encode(imgs).latent_dist.mode() * LATENT_SCALE
            all_latents.append(z.cpu())
            all_labels.append(lbls)

    latents = torch.cat(all_latents, dim=0)   # (50000, 4, 32, 32)
    labels  = torch.cat(all_labels,  dim=0)   # (50000,)

    torch.save({"latents": latents, "labels": labels}, cache_path)
    print(f"[VAE] Sauvegardé — latents: {latents.shape}")
    return latents, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. Index FAISS par classe
# ─────────────────────────────────────────────────────────────────────────────

def build_class_indices(latents, labels, index_dir="./faiss_indices",
                        use_gpu=True, nlist=64):
    """
    Construit un index FAISS par classe (10 index pour CIFAR-10).

    Retourne:
        class_indices : dict {class_id -> faiss.Index}
        class_masks   : dict {class_id -> np.array of global indices}
    """
    os.makedirs(index_dir, exist_ok=True)

    N        = latents.shape[0]
    D        = latents.shape[1] * latents.shape[2] * latents.shape[3]
    all_flat = latents.reshape(N, D).numpy().astype(np.float32)

    class_indices = {}
    class_masks   = {}

    for c in range(NUM_CLASSES):
        index_path = os.path.join(index_dir, f"class_{c}.index")
        mask_path  = os.path.join(index_dir, f"class_{c}_mask.npy")

        mask = (labels == c).numpy()
        idxs = np.where(mask)[0]
        vecs = all_flat[idxs]
        N_c  = len(idxs)

        class_masks[c] = idxs

        if os.path.exists(index_path):
            print(f"[FAISS] Chargement index classe {c} ({N_c} samples)")
            index = faiss.read_index(index_path)
        else:
            print(f"[FAISS] Construction index classe {c} ({N_c} samples)...")
            _nlist    = min(nlist, N_c // 10)
            quantizer = faiss.IndexFlatL2(D)
            index     = faiss.IndexIVFFlat(quantizer, D, _nlist, faiss.METRIC_L2)
            index.train(vecs)
            index.add(vecs)
            index.nprobe = min(32, _nlist)
            faiss.write_index(index, index_path)
            np.save(mask_path, idxs)

        if use_gpu and faiss.get_num_gpus() > 0:
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        class_indices[c] = index

    return class_indices, class_masks


# ─────────────────────────────────────────────────────────────────────────────
# 4. KNN Coupling intra-classe
# ─────────────────────────────────────────────────────────────────────────────

def knn_couple_batch(z0_batch, class_labels, latents_flat,
                     class_indices, class_masks, k=10):
    """
    Pour chaque (z0_i, class_i), trouve les k plus proches voisins de z0_i
    parmi les latents de la classe class_i, et en tire un uniformément.

    Args:
        z0_batch     : (B, D) numpy float32
        class_labels : (B,)   numpy int64
        latents_flat : (N, D) numpy float32
        class_indices: dict {c -> faiss.Index}
        class_masks  : dict {c -> array of global indices}
        k            : nombre de voisins

    Returns:
        z1_batch      : (B, D) numpy float32
        z1_global_idx : (B,)   indices globaux dans latents_flat
    """
    B        = z0_batch.shape[0]
    D        = z0_batch.shape[1]
    z1_batch = np.zeros((B, D), dtype=np.float32)
    z1_idx   = np.zeros(B, dtype=np.int64)

    for c in range(NUM_CLASSES):
        mask_b = (class_labels == c)
        if not mask_b.any():
            continue

        batch_idx_c = np.where(mask_b)[0]
        queries     = z0_batch[batch_idx_c]

        k_eff           = min(k, len(class_masks[c]))
        _, local_nn_idx = class_indices[c].search(queries, k_eff)

        n_c     = len(batch_idx_c)
        chosen  = np.random.randint(0, k_eff, size=n_c)
        local_i = local_nn_idx[np.arange(n_c), chosen]
        global_i = class_masks[c][local_i]

        z1_batch[batch_idx_c] = latents_flat[global_i]
        z1_idx[batch_idx_c]   = global_i

    return z1_batch, z1_idx


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KNNLatentDataset(Dataset):
    """
    Retourne (z0, placeholder, label).
    Le z1 réel est calculé dans le collate_fn pour regrouper les appels FAISS.
    """

    def __init__(self, labels, latent_shape=(4, 32, 32)):
        self.labels      = labels
        self.latent_shape = latent_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int(self.labels[idx].item())
        z0    = torch.randn(*self.latent_shape)
        # z1 est un placeholder — remplacé dans collate_fn
        return z0, torch.zeros(*self.latent_shape), torch.tensor(label, dtype=torch.long)


def make_collate_fn(latents_flat, class_indices, class_masks, k, latent_shape):
    """
    collate_fn efficace : un seul appel FAISS par classe par batch.
    """
    D = int(np.prod(latent_shape))

    def collate(batch):
        z0_batch = torch.stack([b[0] for b in batch])   # (B, *latent_shape)
        y_batch  = torch.stack([b[2] for b in batch])   # (B,)
        B        = z0_batch.shape[0]

        z0_flat = z0_batch.reshape(B, D).numpy().astype(np.float32)
        y_np    = y_batch.numpy().astype(np.int64)

        z1_flat, _ = knn_couple_batch(
            z0_flat, y_np, latents_flat,
            class_indices, class_masks, k
        )

        z1_batch = torch.from_numpy(z1_flat).reshape(B, *latent_shape)
        return z0_batch, z1_batch, y_batch

    return collate


# ─────────────────────────────────────────────────────────────────────────────
# 6. Décodeur
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_latents(vae, latents, device="cuda"):
    latents = latents.to(device) / LATENT_SCALE
    images  = vae.decode(latents).sample
    return images.clamp(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def build_latent_index(cifar_root="./data", device="cuda",
                       cache_latents="./cifar10_latents.pt",
                       index_dir="./faiss_indices",
                       k=10, batch_size=128):
    """
    Pipeline complet.

    Exemple dans ta boucle d'entraînement :

        vae, latents, labels, class_indices, class_masks, loader = build_latent_index(k=10)

        for z0, z1, y in loader:
            z0, z1, y = z0.to(device), z1.to(device), y.to(device)
            t   = torch.rand(z0.shape[0], device=device)
            x_t = (1 - t[:,None,None,None]) * z0 + t[:,None,None,None] * z1
            pred = your_model(x_t, t, class_label=y)
            loss = F.mse_loss(pred, z1 - z0)
    """
    vae                        = load_vae(device=device)
    latents, labels            = encode_cifar(vae, cifar_root=cifar_root,
                                              batch_size=64, device=device,
                                              cache_path=cache_latents)
    class_indices, class_masks = build_class_indices(latents, labels,
                                                      index_dir=index_dir,
                                                      use_gpu=(device == "cuda"))

    latent_shape = tuple(latents.shape[1:])
    D            = int(np.prod(latent_shape))
    latents_flat = latents.reshape(len(latents), D).numpy().astype(np.float32)

    dataset = KNNLatentDataset(labels, latent_shape=latent_shape)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # FAISS non fork-safe
        pin_memory=True,
        collate_fn=make_collate_fn(latents_flat, class_indices,
                                   class_masks, k, latent_shape),
    )

    return vae, latents, labels, class_indices, class_masks, loader


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sanity check
# ─────────────────────────────────────────────────────────────────────────────

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    vae, latents, labels, class_indices, class_masks, loader = build_latent_index(
        cifar_root="./data", device=DEVICE, index_dir="/Data/fm_data", k=10, batch_size=32
    )

    z0, z1, y = next(iter(loader))
    print(f"z0 shape : {z0.shape}")
    print(f"z1 shape : {z1.shape}")
    print(f"labels   : {[CIFAR_CLASSES[i] for i in y[:8].tolist()]}")

    # Vérifie distance KNN < distance random intra-classe
    dist_knn = (z0 - z1).reshape(z0.shape[0], -1).norm(dim=1).mean().item()

    D = z0.shape[1] * z0.shape[2] * z0.shape[3]
    lf = latents.reshape(len(latents), D)
    z1_rand_list = [
        lf[class_masks[c.item()][np.random.randint(len(class_masks[c.item()]))]]
        for c in y
    ]
    z1_rand     = torch.stack(z1_rand_list).reshape(z0.shape)
    dist_random = (z0 - z1_rand).reshape(z0.shape[0], -1).norm(dim=1).mean().item()

    print(f"\nDistance z0→z1 KNN intra-classe    : {dist_knn:.4f}")
    print(f"Distance z0→z1 random intra-classe : {dist_random:.4f}")
    print(f"Ratio KNN/random                   : {dist_knn/dist_random:.3f}  (< 1 attendu)")

    # Visualisation
    imgs = decode_latents(vae, z1[:8].float(), device=DEVICE)
    imgs = (imgs.cpu() + 1) / 2

    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(CIFAR_CLASSES[y[i].item()], fontsize=7)
        ax.axis("off")
    plt.suptitle("Decoded z1 — KNN intra-classe de z0", fontsize=10)
    plt.tight_layout()
    plt.savefig("knn_class_sanity.png", dpi=100, bbox_inches="tight")
    print("Saved knn_class_sanity.png")