"""
Toy 2D — Ablation sur top-k nearest neighbor coupling
======================================================
Étudie l'influence de k sur :
  - La qualité de génération (couverture de la distribution)
  - La rectitude des trajectoires
  - La training loss
 
k=1   : nearest neighbor strict  → trajectoires droites, biais de couverture
k=inf : couplage aléatoire       → bonne couverture, trajectoires courbées
k=?   : le sweet spot             → c'est ce qu'on cherche
 
Run : python toy2d_topk.py
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
 
torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────────────
 
def sample_crescent(n):
    theta = torch.rand(n) * torch.pi
    r     = 3.0 + 0.25 * torch.randn(n)
    return torch.stack([r * theta.cos(), r * theta.sin() - 1.5], dim=1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Couplages
# ─────────────────────────────────────────────────────────────────────────────
 
def topk_coupling(x0, x1, k, temperature=1.0):
    """
    Pour chaque x0_i, sample parmi les k plus proches x1 voisins.
 
    k=1         : nearest neighbor strict
    k=len(x1)   : couplage aléatoire uniforme
    temperature : contrôle le softness du sampling
                  → 0 : toujours le plus proche (déterministe)
                  → ∞ : uniforme sur les k voisins
 
    Returns: x0, x1_reordered (même interface que les autres couplages)
    """
    B     = x0.shape[0]
    dists = torch.cdist(x0, x1)    # (B, B)
 
    if k >= B:
        # Couplage aléatoire pur
        idx = torch.randint(0, B, (B,), device=x0.device)
        return x0, x1[idx]
 
    # Top-k plus proches voisins
    topk_dists, topk_idx = dists.topk(k, dim=1, largest=False)  # (B, k)
 
    if k == 1:
        # Nearest neighbor strict — pas besoin de sampling
        return x0, x1[topk_idx.squeeze(1)]
 
    # Sample selon softmax sur les distances négatives
    logits  = -topk_dists / (temperature + 1e-8)
    probs   = F.softmax(logits, dim=1)                           # (B, k)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(1) # (B,)
    indices = topk_idx[torch.arange(B, device=x0.device), sampled]
 
    return x0, x1[indices]
 
 
def make_coupling_fn(k, temperature=1.0):
    """Factory pour créer une fonction de couplage avec k fixé."""
    def fn(x0, x1):
        return topk_coupling(x0, x1, k=k, temperature=temperature)
    fn.__name__ = f"k={k}"
    return fn
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Velocity field
# ─────────────────────────────────────────────────────────────────────────────
 
class VelocityField(nn.Module):
    def __init__(self, dim=2, hidden=256, depth=4):
        super().__init__()
        self.edim = 32
        layers = [nn.Linear(dim + self.edim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)
 
    def time_emb(self, t):
        d     = self.edim // 2
        freqs = torch.exp(-torch.arange(d, device=t.device, dtype=t.dtype)
                          * (np.log(10000) / d))
        args  = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)
 
    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_emb(t)], dim=-1))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────
 
def train(coupling_fn, x1_data, n_steps=8000, batch_size=512,
          lr=3e-4, device="cpu"):
    x1_data = x1_data.to(device)
    model   = VelocityField(2, 256, 4).to(device)
    opt     = Adam(model.parameters(), lr=lr)
    losses  = []
 
    for step in range(n_steps):
        idx1 = torch.randint(0, len(x1_data), (batch_size,), device=device)
        x0   = torch.randn(batch_size, 2, device=device)
        x1   = x1_data[idx1]
 
        x0_c, x1_c = coupling_fn(x0, x1)
 
        t    = torch.rand(batch_size, device=device)
        x_t  = (1 - t).unsqueeze(-1) * x0_c + t.unsqueeze(-1) * x1_c
        tgt  = x1_c - x0_c
        pred = model(x_t, t)
 
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
 
    return model, losses
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Métriques
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
def sample_with_traj(model, n_gen=2000, n_traj=30, n_steps=100, device="cpu",
                     seed=7):
    dt = 1.0 / n_steps
 
    # Samples généraux (RK4)
    x = torch.randn(n_gen, 2, device=device)
    for i in range(n_steps):
        t  = torch.full((n_gen,), i * dt, device=device)
        k1 = model(x,            t)
        k2 = model(x + .5*dt*k1, t + .5*dt)
        k3 = model(x + .5*dt*k2, t + .5*dt)
        k4 = model(x +    dt*k3, t +    dt)
        x  = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    gen = x.cpu().numpy()
 
    # Trajectoires (Euler simple — plus lisible visuellement)
    torch.manual_seed(seed)
    x_traj  = torch.randn(n_traj, 2, device=device)
    x0_init = x_traj.cpu().numpy().copy()
    traj    = [x0_init.copy()]
 
    x = x_traj.clone()
    for i in range(n_steps):
        t = torch.full((n_traj,), i * dt, device=device)
        x = x + dt * model(x, t)
        traj.append(x.cpu().numpy().copy())
 
    return gen, np.array(traj), x0_init   # traj: (n_steps+1, n_traj, 2)
 
 
def trajectory_straightness(traj):
    """
    Mesure la rectitude des trajectoires.
    = longueur de la corde / longueur du chemin
    Ratio = 1 : trajectoire parfaitement droite
    Ratio < 1 : trajectoire courbée
    """
    # Longueur de la corde (distance début → fin)
    chord  = np.linalg.norm(traj[-1] - traj[0], axis=-1)       # (n_traj,)
 
    # Longueur du chemin (somme des segments)
    diffs  = np.diff(traj, axis=0)                              # (n_steps, n_traj, 2)
    segs   = np.linalg.norm(diffs, axis=-1).sum(axis=0)         # (n_traj,)
 
    ratio  = chord / (segs + 1e-8)
    return ratio.mean()
 
 
def coverage_score(gen, target, n_bins=20):
    """
    Score de couverture : proportion de bins de la grille 2D
    qui contiennent au moins un sample généré ET un sample target.
    Score = 1 : couverture parfaite
    """
    all_pts  = np.concatenate([gen, target], axis=0)
    x_min, x_max = all_pts[:, 0].min() - 0.5, all_pts[:, 0].max() + 0.5
    y_min, y_max = all_pts[:, 1].min() - 0.5, all_pts[:, 1].max() + 0.5
 
    def to_bins(pts):
        bx = np.floor((pts[:, 0] - x_min) / (x_max - x_min) * n_bins).astype(int).clip(0, n_bins-1)
        by = np.floor((pts[:, 1] - y_min) / (y_max - y_min) * n_bins).astype(int).clip(0, n_bins-1)
        return set(zip(bx, by))
 
    bins_gen    = to_bins(gen)
    bins_target = to_bins(target)
    overlap     = len(bins_gen & bins_target)
    return overlap / len(bins_target)
 
 
def compute_mmd(x, y, sigma=1.0):
    """
    Maximum Mean Discrepancy avec noyau RBF.
    Mesure la distance entre deux distributions.
    MMD = 0 : distributions identiques
    """
    def rbf(a, b):
        d = torch.cdist(a, b) ** 2
        return torch.exp(-d / (2 * sigma ** 2))
 
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
 
    Kxx = rbf(x, x).mean()
    Kyy = rbf(y, y).mean()
    Kxy = rbf(x, y).mean()
    return (Kxx + Kyy - 2 * Kxy).item()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────────────────────────────────────
 
# Palette : du bleu froid (k=1) au orange chaud (k=inf)
def get_color(i, n):
    """Interpolation de couleur bleu → violet → orange selon i/n."""
    blues  = np.array([55, 138, 221]) / 255    # #378ADD
    orange = np.array([216, 90, 48])  / 255    # #D85A30
    t      = i / max(n - 1, 1)
    rgb    = (1 - t) * blues + t * orange
    return tuple(rgb)
 
 
def plot_ablation(x1_data_np, k_values, results, x0_fixed, x1_fixed):
    """
    results : dict {k: (model, losses, gen, traj, metrics)}
    """
    n_k   = len(k_values)
    fig   = plt.figure(figsize=(4 * n_k, 16))
    fig.suptitle(
        "Top-k Nearest Neighbor Coupling — Ablation sur k\n"
        "k=1 : NN strict   |   k=∞ : aléatoire",
        fontsize=12, y=0.99
    )
    gs = GridSpec(4, n_k, figure=fig, hspace=0.25, wspace=0.15)
    lim = 4.5
 
    # Collecte les métriques pour la figure de synthèse
    straightness_list = []
    coverage_list     = []
    mmd_list          = []
    final_loss_list   = []
 
    for col, k in enumerate(k_values):
        model, losses, gen, traj, metrics = results[k]
        color = get_color(col, n_k)
        label = f"k = {k}" if k < 512 else "k = ∞\n(random)"
 
        straightness_list.append(metrics["straightness"])
        coverage_list.append(metrics["coverage"])
        mmd_list.append(metrics["mmd"])
        final_loss_list.append(metrics["final_loss"])
 
        # ── Row 0 : samples générés + KDE ────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
 
        # KDE de la distribution générée
        try:
            kde = gaussian_kde(gen.T, bw_method=0.15)
            xx, yy = np.meshgrid(np.linspace(-lim, lim, 80),
                                 np.linspace(-lim, lim, 80))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=8, alpha=0.3,
                        cmap="Blues" if col == 0 else "Oranges" if col == n_k-1 else "Purples")
        except Exception:
            pass
 
        ax.scatter(x1_data_np[:, 0], x1_data_np[:, 1], s=2, alpha=0.1,
                   color="#D85A30", rasterized=True)
        ax.scatter(gen[:, 0], gen[:, 1], s=3, alpha=0.25,
                   color="#378ADD", rasterized=True)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(
            f"{label}\nMMD={metrics['mmd']:.3f}  cov={metrics['coverage']:.2f}",
            fontsize=8
        )
        ax.set_aspect("equal")
        ax.axis("off")
 
        # ── Row 1 : trajectoires ODE ──────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        for j in range(traj.shape[1]):
            pts = traj[:, j, :]
            ax.plot(pts[:, 0], pts[:, 1],
                    alpha=0.45, lw=0.9, color=color)
            ax.scatter(pts[0,  0], pts[0,  1],
                       s=15, color="#378ADD", zorder=4)
            ax.scatter(pts[-1, 0], pts[-1, 1],
                       s=15, color="#1D9E75", zorder=4, marker='x', linewidths=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(
            f"Trajectories\nstraightness={metrics['straightness']:.3f}",
            fontsize=8
        )
        ax.set_aspect("equal")
        ax.axis("off")
 
        # ── Row 2 : segments de couplage ─────────────────────────────────
        ax = fig.add_subplot(gs[2, col])
        coupling_fn = make_coupling_fn(k)
        with torch.no_grad():
            x0_c, x1_c = coupling_fn(x0_fixed.to(DEVICE), x1_fixed.to(DEVICE))
        x0_np = x0_c.cpu().numpy()
        x1_np = x1_c.cpu().numpy()
 
        mean_len = np.linalg.norm(x1_np - x0_np, axis=-1).mean()
 
        for j in range(len(x0_np)):
            ax.plot([x0_np[j, 0], x1_np[j, 0]],
                    [x0_np[j, 1], x1_np[j, 1]],
                    alpha=0.35, lw=0.8, color=color)
        ax.scatter(x0_np[:, 0], x0_np[:, 1], s=14, color="#378ADD", zorder=4)
        ax.scatter(x1_np[:, 0], x1_np[:, 1], s=14, color="#D85A30", zorder=4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"Coupling segments\nmean len={mean_len:.2f}", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
 
    plt.savefig("outputs/toy2d_topk_ablation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
 
    # ── Figure de synthèse : métriques vs k ──────────────────────────────
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig2.suptitle("Métriques vs k — Trade-off couverture / rectitude", fontsize=11)
 
    k_labels = [str(k) if k < 512 else "∞" for k in k_values]
    xs       = list(range(len(k_values)))
    colors_  = [get_color(i, len(k_values)) for i in xs]
 
    for ax, vals, title, better in zip(
        axes,
        [straightness_list, coverage_list, mmd_list, final_loss_list],
        ["Straightness ↑", "Coverage ↑", "MMD ↓", "FM Loss ↓"],
        [True, True, False, False]
    ):
        bars = ax.bar(xs, vals, color=colors_, alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(k_labels, fontsize=9)
        ax.set_xlabel("k")
        ax.set_title(title, fontsize=10)
 
        # Marque le meilleur k
        best_idx = int(np.argmax(vals)) if better else int(np.argmin(vals))
        ax.bar(best_idx, vals[best_idx],
               color=colors_[best_idx], edgecolor="#333", lw=2, alpha=1.0)
        ax.text(best_idx, vals[best_idx] * 1.01, "★",
                ha="center", fontsize=12, color="#333")
 
    plt.tight_layout()
    plt.savefig("outputs/toy2d_topk_metrics.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved toy2d_topk_ablation.png and toy2d_topk_metrics.png")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────
 
def run():
    N       = 30_000
    x1_data = sample_crescent(N)
 
    # Valeurs de k à tester — de NN strict à aléatoire
    # 512 = taille du batch = couplage aléatoire
    K_VALUES = [1, 8, 32, 128, 512]
 
    # Paires fixes pour visualiser les segments (même pour tous les k)
    torch.manual_seed(0)
    n_show   = 50
    x0_fixed = torch.randn(n_show, 2)
    idx      = torch.randint(0, N, (n_show,))
    x1_fixed = x1_data[idx]
 
    results = {}
 
    for k in K_VALUES:
        label = f"k={k}" if k < 512 else "k=∞ (random)"
        print(f"\n--- Training {label} ---")
 
        coupling_fn = make_coupling_fn(k, temperature=1.0)
        model, losses = train(coupling_fn, x1_data,
                              n_steps=8000, batch_size=512, device=DEVICE)
 
        gen, traj, x0_init = sample_with_traj(model, n_gen=2000, n_traj=30,
                                               device=DEVICE)
 
        metrics = {
            "straightness": trajectory_straightness(traj),
            "coverage":     coverage_score(gen, x1_data[:3000].numpy()),
            "mmd":          compute_mmd(gen, x1_data[:2000].numpy()),
            "final_loss":   float(np.mean(losses[-500:])),
        }
 
        print(f"  straightness : {metrics['straightness']:.3f}")
        print(f"  coverage     : {metrics['coverage']:.3f}")
        print(f"  MMD          : {metrics['mmd']:.4f}")
        print(f"  final loss   : {metrics['final_loss']:.5f}")
 
        results[k] = (model, losses, gen, traj, metrics)
 
    plot_ablation(
        x1_data[:3000].numpy(),
        K_VALUES, results,
        x0_fixed, x1_fixed
    )
 
 
if __name__ == "__main__":
    run()