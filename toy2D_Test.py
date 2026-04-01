"""
Validation empirique de la section théorique — toy 2D (version propre)

Trois choses à vérifier :

1. COROLLAIRE 1 (identité exacte)
   E_{pi_k}[||z1-z0||²]  =  E[||T*(z0)-z0||²]  +  E[V_k + B_k + C_k]
   Les deux membres sont calculés indépendamment sur les mêmes (z0, z1_ot).
   L'identité doit tenir à <1% près.

2. BORNES (Lemmes 1 & 2)
   V_k(z0) <= s_k(z0)²    (toujours vrai, Lemme 1)
   B_k(z0) <= s_k(z0)²    (vrai sous Assumption 1, Lemme 2)
   où s_k(z0) = max_j ||z^(j) - T̄_k(z0)|| = rayon centroïde

3. TAUX DE COVERAGE (Assumption 1)
   p_k = P(T*(z0) ∈ conv(N_k(z0)))
   Calculé via LP de faisabilité pour chaque (z0, k).
   On visualise aussi la distance ||T̄_k(z0) - T*(z0)|| vs s_k(z0)
   pour voir si T* est proche du centroïde.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment, linprog

os.makedirs("outputs", exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def sample_crescent(N, noise=0.08):
    theta = torch.FloatTensor(N).uniform_(0, np.pi)
    r     = 2.0 + torch.randn(N) * noise
    x     = r * torch.cos(theta)
    y     = r * torch.sin(theta) - 0.8
    return torch.stack([x, y], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# PLAN OT EMPIRIQUE
# ─────────────────────────────────────────────────────────────────────────────

def compute_OT(z0, z1):
    """
    Plan OT empirique via algorithme hongrois sur les mêmes z0, z1.
    Retourne T_star (B,d) et le coût moyen W2_sq.
    """
    C = torch.cdist(z0, z1).pow(2).numpy()
    rows, cols = linear_sum_assignment(C)
    T_star = z1[cols]
    W2_sq  = float(C[rows, cols].mean())
    return T_star, W2_sq


# ─────────────────────────────────────────────────────────────────────────────
# QUANTITÉS CONDITIONNELLES
# ─────────────────────────────────────────────────────────────────────────────

def compute_quantities(z0, z1_pool, k, T_star):
    """
    Pour chaque z0_i, calcule pointwise :

      T̄_k(z0)  : centroïde des k-NN de z0 dans z1_pool
      s_k(z0)  : rayon centroïde = max_j ||z^(j) - T̄_k||
                 → borne des Lemmes 1 & 2
      V_k(z0)  : variance conditionnelle = (1/k) Σ ||z^(j) - T̄_k||²  ≤ s_k²
      B_k(z0)  : biais centroïde = ||T̄_k - T*(z0)||²   ≤ s_k² (sous Ass.1)
      C_k(z0)  : cross term = 2 <T*(z0)-z0, T̄_k-T*(z0)>

    Args:
        z0       : (B, d)
        z1_pool  : (N, d)  — même ensemble que celui utilisé pour l'OT
        k        : int
        T_star   : (B, d)  — T*(z0_i) via plan OT

    Returns: dict de tenseurs (B,)
    """
    B, N = z0.shape[0], z1_pool.shape[0]
    k_eff = min(k, N)

    # k-NN
    dists  = torch.cdist(z0, z1_pool)                          # (B, N)
    _, idx = dists.topk(k_eff, dim=1, largest=False)           # (B, k)
    nbrs   = z1_pool[idx.reshape(-1)].reshape(B, k_eff, -1)   # (B, k, d)

    # Centroïde
    T_bar = nbrs.mean(dim=1)                                    # (B, d)

    # s_k = rayon centroïde (borne des lemmes)
    diff  = nbrs - T_bar.unsqueeze(1)                          # (B, k, d)
    s_k   = diff.norm(dim=-1).max(dim=1).values                # (B,)

    # V_k = variance conditionnelle
    V_k   = diff.pow(2).sum(dim=-1).mean(dim=1)                # (B,)

    # B_k = biais centroïde
    eps   = T_bar - T_star                                      # (B, d)
    B_k   = eps.pow(2).sum(dim=-1)                             # (B,)

    # C_k = cross term
    disp  = T_star - z0                                         # (B, d)
    C_k   = 2.0 * (disp * eps).sum(dim=-1)                    # (B,)

    return {
        "T_bar": T_bar,   # (B, d)
        "nbrs":  nbrs,    # (B, k, d)
        "s_k":   s_k,     # (B,)  borne
        "V_k":   V_k,     # (B,)
        "B_k":   B_k,     # (B,)
        "C_k":   C_k,     # (B,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# COROLLAIRE 1
# ─────────────────────────────────────────────────────────────────────────────

def verify_corollary1(z0, z1_pool, k_values, T_star, W2_sq, n_rep=100):
    """
    Vérifie l'identité exacte :
      E[||z1-z0||²]  =  E[||T*-z0||²]  +  E[V_k + B_k + C_k]
      LHS             =  W2_sq (coût OT)  +  RHS

    z1_pool DOIT être le même ensemble que celui utilisé pour calculer
    W2_sq — sinon le k-NN peut trouver des voisins plus proches que
    ce que l'OT peut faire sur z1_pool, rendant LHS < W2_sq.
    """
    B = z0.shape[0]
    N = z1_pool.shape[0]
    print("\n" + "="*60)
    print("COROLLAIRE 1 — identité exacte")
    print("  E[||z1-z0||²] = W2² + E[V_k + B_k + C_k]")
    print(f"  W2² = {W2_sq:.5f}")
    print(f"\n  {'k':>5}  {'LHS':>10}  {'W2²+RHS':>10}  {'|err|':>8}  {'err%':>6}")

    rows = []
    for k in k_values:
        k_eff = min(k, N)
        q     = compute_quantities(z0, z1_pool, k, T_star)

        # LHS : E[||z1-z0||²] avec z1 ~ U(N_k(z0)), n_rep tirages
        traj2 = []
        for _ in range(n_rep):
            ch    = torch.randint(0, k_eff, (B,))
            z1_k  = q["nbrs"][torch.arange(B), ch]
            traj2.append((z1_k - z0).pow(2).sum(dim=-1))
        lhs = torch.stack(traj2).mean().item()

        # RHS : W2² + E[V_k + B_k + C_k]
        rhs_excess = (q["V_k"] + q["B_k"] + q["C_k"]).mean().item()
        rhs        = W2_sq + rhs_excess

        err    = abs(lhs - rhs)
        err_pc = 100 * err / (abs(lhs) + 1e-8)
        ok     = "✓" if err_pc < 1.0 else "⚠"
        print(f"  {k:>5}  {lhs:>10.5f}  {rhs:>10.5f}  {err:>8.5f}  {err_pc:>5.2f}%  {ok}")
        rows.append({"k": k, "lhs": lhs, "rhs": rhs,
                     "err": err, "err_pc": err_pc})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# BORNES
# ─────────────────────────────────────────────────────────────────────────────

def verify_bounds(z0, z1_pool, k_values, T_star):
    """
    Vérifie pointwise :
      V_k(z0) <= s_k(z0)²   (Lemme 1, toujours vrai)
      B_k(z0) <= s_k(z0)²   (Lemme 2, vrai seulement sous Ass. 1)
    Retourne les fractions de z0 satisfaisant chaque borne.
    """
    print("\n" + "="*60)
    print("BORNES — V_k ≤ s_k²  et  B_k ≤ s_k²")
    print(f"\n  {'k':>5}  {'E[V_k]':>8}  {'E[B_k]':>8}  {'E[s_k²]':>8}"
          f"  {'V≤s²':>6}  {'B≤s²':>6}")

    rows = []
    for k in k_values:
        q    = compute_quantities(z0, z1_pool, k, T_star)
        s2   = q["s_k"].pow(2)
        V_ok = (q["V_k"] <= s2 + 1e-6).float().mean().item()
        B_ok = (q["B_k"] <= s2 + 1e-6).float().mean().item()
        print(f"  {k:>5}  {q['V_k'].mean():>8.4f}  {q['B_k'].mean():>8.4f}"
              f"  {s2.mean():>8.4f}  {V_ok:>6.3f}  {B_ok:>6.3f}")
        rows.append({"k": k, "EV": q["V_k"].mean().item(),
                     "EB": q["B_k"].mean().item(),
                     "Es2": s2.mean().item(),
                     "V_ok": V_ok, "B_ok": B_ok})
    print("  V≤s² et B≤s² : fraction des z0 satisfaisant la borne")
    print("  B≤s² < 1.0 indique des z0 où Assumption 1 est violée")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# TAUX DE COVERAGE + PROXIMITÉ T* / CENTROÏDE
# ─────────────────────────────────────────────────────────────────────────────

def verify_coverage(z0, z1_pool, k_values, T_star):
    """
    Pour chaque k :
      p_k  : fraction de z0 avec T*(z0) ∈ conv(N_k(z0))  [LP]
      dist  : ||T̄_k(z0) - T*(z0)||  (distance centroïde → T*)
      s_k   : rayon centroïde (borne)
      ratio : dist / s_k  (< 1 ssi T* dans la boule B(T̄_k, s_k))
    """
    print("\n" + "="*60)
    print("COVERAGE — T*(z0) ∈ conv(N_k(z0))  et  dist(T̄_k, T*) / s_k")
    print(f"\n  {'k':>5}  {'p_k':>6}  {'E[dist]':>9}  {'E[s_k]':>8}"
          f"  {'ratio':>7}  {'dist<s_k':>8}")

    B = z0.shape[0]
    rows = []
    for k in k_values:
        k_eff = min(k, z1_pool.shape[0])
        q     = compute_quantities(z0, z1_pool, k, T_star)

        # Distance centroïde → T*
        dist  = q["B_k"].sqrt()        # (B,) = ||T̄_k - T*||
        s_k   = q["s_k"]               # (B,)
        ratio = (dist / (s_k + 1e-8)).mean().item()

        # Fraction T* dans B(T̄_k, s_k) — condition des Lemmes
        in_ball = (dist <= s_k + 1e-6).float().mean().item()

        # LP : T*(z0) ∈ conv(N_k(z0))
        nbrs_np  = q["nbrs"].numpy()   # (B, k, d)
        Ts_np    = T_star.numpy()      # (B, d)
        in_conv  = np.zeros(B, dtype=bool)
        for i in range(B):
            V   = nbrs_np[i].T         # (d, k)
            t   = Ts_np[i]             # (d,)
            A   = np.vstack([V, np.ones((1, k_eff))])
            b   = np.append(t, 1.0)
            res = linprog(np.zeros(k_eff), A_eq=A, b_eq=b,
                          bounds=[(0, None)] * k_eff,
                          method="highs", options={"disp": False})
            in_conv[i] = (res.status == 0)
        p_k = float(in_conv.mean())

        print(f"  {k:>5}  {p_k:>6.3f}  {dist.mean():>9.4f}  {s_k.mean():>8.4f}"
              f"  {ratio:>7.3f}  {in_ball:>8.3f}")
        rows.append({"k": k, "p_k": p_k,
                     "dist": dist.mean().item(),
                     "s_k": s_k.mean().item(),
                     "ratio": ratio,
                     "in_ball": in_ball,
                     "in_conv": in_conv,
                     "dist_all": dist, "s_k_all": s_k,
                     "T_bar": q["T_bar"]})
    print("  ratio = E[dist/s_k] : < 1 → T* typiquement dans B(T̄_k, s_k)")
    print("  in_ball : fraction T* dans B(T̄_k, s_k)  (condition Lemme 2)")
    print("  p_k     : fraction T* dans conv(N_k)     (Assumption 1)")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(z0, z1_pool, z1_full, T_star, k_values,
             rows_cor, rows_bnd, rows_cov):

    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    k_arr = np.array(k_values, dtype=float)

    # Figure : panels 1&2 colonne 0, géométrie colonne 1, trajectoires colonne 2
    fig = plt.figure(figsize=(30, 9))
    fig.suptitle("Validation théorique — toy 2D", fontsize=13, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           width_ratios=[1, 2, 2],
                           hspace=0.45, wspace=0.35)

    ax_cor = fig.add_subplot(gs[0, 0])   # Panel 1 : Corollaire 1
    ax_bnd = fig.add_subplot(gs[1, 0])   # Panel 2 : Bornes
    ax_geo = fig.add_subplot(gs[:, 1])   # Panel 3 : Géométrie (grande)
    ax_traj = fig.add_subplot(gs[:, 2])  # Panel 4 : Trajectoires

    # ── Panel 1 : Corollaire 1 ───────────────────────────────────────────
    lhs = np.array([r["lhs"]    for r in rows_cor])
    rhs = np.array([r["rhs"]    for r in rows_cor])
    epc = np.array([r["err_pc"] for r in rows_cor])
    xs  = np.arange(len(k_values))

    ax_cor.plot(xs, lhs, "o-", color="#378ADD", lw=2, ms=6,
                label="LHS : $E[||z_1-z_0||^2]$")
    ax_cor.plot(xs, rhs, "s--", color="#D85A30", lw=2, ms=6,
                label="RHS : $W_2^2 + E[V_k+B_k+C_k]$")
    ax2 = ax_cor.twinx()
    ax2.bar(xs, epc, alpha=0.2, color="green")
    ax2.set_ylabel("erreur %", color="green", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(0, max(epc.max() * 3, 2))
    ax_cor.set_xticks(xs)
    ax_cor.set_xticklabels([str(k) for k in k_values], fontsize=8)
    ax_cor.set_xlabel("k"); ax_cor.set_ylabel("coût")
    ax_cor.set_title("Panel 1 — Corollaire 1\nLHS = RHS à <1% ?")
    ax_cor.legend(fontsize=7); ax_cor.grid(True, alpha=0.3, axis="y")

    # ── Panel 2 : Bornes ─────────────────────────────────────────────────
    EV  = np.array([r["EV"]   for r in rows_bnd])
    EB  = np.array([r["EB"]   for r in rows_bnd])
    Es2 = np.array([r["Es2"]  for r in rows_bnd])
    Vok = np.array([r["V_ok"] for r in rows_bnd])
    Bok = np.array([r["B_ok"] for r in rows_bnd])

    ax_bnd.semilogy(k_arr, EV,  "o-",  color="#378ADD", lw=2, ms=6,
                    label="$E[V_k]$")
    ax_bnd.semilogy(k_arr, EB,  "s-",  color="#D85A30", lw=2, ms=6,
                    label="$E[B_k]$")
    ax_bnd.semilogy(k_arr, Es2, "D--", color="#5B2D8E", lw=1.5, ms=5,
                    label="$E[s_k^2]$ (borne)")
    for i, k in enumerate(k_values):
        ax_bnd.annotate(f"V:{Vok[i]:.2f}\nB:{Bok[i]:.2f}",
                        (k_arr[i], max(EV[i], EB[i])),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6, color="#333")
    ax_bnd.set_xlabel("k"); ax_bnd.set_ylabel("valeur (log)")
    ax_bnd.set_title("Panel 2 — Bornes\n$V_k <= s_k^2$ | $B_k <= s_k^2$ (Ass.1)")
    ax_bnd.legend(fontsize=7); ax_bnd.grid(True, alpha=0.3, which="both")

    # ── Panel géométrie : B(T̄_k, s_k) et T*(z0), k=16 ──────────────────
    k_vis = 16
    r_vis = rows_cov[k_values.index(k_vis)]

    Ts_np  = T_star.numpy()
    z1_np  = z1_full.numpy()
    Tb_np  = r_vis["T_bar"].numpy()
    dist_a = r_vis["dist_all"].numpy()
    sk_a   = r_vis["s_k_all"].numpy()
    in_ball = (dist_a <= sk_a + 1e-6)

    # Dataset en fond
    ax_geo.scatter(z1_np[:, 0], z1_np[:, 1], s=3, alpha=0.07,
                   color="#AAAAAA", zorder=1)

    # Sélection N_SHOW : moitié dans boule, moitié hors
    N_SHOW  = 10
    rng     = np.random.default_rng(2)
    out_idx = np.where(~in_ball)[0]
    in_idx  = np.where(in_ball)[0]
    out_sel = out_idx[rng.choice(len(out_idx), min(N_SHOW//2, len(out_idx)),
                                 replace=False)] if len(out_idx) else []
    in_sel  = in_idx[rng.choice(len(in_idx),   min(N_SHOW//2, len(in_idx)),
                                 replace=False)] if len(in_idx)  else []

    k_eff_vis = min(k_vis, z1_full.shape[0])
    pt_num = 1
    for sel, color, inside_ball in [(in_sel, "#1D9E75", True),
                                     (out_sel, "#D85A30", False)]:
        for i in sel:
            tsi   = Ts_np[i]
            centi = Tb_np[i]
            sk    = float(sk_a[i])
            d     = float(dist_a[i])
            z0i   = z0[i].numpy()

            # k voisins (croix légères)
            dists_vis  = torch.cdist(z0[i:i+1], z1_full)
            _, idx_vis = dists_vis.topk(k_eff_vis, dim=1, largest=False)
            nbrs_vis   = z1_full[idx_vis[0]].numpy()
            ax_geo.scatter(nbrs_vis[:, 0], nbrs_vis[:, 1],
                           s=14, color=color, alpha=0.30,
                           marker="x", linewidths=0.8, zorder=3)

            # Boule B(T̄_k, s_k)
            circ = patches.Circle(centi, sk, fill=False,
                                   edgecolor=color, lw=1.3,
                                   linestyle="--", alpha=0.85, zorder=2)
            ax_geo.add_patch(circ)

            # z0 (losange creux)
            ax_geo.scatter(*z0i, s=60, facecolors="none", edgecolors=color,
                           linewidths=1.3, zorder=4, marker="D")

            # Centroïde (carré plein)
            ax_geo.scatter(*centi, s=70, color=color, zorder=5,
                           marker="s", edgecolors="black", linewidths=0.7)

            # Numéro sur z0 et centroïde
            ax_geo.annotate(str(pt_num), xy=z0i,
                            textcoords="offset points", xytext=(-9, 4),
                            fontsize=8, fontweight="bold", color=color, zorder=8)
            ax_geo.annotate(str(pt_num), xy=centi,
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=8, fontweight="bold", color=color, zorder=8)

            # Flèche centroïde → T*
            ax_geo.annotate("", xy=tsi, xytext=centi,
                            arrowprops=dict(arrowstyle="->", color=color,
                                            lw=1.4, alpha=0.9), zorder=6)

            # T*(z0)
            ax_geo.scatter(*tsi, s=90, color=color, zorder=7,
                           marker="^" if inside_ball else "*",
                           edgecolors="black", linewidths=0.6)
            pt_num += 1

    legend_elems = [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="none",
               markeredgecolor="#555", markeredgewidth=1.3, markersize=8,
               label="$z_0$ (point source)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#1D9E75",
               markeredgecolor="black", markersize=8,
               label="centroide — $T^*$ dans boule"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#D85A30",
               markeredgecolor="black", markersize=8,
               label="centroide — $T^*$ hors boule"),
        Line2D([0],[0], linestyle="--", color="#555", lw=1.5,
               label="boule $B(\\bar{T}_k,\\, s_k)$"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#1D9E75",
               markeredgecolor="black", markersize=8,
               label="$T^*(z_0)$ dans boule"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#D85A30",
               markeredgecolor="black", markersize=10,
               label="$T^*(z_0)$ hors boule"),
        Line2D([0],[0], marker="x", color="#888", markersize=7,
               linestyle="None", label=f"$k={k_vis}$ voisins"),
    ]
    ax_geo.legend(handles=legend_elems, fontsize=8, loc="lower right",
                  framealpha=0.9)
    ax_geo.set_aspect("equal")
    ax_geo.grid(True, alpha=0.15)
    ax_geo.set_title(
        f"Panel 3 — Géométrie locale ($k={k_vis}$, {N_SHOW} exemples, "
        f"kNN sur {len(z1_full)} pts)\n"
        "losange=$z_0$ | carré=centroïde | tirets=$s_k$ | flèche=dist(centroïde,$T^*$)",
        fontsize=10
    )

    # ── Panel 4 : Trajectoires kNN vs optimale ──────────────────────────
    # Pour N_TRAJ points z0, on tire un voisin z1 ~ U(N_k(z0)) et on
    # trace la trajectoire linéaire z0→z1 (ce que le réseau apprend)
    # vs la trajectoire optimale z0→T*(z0) (ce qu'on voudrait).
    # On calcule l'angle entre les deux directions.
    k_traj   = 8
    N_TRAJ   = 30
    rng_traj = np.random.default_rng(7)

    # Dataset en fond
    ax_traj.scatter(z1_np[:, 0], z1_np[:, 1], s=3, alpha=0.07,
                    color="#AAAAAA", zorder=1)

    # Sélection aléatoire de N_TRAJ points z0
    sel_traj = rng_traj.choice(len(z0), N_TRAJ, replace=False)

    k_eff_traj = min(k_traj, z1_full.shape[0])
    angles_deg = []

    for i in sel_traj:
        z0i = z0[i].numpy()
        tsi = T_star.numpy()[i]        # T*(z0) — cible optimale

        # Tirer un voisin uniformément parmi les k-NN
        dists_t    = torch.cdist(z0[i:i+1], z1_full)
        _, idx_t   = dists_t.topk(k_eff_traj, dim=1, largest=False)
        chosen     = rng_traj.integers(0, k_eff_traj)
        z1_knn     = z1_full[idx_t[0, chosen]].numpy()  # voisin tiré

        # Directions normalisées
        dir_knn = z1_knn - z0i
        dir_opt = tsi    - z0i
        norm_knn = np.linalg.norm(dir_knn) + 1e-8
        norm_opt = np.linalg.norm(dir_opt) + 1e-8
        cos_a    = np.clip(np.dot(dir_knn, dir_opt) / (norm_knn * norm_opt), -1, 1)
        angle    = float(np.degrees(np.arccos(cos_a)))
        angles_deg.append(angle)

        # Couleur selon l'angle : vert=bien aligné, orange=mauvais
        c_traj = plt.cm.RdYlGn(1.0 - angle / 90.0)

        # Trajectoire kNN (trait plein)
        ax_traj.annotate("", xy=z1_knn, xytext=z0i,
                         arrowprops=dict(arrowstyle="->", color=c_traj,
                                         lw=1.2, alpha=0.75), zorder=3)
        # Trajectoire optimale (tirets noirs fins)
        ax_traj.annotate("", xy=tsi, xytext=z0i,
                         arrowprops=dict(arrowstyle="->", color="black",
                                         lw=0.7, alpha=0.4,
                                         linestyle="dashed"), zorder=4)
        # Angle annoté au milieu de la trajectoire kNN
        mid = (z0i + z1_knn) / 2
        ax_traj.annotate(f"{angle:.0f}°", xy=mid,
                         fontsize=5, color=c_traj, ha="center",
                         zorder=5, alpha=0.85)

        # z0
        ax_traj.scatter(*z0i, s=20, color="black", zorder=6, marker="o")

    mean_angle = float(np.mean(angles_deg))
    med_angle  = float(np.median(angles_deg))

    from matplotlib.lines import Line2D as L2D
    leg_traj = [
        L2D([0],[0], color="green",  lw=1.5, label="trajectoire kNN (bien alignée)"),
        L2D([0],[0], color="orange", lw=1.5, label="trajectoire kNN (mal alignée)"),
        L2D([0],[0], color="black",  lw=0.8, linestyle="--",
            label="trajectoire optimale $z_0 \\to T^*(z_0)$"),
        L2D([0],[0], marker="o", color="w", markerfacecolor="black",
            markersize=6, label="$z_0$"),
    ]
    ax_traj.legend(handles=leg_traj, fontsize=7, loc="lower right", framealpha=0.9)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.15)
    ax_traj.set_title(
        f"Panel 4 — Trajectoires kNN vs optimales ($k={k_traj}$, {N_TRAJ} ex.)\n"
        f"couleur = angle(kNN, optimal) | moy={mean_angle:.1f}° | med={med_angle:.1f}°\n"
        "vert=bien aligné, rouge=mal aligné | tirets=trajectoire optimale",
        fontsize=9
    )

    # ── Sauvegarde 1 : figure globale ───────────────────────────────────
    plt.savefig("outputs/toy2d_theory_clean.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved outputs/toy2d_theory_clean.png")

    # ── Sauvegarde 2a : Panel 1 seul (Corollaire 1) ─────────────────────
    fig_p1, ax12a = plt.subplots(1, 1, figsize=(6, 5))
    fig_p1.suptitle("Corollaire 1 — Identité exacte", fontsize=10)

    # Panel 1
    ax12a.plot(xs, lhs, "o-", color="#378ADD", lw=2, ms=6,
               label="LHS : $E[||z_1-z_0||^2]$")
    ax12a.plot(xs, rhs, "s--", color="#D85A30", lw=2, ms=6,
               label="RHS : $W_2^2 + E[V_k+B_k+C_k]$")
    ax12a_r = ax12a.twinx()
    ax12a_r.bar(xs, epc, alpha=0.2, color="green")
    ax12a_r.set_ylabel("erreur %", color="green", fontsize=8)
    ax12a_r.tick_params(axis="y", labelcolor="green")
    ax12a_r.set_ylim(0, max(epc.max() * 3, 2))
    ax12a.set_xticks(xs); ax12a.set_xticklabels([str(k) for k in k_values], fontsize=8)
    ax12a.set_xlabel("k"); ax12a.set_ylabel("coût")
    ax12a.set_title("Panel 1 — Corollaire 1\nLHS = RHS à <1% ?")
    ax12a.legend(fontsize=7); ax12a.grid(True, alpha=0.3, axis="y")

    fig_p1.tight_layout()
    fig_p1.savefig("outputs/toy2d_panel1_corollary.png", dpi=150, bbox_inches="tight")
    plt.close(fig_p1)
    print("Saved outputs/toy2d_panel1_corollary.png")

    # ── Sauvegarde 2b : Panel 2 seul (Bornes) ────────────────────────────
    fig_p2, ax12b = plt.subplots(1, 1, figsize=(6, 5))
    fig_p2.suptitle("Bornes — Lemmes 1 & 2", fontsize=10)

    ax12b.semilogy(k_arr, EV, "o-", color="#378ADD", lw=2, ms=6, label="$E[V_k]$")
    ax12b.semilogy(k_arr, EB, "s-", color="#D85A30", lw=2, ms=6, label="$E[B_k]$")
    ax12b.semilogy(k_arr, Es2, "D--", color="#5B2D8E", lw=1.5, ms=5,
                   label="$E[s_k^2]$ (borne)")
    for i, k in enumerate(k_values):
        ax12b.annotate(f"V:{Vok[i]:.2f}\nB:{Bok[i]:.2f}",
                       (k_arr[i], max(EV[i], EB[i])),
                       textcoords="offset points", xytext=(0, 6),
                       ha="center", fontsize=6, color="#333")
    ax12b.set_xlabel("k"); ax12b.set_ylabel("valeur (log)")
    ax12b.set_title("Panel 2 — Bornes\n$V_k <= s_k^2$ | $B_k <= s_k^2$ (Ass.1)")
    ax12b.legend(fontsize=7); ax12b.grid(True, alpha=0.3, which="both")

    fig_p2.tight_layout()
    fig_p2.savefig("outputs/toy2d_panel2_bounds.png", dpi=150, bbox_inches="tight")
    plt.close(fig_p2)
    print("Saved outputs/toy2d_panel2_bounds.png")

    # ── Sauvegarde 2c : Coverage (p_k, ratio, in_ball vs k) ─────────────
    fig_cov, ax_cov = plt.subplots(1, 1, figsize=(7, 5))

    k_arr_cov = np.array([r["k"]       for r in rows_cov], dtype=float)
    pk_arr    = np.array([r["p_k"]     for r in rows_cov])
    ib_arr    = np.array([r["in_ball"] for r in rows_cov])
    rt_arr    = np.array([r["ratio"]   for r in rows_cov])

    ax_cov.plot(k_arr_cov, pk_arr, "o-", color="#378ADD", lw=2, ms=7,
                label=r"$p_k$ = P($T^*\in\mathrm{conv}(\mathcal{N}_k)$)  [LP]")
    ax_cov.plot(k_arr_cov, ib_arr, "s--", color="#1D9E75", lw=2, ms=7,
                label=r"P($\|\bar{T}_k - T^*\| \leq s_k$)  [boule]")
    ax_cov.axhline(1.0, color="gray", lw=1, linestyle=":", label="couverture parfaite")
    ax_cov.set_ylim(0, 1.12)

    ax_cov2 = ax_cov.twinx()
    # Clipper le ratio à 2.5 pour éviter que k=2 (s_k minuscule) écrase le plot
    rt_clip = np.clip(rt_arr, 0, 2.5)
    exploded = rt_arr > 2.5   # marquer les valeurs tronquées
    ax_cov2.plot(k_arr_cov, rt_clip, "^:", color="#D85A30", lw=1.8, ms=7,
                 label=r"ratio $E[\|\bar{T}_k - T^*\| / s_k]$ (max=2.5)")
    ax_cov2.axhline(1.0, color="#D85A30", lw=0.6, linestyle="--")
    # Annoter les valeurs tronquées avec la vraie valeur
    for ki, rt, clipped in zip(k_arr_cov, rt_arr, exploded):
        if clipped:
            ax_cov2.annotate(f"{rt:.1f}*", (ki, 2.5),
                             textcoords="offset points", xytext=(0, 5),
                             ha="center", fontsize=7, color="#D85A30",
                             fontstyle="italic")
    ax_cov2.set_ylim(0, 3.0)
    ax_cov2.set_ylabel("ratio dist / $s_k$ (tronqué à 2.5)", color="#D85A30", fontsize=9)
    ax_cov2.tick_params(axis="y", labelcolor="#D85A30")

    # Annoter les valeurs de p_k
    for ki, pk in zip(k_arr_cov, pk_arr):
        ax_cov.annotate(f"{pk:.2f}", (ki, pk),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="#378ADD")

    ax_cov.set_xlabel("$k$"); ax_cov.set_ylabel("fraction de $z_0$")
    ax_cov.set_title(
        "Coverage — Assumption 1 et proximité $T^*$ / centroïde\n"
        "ratio < 1 $\\Rightarrow$ $T^*(z_0)$ dans $B(\\bar{T}_k, s_k)$",
        fontsize=10
    )
    lines1, lab1 = ax_cov.get_legend_handles_labels()
    lines2, lab2 = ax_cov2.get_legend_handles_labels()
    ax_cov.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="lower right")
    ax_cov.grid(True, alpha=0.3)

    fig_cov.tight_layout()
    fig_cov.savefig("outputs/toy2d_panel_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cov)
    print("Saved outputs/toy2d_panel_coverage.png")

    # ── Sauvegarde 3 : Panel 3 seul (Géométrie) ──────────────────────────
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 9))

    ax3.scatter(z1_np[:, 0], z1_np[:, 1], s=3, alpha=0.07,
                color="#AAAAAA", zorder=1)
    pt_num3 = 1
    for sel, color, inside_ball in [(in_sel, "#1D9E75", True),
                                     (out_sel, "#D85A30", False)]:
        for i in sel:
            tsi   = Ts_np[i]; centi = Tb_np[i]
            sk    = float(sk_a[i]); d = float(dist_a[i])
            z0i   = z0[i].numpy()
            dists_vis  = torch.cdist(z0[i:i+1], z1_full)
            _, idx_vis = dists_vis.topk(k_eff_vis, dim=1, largest=False)
            nbrs_vis   = z1_full[idx_vis[0]].numpy()
            ax3.scatter(nbrs_vis[:, 0], nbrs_vis[:, 1],
                        s=14, color=color, alpha=0.30,
                        marker="x", linewidths=0.8, zorder=3)
            circ = patches.Circle(centi, sk, fill=False, edgecolor=color,
                                   lw=1.3, linestyle="--", alpha=0.85, zorder=2)
            ax3.add_patch(circ)
            ax3.scatter(*z0i, s=60, facecolors="none", edgecolors=color,
                        linewidths=1.3, zorder=4, marker="D")
            ax3.scatter(*centi, s=70, color=color, zorder=5,
                        marker="s", edgecolors="black", linewidths=0.7)
            ax3.annotate(str(pt_num3), xy=z0i, textcoords="offset points",
                         xytext=(-9, 4), fontsize=9, fontweight="bold",
                         color=color, zorder=8)
            ax3.annotate(str(pt_num3), xy=centi, textcoords="offset points",
                         xytext=(5, 4), fontsize=9, fontweight="bold",
                         color=color, zorder=8)
            ax3.annotate("", xy=tsi, xytext=centi,
                         arrowprops=dict(arrowstyle="->", color=color,
                                         lw=1.4, alpha=0.9), zorder=6)
            ax3.scatter(*tsi, s=90, color=color, zorder=7,
                        marker="^" if inside_ball else "*",
                        edgecolors="black", linewidths=0.6)
            pt_num3 += 1

    legend3 = [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="none",
               markeredgecolor="#555", markeredgewidth=1.3, markersize=9,
               label="$z_0$ (point source)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#1D9E75",
               markeredgecolor="black", markersize=9,
               label="centroide — $T^*$ dans boule"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#D85A30",
               markeredgecolor="black", markersize=9,
               label="centroide — $T^*$ hors boule"),
        Line2D([0],[0], linestyle="--", color="#555", lw=1.5,
               label="boule $B(\\bar{T}_k, s_k)$"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#1D9E75",
               markeredgecolor="black", markersize=9,
               label="$T^*(z_0)$ dans boule"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#D85A30",
               markeredgecolor="black", markersize=11,
               label="$T^*(z_0)$ hors boule"),
        Line2D([0],[0], marker="x", color="#888", markersize=8,
               linestyle="None", label=f"$k={k_vis}$ voisins"),
    ]
    ax3.legend(handles=legend3, fontsize=9, loc="lower right", framealpha=0.9)
    ax3.set_aspect("equal"); ax3.grid(True, alpha=0.15)
    ax3.set_title(
        f"Panel 3 — Géométrie locale ($k={k_vis}$, {N_SHOW} exemples, "
        f"kNN sur {len(z1_full)} pts)\n"
        "losange=$z_0$ | carré=centroïde | tirets=$s_k$ | flèche=dist(centroïde,$T^*$)",
        fontsize=11
    )
    fig3.tight_layout()
    fig3.savefig("outputs/toy2d_panel3_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved outputs/toy2d_panel3_geometry.png")

    # ── Sauvegarde 4 : Panel 4 — kNN vs indépendant côte à côte ────────
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(18, 9))
    fig4.suptitle(
        f"Trajectoires kNN ($k={k_traj}$) vs couplage indépendant — {N_TRAJ} exemples",
        fontsize=12
    )

    N_z1_full = len(z1_full)

    def draw_trajectories(ax, use_knn, title_prefix):
        ax.scatter(z1_np[:, 0], z1_np[:, 1], s=3, alpha=0.07,
                   color="#AAAAAA", zorder=1)
        angles  = []
        rng_draw = np.random.default_rng(7)   # même seed pour comparer à l’identique
        for i in sel_traj:
            z0i = z0[i].numpy()
            tsi = T_star.numpy()[i]

            if use_knn:
                dists_t    = torch.cdist(z0[i:i+1], z1_full)
                _, idx_t   = dists_t.topk(k_eff_traj, dim=1, largest=False)
                chosen     = rng_draw.integers(0, k_eff_traj)
                z1_sampled = z1_full[idx_t[0, chosen]].numpy()
            else:
                # Couplage indépendant : tirage uniformément dans tout le dataset
                chosen     = rng_draw.integers(0, N_z1_full)
                z1_sampled = z1_full[chosen].numpy()

            dir_s  = z1_sampled - z0i
            dir_o  = tsi - z0i
            norm_s = np.linalg.norm(dir_s) + 1e-8
            norm_o = np.linalg.norm(dir_o) + 1e-8
            cos_a  = np.clip(np.dot(dir_s, dir_o) / (norm_s * norm_o), -1, 1)
            angle  = float(np.degrees(np.arccos(cos_a)))
            angles.append(angle)

            c_traj = plt.cm.RdYlGn(1.0 - angle / 90.0)
            ax.annotate("", xy=z1_sampled, xytext=z0i,
                        arrowprops=dict(arrowstyle="->", color=c_traj,
                                        lw=1.3, alpha=0.8), zorder=3)
            ax.annotate("", xy=tsi, xytext=z0i,
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=0.8, alpha=0.4,
                                        linestyle="dashed"), zorder=4)
            mid = (z0i + z1_sampled) / 2
            ax.annotate(f"{angle:.0f}°", xy=mid,
                        fontsize=6, color=c_traj, ha="center",
                        zorder=5, alpha=0.9)
            ax.scatter(*z0i, s=22, color="black", zorder=6, marker="o")

        mean_a = float(np.mean(angles))
        med_a  = float(np.median(angles))
        coupling_str = f"kNN ($k={k_traj}$)" if use_knn else "indépendant (k=N)"
        leg = [
            L2D([0],[0], color="green",  lw=1.5, label="trajectoire (bien alignée)"),
            L2D([0],[0], color="orange", lw=1.5, label="trajectoire (mal alignée)"),
            L2D([0],[0], color="black",  lw=0.9, linestyle="--",
                label="trajectoire optimale $z_0 \\to T^*(z_0)$"),
            L2D([0],[0], marker="o", color="w", markerfacecolor="black",
                markersize=7, label="$z_0$"),
        ]
        ax.legend(handles=leg, fontsize=9, loc="lower right", framealpha=0.9)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_title(
            f"{title_prefix} — couplage {coupling_str}\n"
            f"moy={mean_a:.1f}° | med={med_a:.1f}° | couleur=angle(couplage, optimal)",
            fontsize=11
        )
        return mean_a, med_a

    mean_knn, med_knn = draw_trajectories(ax4a, use_knn=True,  title_prefix="kNN coupling")
    mean_ind, med_ind = draw_trajectories(ax4b, use_knn=False, title_prefix="Indépendant")

    fig4.tight_layout()
    fig4.savefig("outputs/toy2d_panel4_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved outputs/toy2d_panel4_trajectories.png")
    print(f"  kNN     : moy={mean_knn:.1f}°  med={med_knn:.1f}°")
    print(f"  Indép.  : moy={mean_ind:.1f}°  med={med_ind:.1f}°")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    N_DATA   = 5_000   # taille dataset µ1
    N_OT     = 400     # points pour le plan OT (O(n³))
    K_VALUES = [2, 4, 8, 16, 32, 64, 128]
    D        = 2

    print(f"Dataset µ1 : crescent, N={N_DATA}")
    z1_data = sample_crescent(N_DATA)

    # Batch OT — z0 et z1_ot utilisés PARTOUT (OT + k-NN)
    torch.manual_seed(42)
    z0    = torch.randn(N_OT, D)
    idx   = torch.randint(0, N_DATA, (N_OT,))
    z1_ot = z1_data[idx]

    print(f"\nPlan OT ({N_OT}x{N_OT})...")
    T_star_np, W2_sq = compute_OT(z0, z1_ot)
    T_star = torch.tensor(T_star_np, dtype=torch.float32)
    print(f"W2² = {W2_sq:.4f}")

    # 1. Corollaire 1
    rows_cor = verify_corollary1(z0, z1_ot, K_VALUES, T_star, W2_sq)

    # 2. Bornes
    rows_bnd = verify_bounds(z0, z1_ot, K_VALUES, T_star)

    # 3. Coverage
    print("\nLP coverage (peut prendre quelques secondes)...")
    rows_cov = verify_coverage(z0, z1_ot, K_VALUES, T_star)

    # Plots
    plot_all(z0, z1_ot, z1_data, T_star, K_VALUES, rows_cor, rows_bnd, rows_cov)
    print("\nDone.")


if __name__ == "__main__":
    run()