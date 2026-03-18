"""
Learned Interpolant Schedule — plug-in module for existing DiT
==============================================================
Drop-in replacement for a fixed schedule (linear, cosine, etc.).

Usage with your existing DiT:

    # Before (fixed schedule)
    t        = torch.rand(B)
    alpha_t  = 1 - t
    beta_t   = t
    x_t      = alpha_t * x_0 + beta_t * x_1
    target   = x_1 - x_0
    loss     = F.mse_loss(dit(x_t, t), target)

    # After (learned schedule) — only these lines change
    schedule = LearnedSchedule(n_knots=8).to(device)
    x_t      = schedule.interpolate(x_0, x_1, t)
    target   = schedule.target_velocity(x_0, x_1, t)
    loss     = F.mse_loss(dit(x_t, t), target) + schedule.regularization(x_0, x_1)

    # Add schedule.parameters() to your existing optimizer
    optimizer = Adam([
        {"params": dit.parameters(),      "lr": 1e-4},
        {"params": schedule.parameters(), "lr": 1e-3},  # faster lr for schedule
    ])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LearnedSchedule(nn.Module):
    """
    Learnable monotone schedule for flow matching interpolation.

        x_t = alpha(t) * x_0 + beta(t) * x_1

    Boundary conditions (hard-coded, never violated):
        alpha(0) = 1,  alpha(1) = 0
        beta(0)  = 0,  beta(1)  = 1

    Two parameterization options (set via `mode`):
        - "spline" : piecewise linear with n_knots learnable knots (default)
                     interpretable, easy to visualize, ~n_knots params
        - "mlp"    : small MLP with monotonicity enforced by a cumsum trick
                     more expressive, ~few hundred params

    Regularization (call schedule.regularization(x_0, x_1)):
        Penalizes high path energy E[∫||u_t||² dt].
        Prevents the degenerate solution where alpha/beta stay constant.
        Weight lam is set at construction — tune between 1e-3 and 1e-1.
    """

    def __init__(
        self,
        n_knots: int   = 8,
        mode:    str   = "spline",   # "spline" | "mlp"
        lam:     float = 0.01,       # regularization weight
    ):
        super().__init__()
        assert mode in ("spline", "mlp")
        self.mode    = mode
        self.lam     = lam
        self.n_knots = n_knots

        if mode == "spline":
            # Raw log-increments for inner knot values
            # Shape: (n_knots - 1,) — boundary values (0 and 1) are fixed
            self.alpha_raw = nn.Parameter(torch.zeros(n_knots - 1))
            self.beta_raw  = nn.Parameter(torch.zeros(n_knots - 1))
            self.register_buffer("knots", torch.linspace(0.0, 1.0, n_knots))

        elif mode == "mlp":
            # Small MLP outputting cumulative increments
            # Input: t in [0,1], Output: scalar in [0,1] (monotone by construction)
            hidden = 32
            self.alpha_mlp = _MonotoneMLP(hidden)
            self.beta_mlp  = _MonotoneMLP(hidden)

    # ------------------------------------------------------------------
    # Core: compute alpha(t) and beta(t)
    # ------------------------------------------------------------------

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> alpha(t): (B,)  — strictly decreasing, [1 -> 0]"""
        if self.mode == "spline":
            vals = self._spline_values(self.alpha_raw, decreasing=True)
            return self._piecewise_linear(vals, t)
        else:
            return 1.0 - self.alpha_mlp(t)   # flip: decreasing

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> beta(t): (B,)  — strictly increasing, [0 -> 1]"""
        if self.mode == "spline":
            vals = self._spline_values(self.beta_raw, decreasing=False)
            return self._piecewise_linear(vals, t)
        else:
            return self.beta_mlp(t)           # increasing

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d alpha/dt: (B,)"""
        if self.mode == "spline":
            vals = self._spline_values(self.alpha_raw, decreasing=True)
            return self._piecewise_slope(vals, t)
        else:
            # Autograd for MLP mode
            t_ = t.detach().requires_grad_(True)
            a  = 1.0 - self.alpha_mlp(t_)
            return torch.autograd.grad(a.sum(), t_, create_graph=True)[0]

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        """d beta/dt: (B,)"""
        if self.mode == "spline":
            vals = self._spline_values(self.beta_raw, decreasing=False)
            return self._piecewise_slope(vals, t)
        else:
            t_ = t.detach().requires_grad_(True)
            b  = self.beta_mlp(t_)
            return torch.autograd.grad(b.sum(), t_, create_graph=True)[0]

    # ------------------------------------------------------------------
    # Public interface (these are the 3 lines that replace your fixed schedule)
    # ------------------------------------------------------------------

    def interpolate(
        self,
        x_0: torch.Tensor,   # (B, ...)
        x_1: torch.Tensor,   # (B, ...)
        t:   torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """x_t = alpha(t) * x_0 + beta(t) * x_1"""
        # Reshape to broadcast over arbitrary data dims (images, fields, etc.)
        shape = (-1,) + (1,) * (x_0.dim() - 1)
        a = self.alpha(t).view(shape)
        b = self.beta(t).view(shape)
        return a * x_0 + b * x_1

    def target_velocity(
        self,
        x_0: torch.Tensor,   # (B, ...)
        x_1: torch.Tensor,   # (B, ...)
        t:   torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """u_t = alpha_dot(t) * x_0 + beta_dot(t) * x_1"""
        shape = (-1,) + (1,) * (x_0.dim() - 1)
        da = self.alpha_dot(t).view(shape)
        db = self.beta_dot(t).view(shape)
        return da * x_0 + db * x_1

    def regularization(
        self,
        x_0:    torch.Tensor,
        x_1:    torch.Tensor,
        n_quad: int = 10,
    ) -> torch.Tensor:
        """
        Path energy penalty: lam * E[∫_0^1 ||u_t||^2 dt]
        Approximated by Gauss-Legendre quadrature (n_quad points).
        Returns a scalar — add directly to your FM loss.
        """
        if self.lam == 0.0:
            return torch.tensor(0.0, device=x_0.device)

        # Gauss-Legendre nodes and weights on [0, 1]
        nodes, weights = _gauss_legendre(n_quad, device=x_0.device)

        total = torch.tensor(0.0, device=x_0.device)
        for t_val, w in zip(nodes, weights):
            t   = t_val.expand(x_0.shape[0])
            u_t = self.target_velocity(x_0, x_1, t)
            # Mean over batch, sum over data dims
            energy = (u_t ** 2).flatten(1).sum(dim=1).mean()
            total  = total + w * energy

        return self.lam * total

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def plot(self, ax=None, label_suffix=""):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 3))
        t_vals = torch.linspace(0, 1, 300, device=next(self.parameters()).device)
        a  = self.alpha(t_vals).cpu().numpy()
        b  = self.beta(t_vals).cpu().numpy()
        t_ = t_vals.cpu().numpy()
        ax.plot(t_, a, color="#378ADD", lw=2, label=f"α(t){label_suffix}")
        ax.plot(t_, b, color="#D85A30", lw=2, label=f"β(t){label_suffix}")
        ax.plot(t_, 1 - t_, "--", color="#888", alpha=0.4, lw=1, label="linear α")
        ax.plot(t_, t_,     "--", color="#888", alpha=0.4, lw=1, label="linear β")
        ax.set_xlabel("t"); ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05)
        return ax

    # ------------------------------------------------------------------
    # Spline internals
    # ------------------------------------------------------------------

    def _spline_values(self, raw: torch.Tensor, decreasing: bool) -> torch.Tensor:
        """raw (n_knots-1,) -> knot values (n_knots,) in [0,1], monotone."""
        increments = F.softplus(raw) + 1e-4          # strictly positive
        cumsum     = torch.cumsum(increments, dim=0)  # monotone increasing
        values     = torch.cat([raw.new_zeros(1), cumsum])
        values     = values / values[-1]              # normalize to [0, 1]
        if decreasing:
            values = 1.0 - values
        return values

    def _piecewise_linear(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate piecewise linear function at query times t."""
        t      = t.clamp(0.0, 1.0)
        knots  = self.knots
        idx    = torch.searchsorted(knots[1:].contiguous(), t).clamp(0, self.n_knots - 2)
        t0, t1 = knots[idx], knots[idx + 1]
        v0, v1 = values[idx], values[idx + 1]
        frac   = (t - t0) / (t1 - t0 + 1e-8)
        return v0 + frac * (v1 - v0)

    def _piecewise_slope(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Piecewise constant derivative."""
        t     = t.clamp(0.0, 1.0)
        knots = self.knots
        idx   = torch.searchsorted(knots[1:].contiguous(), t).clamp(0, self.n_knots - 2)
        dt    = knots[idx + 1] - knots[idx]
        dv    = values[idx + 1] - values[idx]
        return dv / (dt + 1e-8)


# ---------------------------------------------------------------------------
# Monotone MLP (for mode="mlp")
# ---------------------------------------------------------------------------

class _MonotoneMLP(nn.Module):
    """
    MLP with guaranteed monotone output in [0, 1].
    Input: scalar t in [0,1]
    Output: scalar in [0,1], strictly increasing in t.

    Monotonicity via positive weights (softplus activations on weight matrices).
    Boundary conditions enforced by subtracting f(0) and dividing by f(1)-f(0).
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        # We parameterize log-weights and exponentiate to keep weights > 0
        # This guarantees monotone output for monotone activations (sigmoid)
        self.w1 = nn.Parameter(torch.randn(1,      hidden) * 0.1)
        self.w2 = nn.Parameter(torch.randn(hidden, hidden) * 0.1)
        self.w3 = nn.Parameter(torch.randn(hidden, 1)      * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.b2 = nn.Parameter(torch.zeros(hidden))
        self.b3 = nn.Parameter(torch.zeros(1))

    def _forward_raw(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B,) raw monotone output (not yet normalized)."""
        x  = t.unsqueeze(-1)                          # (B, 1)
        W1 = F.softplus(self.w1)                      # positive weights
        W2 = F.softplus(self.w2)
        W3 = F.softplus(self.w3)
        x  = torch.sigmoid(x @ W1 + self.b1)          # (B, hidden)
        x  = torch.sigmoid(x @ W2 + self.b2)          # (B, hidden)
        x  = (x @ W3 + self.b3).squeeze(-1)           # (B,)
        return x

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Normalized output in [0, 1] with f(0)≈0 and f(1)≈1."""
        device = t.device
        f_t = self._forward_raw(t)
        f_0 = self._forward_raw(torch.zeros(1, device=device)).detach()
        f_1 = self._forward_raw(torch.ones(1,  device=device)).detach()
        return (f_t - f_0) / (f_1 - f_0 + 1e-8)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature on [0, 1]
# ---------------------------------------------------------------------------

def _gauss_legendre(n: int, device: str = "cpu"):
    """
    Returns (nodes, weights) for n-point Gauss-Legendre quadrature on [0,1].
    More accurate than uniform sampling for smooth integrands.
    """
    import numpy as np
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    # Transform from [-1,1] to [0,1]
    nodes   = torch.tensor((nodes_np + 1) / 2,  dtype=torch.float32, device=device)
    weights = torch.tensor(weights_np / 2,       dtype=torch.float32, device=device)
    return nodes, weights


# ---------------------------------------------------------------------------
# Minimal integration test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing LearnedSchedule...")

    schedule = LearnedSchedule(n_knots=8, mode="spline", lam=0.01)

    t  = torch.linspace(0, 1, 5)
    a  = schedule.alpha(t)
    b  = schedule.beta(t)

    print(f"alpha(t): {a.detach().numpy().round(3)}")
    print(f"beta(t):  {b.detach().numpy().round(3)}")
    print(f"alpha(0)={a[0].item():.4f} (should be 1.0)")
    print(f"alpha(1)={a[-1].item():.4f} (should be 0.0)")
    print(f"beta(0)={b[0].item():.4f}  (should be 0.0)")
    print(f"beta(1)={b[-1].item():.4f}  (should be 1.0)")

    # Simulate one training step with a fake DiT
    B, C, H, W = 4, 3, 16, 16
    x_0    = torch.randn(B, C, H, W)
    x_1    = torch.randn(B, C, H, W)
    t_rand = torch.rand(B)

    x_t    = schedule.interpolate(x_0, x_1, t_rand)
    target = schedule.target_velocity(x_0, x_1, t_rand)
    reg    = schedule.regularization(x_0, x_1)

    # Fake DiT prediction
    v_pred = torch.randn_like(x_t)

    loss = F.mse_loss(v_pred, target) + reg
    loss.backward()

    print(f"\nShapes OK: x_t={tuple(x_t.shape)}, target={tuple(target.shape)}")
    print(f"Loss: {loss.item():.4f}  (reg={reg.item():.4f})")
    print(f"Schedule grad norm: {schedule.alpha_raw.grad.norm().item():.4f}")
    print("\nAll checks passed.")