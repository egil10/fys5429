"""greeks.py
---------
Greeks computation and visualisation for BS and (numerical) Heston.

  Analytical BS Greeks via bs.py.
  Numerical Greeks via finite differences (for PINN or Heston).

  Usage:
    from fys5429.greeks import bs_greeks, num_delta, plot_greeks
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Analytical BS Greeks ──────────────────────────────────────────────────────

def bs_greeks(S, K, T, r, sig):
    """All five BS Greeks at once.

    Returns:
        dict with keys delta, gamma, vega, theta, rho
    """
    from fys5429.bs import delta, gamma, vega, theta, rho
    return dict(
        delta = delta(S, K, T, r, sig, cp="call"),
        gamma = gamma(S, K, T, r, sig),
        vega  = vega(S, K, T, r, sig),
        theta = theta(S, K, T, r, sig, cp="call"),
        rho   = rho(S, K, T, r, sig,   cp="call"),
    )


# ── Numerical Greeks (finite differences) ────────────────────────────────────

def num_delta(pricer, S, eps=0.5, **kwargs):
    """∂V/∂S via central difference."""
    return (pricer(S + eps, **kwargs) - pricer(S - eps, **kwargs)) / (2 * eps)


def num_gamma(pricer, S, eps=0.5, **kwargs):
    """∂²V/∂S² via central difference."""
    V     = pricer(S, **kwargs)
    V_up  = pricer(S + eps, **kwargs)
    V_dn  = pricer(S - eps, **kwargs)
    return (V_up - 2 * V + V_dn) / eps**2


def num_vega(pricer, sig, eps=0.001, **kwargs):
    """∂V/∂σ via central difference."""
    return (pricer(sig=sig + eps, **kwargs) - pricer(sig=sig - eps, **kwargs)) / (2 * eps)


def num_theta(pricer, T, eps=1/365, **kwargs):
    """−∂V/∂T via forward difference (per day)."""
    return -(pricer(T=T - eps, **kwargs) - pricer(T=T, **kwargs)) / eps


# ── PINN Greeks via autograd ──────────────────────────────────────────────────

def pinn_delta(model, S, tau):
    """∂V/∂S from a trained BSPINN via finite difference."""
    eps = 0.5
    return (model.predict(S + eps, tau) - model.predict(S - eps, tau)) / (2 * eps)


def pinn_gamma(model, S, tau):
    """∂²V/∂S² from a trained BSPINN via finite difference."""
    eps = 0.5
    V    = model.predict(S,       tau)
    V_up = model.predict(S + eps, tau)
    V_dn = model.predict(S - eps, tau)
    return (V_up - 2 * V + V_dn) / eps**2


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_greeks(S, greeks: dict, title="BS Greeks", path=None):
    """Panel plot of all provided Greeks vs S.

    Args:
        S: spot array
        greeks: dict {name: array}
        title: figure title
        path: save path (optional)
    """
    n = len(greeks)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, greeks.items()):
        ax.plot(S, vals)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set(xlabel="S", title=name)
    fig.suptitle(title)
    plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    return fig, axes


def compare_greeks(S, g_exact: dict, g_pinn: dict, path=None):
    """Overlay exact vs PINN Greeks and show error.

    Args:
        S: spot array
        g_exact, g_pinn: dicts {name: array} with the same keys
    """
    names = list(g_exact.keys())
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 6))
    for j, name in enumerate(names):
        axes[0, j].plot(S, g_exact[name], "k-",  label="exact")
        axes[0, j].plot(S, g_pinn[name],  "r--", label="PINN")
        axes[0, j].set(title=name)
        axes[0, j].legend(fontsize=8)

        axes[1, j].plot(S, g_pinn[name] - g_exact[name])
        axes[1, j].axhline(0, color="k", lw=0.5)
        axes[1, j].set(title=f"{name} error")

    plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    return fig, axes


# ── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    K, r, sig, T = 100.0, 0.05, 0.20, 1.0
    S = np.linspace(60, 160, 200)
    g = bs_greeks(S, K, T, r, sig)

    fig, axes = plot_greeks(S, g, title="Black-Scholes Greeks  (K=100, T=1, σ=0.20)")
    out = Path(__file__).parent.parent / "plots" / "analytical" / "bs_greeks.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"Saved → {out}")
    plt.show()
