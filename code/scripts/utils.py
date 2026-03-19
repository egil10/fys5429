"""utils.py
---------
Shared utilities: seeding, plotting, model I/O.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Seed numpy, random, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_surface(S, T, V, title="", path=None, cmap="viridis"):
    """3D surface. S, T, V are 2D arrays."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9, 6))
    ax.plot_surface(S, T, V, cmap=cmap, edgecolor="none", alpha=0.9)
    ax.set(xlabel="S", ylabel="T", zlabel="V", title=title)
    plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    return fig, ax


def plot_error(S, T, V_pred, V_exact, title="Absolute Error", path=None):
    """Absolute error surface |V_pred − V_exact|."""
    return plot_surface(S, T, np.abs(V_pred - V_exact),
                        title=title, path=path, cmap="Reds")


def plot_loss(history: dict, path=None):
    """Training loss curves from a history dict {label: [values]}."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals in history.items():
        ax.semilogy(vals, label=label)
    ax.set(xlabel="step", ylabel="loss", title="Training History")
    ax.legend()
    plt.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    return fig, ax


# ── Model I/O ─────────────────────────────────────────────────────────────────

def save_model(model, path):
    model.save(path)


def load_model(model_cls, path, **kwargs):
    m = model_cls(**kwargs)
    m.load(path)
    return m
