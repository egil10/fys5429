"""pipeline.py
-----------
End-to-end BS PINN pipeline.

  1. Load generated BS surface data
  2. Train PINN
  3. Evaluate: RMSE, MAE, rel-L2 across the surface
  4. Plot: surface comparison, error map, loss curves, Greeks
  5. Save all PDFs to code/plots/pinn/

  python pipeline.py [--steps N] [--seed S]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import style  # noqa: applies rcParams on import

from utils   import set_seed, plot_loss
from bs      import call as bs_call, surface as bs_surface
from pinn_bs import BSPINN
from metrics import summary
from greeks  import bs_greeks, pinn_delta, pinn_gamma, compare_greeks

OUT = Path(__file__).parent.parent / "plots" / "pinn"
OUT.mkdir(parents=True, exist_ok=True)


# -- helpers ------------------------------------------------------------------

def _savefig(name):
    path = OUT / name
    plt.savefig(path)
    print(f"  saved: {path.name}")


# -- 1. data ------------------------------------------------------------------

def load_data():
    data_dir = Path(__file__).parent.parent / "data" / "generated"
    df = pd.read_parquet(data_dir / "bs_surface.parquet")
    print(f"Loaded BS surface: {len(df):,} rows  cols={list(df.columns)}")
    return df


# -- 2. train -----------------------------------------------------------------

def train(K=100.0, r=0.05, sig=0.20, n_steps=5000, seed=42):
    set_seed(seed)
    model = BSPINN(K=K, r=r, sig=sig, lam=(1.0, 10.0, 5.0))
    print(f"\nTraining BS PINN  ({n_steps} steps, device={model.device})")
    model.train(n_steps=n_steps, log=n_steps // 10)
    model.save(OUT / "bs_pinn.pt")
    return model


# -- 3. evaluate --------------------------------------------------------------

def evaluate(model, df, K, r, sig):
    print("\nMetrics on full 100x100 surface")
    S   = df["S"].values
    tau = df["T"].values
    V_nn = model.predict(S, tau)
    V_ex = df["call"].values
    summary(V_nn, V_ex, label="BS PINN vs analytical")
    return S, tau, V_nn, V_ex


# -- 4. plots -----------------------------------------------------------------

def plot_slice(model, K, r, sig):
    """PINN vs exact for three maturities."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    S = np.linspace(60, 160, 200)
    for ax, tau in zip(axes, [0.25, 1.0, 2.0]):
        V_nn = model.predict(S, np.full_like(S, tau))
        V_ex = bs_call(S, K, tau, r, sig)
        ax.plot(S, V_ex, color=style.COLORS["exact"],  lw=2,   label="exact")
        ax.plot(S, V_nn, color=style.COLORS["pinn"],   lw=1.5, ls="--", label="PINN")
        ax.set(xlabel="S", ylabel="V", title=f"tau = {tau}")
        ax.legend()
    fig.suptitle("BS PINN vs Analytical")
    plt.tight_layout()
    _savefig("bs_slices.pdf")
    plt.close()


def plot_surface_comparison(model, K, r, sig):
    """Side-by-side 3D: exact | PINN | error."""
    surf = bs_surface(nS=50, nT=50, K=K, r=r, sig=sig)
    S, T = surf["S"], surf["T"]
    V_ex = surf["V"]
    V_nn = model.predict(S.ravel(), T.ravel()).reshape(S.shape)
    err  = np.abs(V_nn - V_ex)

    fig = plt.figure(figsize=(16, 5))
    titles = ["Exact (analytical)", "PINN", "|Error|"]
    data   = [V_ex, V_nn, err]
    cmaps  = [style.CMAPS["surface"], style.CMAPS["surface"], style.CMAPS["error"]]

    for i, (title, V, cmap) in enumerate(zip(titles, data, cmaps), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(S, T, V, cmap=cmap, edgecolor="none", alpha=0.9)
        ax.set(xlabel="S", ylabel="tau", zlabel="V", title=title)

    plt.suptitle("Black-Scholes PINN")
    plt.tight_layout()
    _savefig("bs_surface_comparison.pdf")
    plt.close()


def plot_error_map(model, K, r, sig):
    """2D heatmap of relative error across (S, tau)."""
    surf = bs_surface(nS=60, nT=60, K=K, r=r, sig=sig)
    S, T, V_ex = surf["S"], surf["T"], surf["V"]
    V_nn = model.predict(S.ravel(), T.ravel()).reshape(S.shape)
    rel_err = np.abs(V_nn - V_ex) / (np.abs(V_ex) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(T, S, rel_err * 100, cmap="Reds", shading="auto")
    plt.colorbar(im, ax=ax, label="relative error (%)")
    ax.set(xlabel="tau (time to maturity)", ylabel="S (spot)", title="BS PINN — Relative Error (%)")
    plt.tight_layout()
    _savefig("bs_error_map.pdf")
    plt.close()


def plot_training_loss(model):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals in model.history.items():
        ax.semilogy(vals, label=label, lw=1.4)
    ax.set(xlabel="step", ylabel="loss (log scale)", title="Training History")
    ax.legend()
    plt.tight_layout()
    _savefig("bs_loss.pdf")
    plt.close()


def plot_greeks_comparison(model, K, r, sig):
    """Exact vs PINN Greeks (delta, gamma) across S."""
    S   = np.linspace(60, 160, 200)
    tau = 1.0

    g_exact = {k: v for k, v in bs_greeks(S, K, tau, r, sig).items()
               if k in ("delta", "gamma")}
    g_pinn  = {
        "delta": pinn_delta(model, S, np.full_like(S, tau)),
        "gamma": pinn_gamma(model, S, np.full_like(S, tau)),
    }

    fig, axes = compare_greeks(S, g_exact, g_pinn)
    fig.suptitle("Greeks: BS exact vs PINN  (tau=1)")
    plt.tight_layout()
    _savefig("bs_greeks.pdf")
    plt.close()


def plot_all_greeks(K, r, sig):
    """All 5 BS Greeks on one panel."""
    from greeks import plot_greeks
    S = np.linspace(60, 160, 300)
    g = bs_greeks(S, K, 1.0, r, sig)
    fig, _ = plot_greeks(S, g, title="Black-Scholes Greeks  (K=100, T=1, sig=0.20)")
    _savefig("bs_greeks_all.pdf")
    plt.close()


# -- main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int,   default=5000)
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--K",     type=float, default=100.0)
    p.add_argument("--r",     type=float, default=0.05)
    p.add_argument("--sig",   type=float, default=0.20)
    args = p.parse_args()

    K, r, sig = args.K, args.r, args.sig

    df    = load_data()
    model = train(K=K, r=r, sig=sig, n_steps=args.steps, seed=args.seed)

    evaluate(model, df, K, r, sig)

    print("\nSaving plots...")
    plot_slice(model, K, r, sig)
    plot_surface_comparison(model, K, r, sig)
    plot_error_map(model, K, r, sig)
    plot_training_loss(model)
    plot_greeks_comparison(model, K, r, sig)
    plot_all_greeks(K, r, sig)

    print(f"\nAll plots saved to {OUT}/")


if __name__ == "__main__":
    main()
