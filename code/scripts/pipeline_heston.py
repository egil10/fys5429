"""pipeline_heston.py
---------------------
End-to-end Heston PINN pipeline.

  1. Load generated Heston surface data
  2. Train HestonPINN
  3. Evaluate: metrics vs analytical
  4. Plot: surface comparison, error map, slices, loss
  5. Save all PDFs to code/plots/pinn/

  python pipeline_heston.py [--steps N] [--seed S]
"""

import argparse
from pathlib import Path

import torch  # must load before numpy on Windows (DLL order)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fys5429.style as style  # noqa

from fys5429.utils       import set_seed
from fys5429.heston      import call_cos, surface as heston_surface
from fys5429.pinn_heston import HestonPINN
from fys5429.metrics     import summary

OUT = Path(__file__).parent.parent / "plots" / "pinn"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical Heston parameters (match generate.py)
K, r = 100.0, 0.05
v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7


def _savefig(name):
    path = OUT / name
    plt.savefig(path)
    print(f"  saved: {path.name}")


def load_data():
    data_dir = Path(__file__).parent.parent / "data" / "generated"
    df = pd.read_parquet(data_dir / "heston_surface.parquet")
    print(f"Loaded Heston surface: {len(df):,} rows  cols={list(df.columns)}")
    return df


def train(n_steps=5000, seed=42):
    set_seed(seed)
    model = HestonPINN(K=K, r=r, kappa=kappa, theta=theta, xi=xi, rho=rho,
                       lam=(1.0, 10.0, 5.0))
    print(f"\nTraining Heston PINN  ({n_steps} steps, device={model.device})")
    model.train(n_steps=n_steps, n_col=2000, n_ic=500, n_bc=200,
                S_min=50.0, S_max=150.0, v_min=1e-4, v_max=0.5, tau_max=2.0,
                log=n_steps // 10)
    model.save(OUT / "heston_pinn.pt")
    return model


def evaluate(model, df):
    print("\nMetrics on full Heston surface")
    V_nn = model.predict(df["S"].values, df["v0"].values, df["T"].values)
    V_ex = df["call"].values
    summary(V_nn, V_ex, label="Heston PINN vs analytical")


def plot_slices(model):
    """PINN vs exact at fixed v=v0 for three maturities."""
    S = np.linspace(60, 140, 150)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, tau in zip(axes, [0.5, 1.0, 2.0]):
        V_ex = np.array([call_cos(s, K, tau, r, v0, kappa, theta, xi, rho) for s in S])
        V_nn = model.predict(S, np.full_like(S, v0), np.full_like(S, tau))
        ax.plot(S, V_ex, color=style.COLORS["exact"], lw=2,   label="exact")
        ax.plot(S, V_nn, color=style.COLORS["pinn"],  lw=1.5, ls="--", label="PINN")
        ax.set(xlabel="S", ylabel="V", title=f"tau = {tau}")
        ax.legend()
    fig.suptitle("Heston PINN vs Analytical  (v = v0 = 0.04)")
    plt.tight_layout()
    _savefig("heston_slices.pdf")
    plt.close()


def plot_surface_comparison(model):
    """3D side-by-side: exact | PINN | |error| at fixed v=v0."""
    surf = heston_surface(S_min=50, S_max=150, T_min=0.25, T_max=2.0,
                          nS=30, nT=30, K=K, r=r,
                          v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    S, T, V_ex = surf["S"], surf["T"], surf["V"]
    V_nn = model.predict(S.ravel(), np.full(S.size, v0), T.ravel()).reshape(S.shape)
    err  = np.abs(V_nn - V_ex)

    fig = plt.figure(figsize=(16, 5))
    for i, (title, V, cmap) in enumerate(
            zip(["Exact", "PINN", "|Error|"],
                [V_ex, V_nn, err],
                [style.CMAPS["surface"], style.CMAPS["surface"], style.CMAPS["error"]]), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(S, T, V, cmap=cmap, edgecolor="none", alpha=0.9)
        ax.set(xlabel="S", ylabel="tau", zlabel="V", title=title)
    plt.suptitle("Heston PINN  (v = v0 = 0.04)")
    plt.tight_layout()
    _savefig("heston_surface_comparison.pdf")
    plt.close()


def plot_error_map(model):
    """2D relative error heatmap at fixed v=v0."""
    surf = heston_surface(S_min=50, S_max=150, T_min=0.25, T_max=2.0,
                          nS=40, nT=40, K=K, r=r,
                          v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    S, T, V_ex = surf["S"], surf["T"], surf["V"]
    V_nn = model.predict(S.ravel(), np.full(S.size, v0), T.ravel()).reshape(S.shape)
    rel_err = np.abs(V_nn - V_ex) / (np.abs(V_ex) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(T, S, rel_err * 100, cmap="Reds", shading="auto")
    plt.colorbar(im, ax=ax, label="relative error (%)")
    ax.set(xlabel="tau", ylabel="S",
           title="Heston PINN — Relative Error (%)  (v = v0)")
    plt.tight_layout()
    _savefig("heston_error_map.pdf")
    plt.close()


def plot_loss(model):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals in model.history.items():
        ax.semilogy(vals, label=label, lw=1.4)
    ax.set(xlabel="step", ylabel="loss (log scale)", title="Heston PINN Training")
    ax.legend()
    plt.tight_layout()
    _savefig("heston_loss.pdf")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()

    df    = load_data()
    model = train(n_steps=args.steps, seed=args.seed)
    evaluate(model, df)

    print("\nSaving plots...")
    plot_slices(model)
    plot_surface_comparison(model)
    plot_error_map(model)
    plot_loss(model)
    print(f"\nAll plots saved to {OUT}/")


if __name__ == "__main__":
    main()
