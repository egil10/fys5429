"""pipeline_calibrate.py
------------------------
Inverse problem: recover BS implied vol and Heston parameters from prices.

  1. BS: implied vol surface from bs_surface.parquet
  2. Heston: calibrate (v0, kappa, theta, xi, rho) from a noisy surface subset
  3. Plot results to code/plots/calibrate/

  python pipeline_calibrate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import fys5429.style as style  # noqa

from fys5429.calibrate import calibrate_bs
from fys5429.heston    import call as heston_call

OUT = Path(__file__).parent.parent / "plots" / "calibrate"
OUT.mkdir(parents=True, exist_ok=True)

# True Heston parameters (match generate.py)
TRUE = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


def _savefig(name):
    path = OUT / name
    plt.savefig(path)
    print(f"  saved: {path.name}")


# ── BS implied vol ────────────────────────────────────────────────────────────

def bs_iv_pipeline(data_dir):
    df = pd.read_parquet(data_dir / "bs_surface.parquet")
    print(f"Loaded BS surface: {len(df):,} rows")

    df = df.sort_values(["S", "T"]).reset_index(drop=True)
    r_val    = float(df["r"].iloc[0])
    true_sig = float(df["sig"].iloc[0])

    print("\nBS implied vol calibration")
    iv = calibrate_bs(df["call"].values, df["S"].values,
                      df["K"].values, df["T"].values, r_val)

    liquid = np.isfinite(iv)
    print(f"  liquid options: {liquid.sum():,} / {len(iv):,}")
    print(f"  true sigma  = {true_sig:.4f}")
    print(f"  recovered   = {iv[liquid].mean():.4f} +/- {iv[liquid].std():.6f}")
    print(f"  max abs err = {np.abs(iv[liquid] - true_sig).max():.2e}")

    # heatmap (NaN shown as white)
    nS = int(df["S"].nunique())
    nT = int(df["T"].nunique())
    S  = df["S"].values.reshape(nS, nT)
    T  = df["T"].values.reshape(nS, nT)
    IV = iv.reshape(nS, nT)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(T, S, IV, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="implied vol")
    ax.set(xlabel="T (maturity)", ylabel="S (spot)",
           title=f"BS Implied Vol Surface  (true sigma = {true_sig:.2f})")
    plt.tight_layout()
    _savefig("calibrate_iv_heatmap.pdf")
    plt.close()


# ── Heston calibration ────────────────────────────────────────────────────────

def _obj(params, prices, S, K, T, r):
    v0, kappa, theta, xi, rho = params
    try:
        phat = np.array([heston_call(s, k, t, r, v0, kappa, theta, xi, rho)
                         for s, k, t in zip(S, K, T)])
    except Exception:
        return 1e10
    if not np.all(np.isfinite(phat)):
        return 1e10
    return np.mean((phat - prices) ** 2)


def heston_calibration_pipeline(data_dir):
    df = pd.read_parquet(data_dir / "heston_surface.parquet")
    print(f"\nLoaded Heston surface: {len(df):,} rows")

    sub    = df.sample(30, random_state=42).reset_index(drop=True)
    rng    = np.random.default_rng(42)
    prices = sub["call"].values * (1 + 0.01 * rng.standard_normal(len(sub)))
    r_val  = float(sub["r"].iloc[0])

    p0     = (0.06, 1.5, 0.05, 0.40, -0.50)   # perturbed from true
    bounds = ((1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (0.01, 1.0), (-0.99, 0.0))

    print("\nHeston parameter calibration")
    print(f"  points: {len(sub)}  noise: 1%  initial guess: {p0}")

    res = minimize(_obj, p0,
                   args=(prices, sub["S"].values, sub["K"].values, sub["T"].values, r_val),
                   bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    rec = dict(zip(("v0", "kappa", "theta", "xi", "rho"), res.x))
    rec["mse"]     = res.fun
    rec["success"] = res.success

    print(f"  converged: {rec['success']}   MSE: {rec['mse']:.4e}")
    print(f"\n  {'param':>8}  {'true':>10}  {'recovered':>10}  {'err%':>8}")
    for k in ("v0", "kappa", "theta", "xi", "rho"):
        t, rv = TRUE[k], rec[k]
        print(f"  {k:>8}  {t:>10.4f}  {rv:>10.4f}  {abs(rv - t) / abs(t) * 100:>7.2f}%")

    # bar chart
    keys   = list(TRUE.keys())
    true_v = [TRUE[k] for k in keys]
    rec_v  = [rec[k]  for k in keys]
    x, w   = np.arange(len(keys)), 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, true_v, w, label="true",      color=style.COLORS["exact"], alpha=0.85)
    ax.bar(x + w / 2, rec_v,  w, label="recovered", color=style.COLORS["pinn"],  alpha=0.85)
    ax.set(xticks=x, xticklabels=keys, ylabel="value",
           title="Heston Calibration: True vs Recovered Parameters")
    ax.legend()
    plt.tight_layout()
    _savefig("calibrate_heston_params.pdf")
    plt.close()
