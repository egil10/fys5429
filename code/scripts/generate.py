"""
generate.py
-----------
Generate synthetic option pricing data for PINN training / validation.

Outputs parquet files to code/data/generated/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import bs
import heston as hs


OUTDIR = Path(__file__).resolve().parent.parent / "data" / "generated"


def save(df, name, n_sample=100):
    """Save df as parquet + a small CSV sample for quick inspection."""
    df.to_parquet(OUTDIR / f"{name}.parquet", index=False)
    df.sample(min(n_sample, len(df)), random_state=0).to_csv(
        OUTDIR / f"{name}_sample.csv", index=False, float_format="%.6f"
    )
    print(f"  {name}: {len(df)} rows → .parquet + {min(n_sample, len(df))}-row .csv")


# --- Black-Scholes data ---

def gen_bs(S_min=50, S_max=150, T_min=0.01, T_max=2.0,
           nS=100, nT=100, K=100, r=0.05, sig=0.2):
    """Generate BS call+put prices on a (S, T) grid → parquet."""
    surf_c = bs.surface(S_min, S_max, T_min, T_max, nS, nT, K, r, sig, cp="call")
    surf_p = bs.surface(S_min, S_max, T_min, T_max, nS, nT, K, r, sig, cp="put")

    df = pd.DataFrame({
        "S":    surf_c["S"].ravel(),
        "T":    surf_c["T"].ravel(),
        "K":    K,
        "r":    r,
        "sig":  sig,
        "call": surf_c["V"].ravel(),
        "put":  surf_p["V"].ravel(),
    })

    save(df, "bs_surface")
    return df


# --- Heston data ---

def gen_heston(S_min=50, S_max=150, T_min=0.05, T_max=2.0,
               nS=40, nT=40, K=100, r=0.05,
               v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7):
    """Generate Heston call+put prices on a (S, T) grid → parquet."""
    surf_c = hs.surface(S_min, S_max, T_min, T_max, nS, nT,
                        K, r, v0, kappa, theta, xi, rho, cp="call")
    surf_p = hs.surface(S_min, S_max, T_min, T_max, nS, nT,
                        K, r, v0, kappa, theta, xi, rho, cp="put")

    df = pd.DataFrame({
        "S":     surf_c["S"].ravel(),
        "T":     surf_c["T"].ravel(),
        "K":     K,
        "r":     r,
        "v0":    v0,
        "kappa": kappa,
        "theta": theta,
        "xi":    xi,
        "rho":   rho,
        "call":  surf_c["V"].ravel(),
        "put":   surf_p["V"].ravel(),
    })

    save(df, "heston_surface")
    return df


# --- Collocation points for PINN interior ---

def gen_collocation(n=10000, S_min=50, S_max=150, T_min=0.01, T_max=2.0):
    """Random (S, T) collocation points for PDE residual training."""
    rng = np.random.default_rng(42)
    S = rng.uniform(S_min, S_max, n)
    T = rng.uniform(T_min, T_max, n)

    df = pd.DataFrame({"S": S, "T": T})
    save(df, "collocation")
    return df


# --- Main ---

if __name__ == "__main__":
    from time import time

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Generating BS surface...")
    t0 = time()
    gen_bs()
    print(f"  ({time()-t0:.1f}s)\n")

    print("Generating Heston surface (this takes a few minutes)...")
    t0 = time()
    gen_heston()
    print(f"  ({time()-t0:.1f}s)\n")

    print("Generating collocation points...")
    gen_collocation()

    print("\nDone. Files in:", OUTDIR)
