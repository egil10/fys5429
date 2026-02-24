"""
bs.py
-----
Analytical Black-Scholes pricing.

Provides closed-form European call/put prices, Greeks, implied vol,
and surface generation for PINN benchmarking.
"""

import numpy as np
from scipy.stats import norm


# --- Core ---

def d1(S, K, T, r, sig):
    return (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))


def d2(S, K, T, r, sig):
    return d1(S, K, T, r, sig) - sig * np.sqrt(T)


def call(S, K, T, r, sig):
    """European call price."""
    _d1 = d1(S, K, T, r, sig)
    _d2 = d2(S, K, T, r, sig)
    return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put(S, K, T, r, sig):
    """European put price."""
    _d1 = d1(S, K, T, r, sig)
    _d2 = d2(S, K, T, r, sig)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


# --- Greeks ---

def delta(S, K, T, r, sig, cp="call"):
    """Delta. cp = 'call' or 'put'."""
    _d1 = d1(S, K, T, r, sig)
    return norm.cdf(_d1) if cp == "call" else norm.cdf(_d1) - 1.0


def gamma(S, K, T, r, sig):
    """Gamma (same for call/put)."""
    return norm.pdf(d1(S, K, T, r, sig)) / (S * sig * np.sqrt(T))


def vega(S, K, T, r, sig):
    """Vega (same for call/put)."""
    return S * norm.pdf(d1(S, K, T, r, sig)) * np.sqrt(T)


def theta(S, K, T, r, sig, cp="call"):
    """Theta (per year)."""
    _d1 = d1(S, K, T, r, sig)
    _d2 = d2(S, K, T, r, sig)
    t1 = -S * norm.pdf(_d1) * sig / (2.0 * np.sqrt(T))
    if cp == "call":
        return t1 - r * K * np.exp(-r * T) * norm.cdf(_d2)
    return t1 + r * K * np.exp(-r * T) * norm.cdf(-_d2)


def rho(S, K, T, r, sig, cp="call"):
    """Rho."""
    _d2 = d2(S, K, T, r, sig)
    if cp == "call":
        return K * T * np.exp(-r * T) * norm.cdf(_d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-_d2)


# --- Implied vol ---

def iv(price, S, K, T, r, cp="call", tol=1e-8, maxiter=200):
    """Implied volatility via Newton-Raphson."""
    sig = 0.25
    pricer = call if cp == "call" else put
    for _ in range(maxiter):
        p = pricer(S, K, T, r, sig)
        v = vega(S, K, T, r, sig)
        if abs(v) < 1e-14:
            return np.nan
        sig -= (p - price) / v
        if sig <= 0:
            return np.nan
        if abs(p - price) < tol:
            return sig
    return np.nan


# --- Surface ---

def surface(S_min=50, S_max=150, T_min=0.01, T_max=2.0,
            nS=100, nT=100, K=100, r=0.05, sig=0.2, cp="call"):
    """Generate V(S,T) on a grid."""
    Sv = np.linspace(S_min, S_max, nS)
    Tv = np.linspace(T_min, T_max, nT)
    Sg, Tg = np.meshgrid(Sv, Tv, indexing="ij")
    pricer = call if cp == "call" else put
    V = pricer(Sg, K, Tg, r, sig)
    return {"S": Sg, "T": Tg, "V": V}


# --- Demo ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    plotdir = Path(__file__).resolve().parent.parent / "plots" / "analytical"
    plotdir.mkdir(parents=True, exist_ok=True)

    K, r, sig = 100.0, 0.05, 0.20
    S0, T0 = 100.0, 1.0

    c = call(S0, K, T0, r, sig)
    p = put(S0, K, T0, r, sig)
    print(f"Call: {c:.4f}   Put: {p:.4f}")
    print(f"Put-call parity: C-P={c-p:.4f}, S-Ke^(-rT)={S0 - K*np.exp(-r*T0):.4f}")

    dc = delta(S0, K, T0, r, sig)
    g  = gamma(S0, K, T0, r, sig)
    v  = vega(S0, K, T0, r, sig)
    print(f"\nDelta={dc:.4f}  Gamma={g:.4f}  Vega={v:.4f}")

    surf = surface()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
    ax.plot_surface(surf["S"], surf["T"], surf["V"],
                    cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("S"); ax.set_ylabel("T"); ax.set_zlabel("V(S,T)")
    ax.set_title("Black-Scholes Call Surface")
    plt.tight_layout()
    out = plotdir / "bs_surface.pdf"
    plt.savefig(out)
    print(f"\nSaved to {out}")
    plt.show()
