"""
heston.py
---------
Semi-analytical Heston stochastic-volatility pricing.

Provides European call/put via characteristic-function integration,
COS method for fast pricing, and surface generation for PINN benchmarking.
"""

import numpy as np
from scipy.integrate import quad


# --- Characteristic function ---

def cf(u, S, T, r, v0, kappa, theta, xi, rho):
    """
    Heston log-price characteristic function (stable formulation).
    """
    i = 1j
    d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
    g = (kappa - rho * xi * i * u + d) / (kappa - rho * xi * i * u - d)

    C = r * i * u * T + (kappa * theta / xi**2) * (
        (kappa - rho * xi * i * u + d) * T
        - 2.0 * np.log((1.0 - g * np.exp(d * T)) / (1.0 - g))
    )
    D = ((kappa - rho * xi * i * u + d) / xi**2) * (
        (1.0 - np.exp(d * T)) / (1.0 - g * np.exp(d * T))
    )
    return np.exp(C + D * v0 + i * u * np.log(S))


# --- Integrands ---

def _integrand(u, S, K, T, r, v0, kappa, theta, xi, rho, j):
    """Integrand for P_j (j=1 or 2)."""
    i = 1j
    if j == 1:
        phi = cf(u - i, S, T, r, v0, kappa, theta, xi, rho) / (S * np.exp(r * T))
    else:
        phi = cf(u, S, T, r, v0, kappa, theta, xi, rho)
    return np.real(np.exp(-i * u * np.log(K)) * phi / (i * u))


# --- Pricing ---

def call(S, K, T, r, v0, kappa, theta, xi, rho, ulim=200.0):
    """European call via numerical integration."""
    args = (S, K, T, r, v0, kappa, theta, xi, rho)
    I1, _ = quad(_integrand, 1e-8, ulim, args=(*args, 1), limit=500)
    I2, _ = quad(_integrand, 1e-8, ulim, args=(*args, 2), limit=500)
    P1 = 0.5 + I1 / np.pi
    P2 = 0.5 + I2 / np.pi
    return S * P1 - K * np.exp(-r * T) * P2


def put(S, K, T, r, v0, kappa, theta, xi, rho, ulim=200.0):
    """European put via put-call parity."""
    return call(S, K, T, r, v0, kappa, theta, xi, rho, ulim) - S + K * np.exp(-r * T)


# --- COS method ---

def call_cos(S, K, T, r, v0, kappa, theta, xi, rho, N=256):
    """European call via COS method (Fang & Oosterlee 2008)."""
    c1 = np.log(S) + (r - 0.5 * theta) * T
    c2 = v0 * T + theta * T
    L  = 12.0
    a  = c1 - L * np.sqrt(abs(c2))
    b  = c1 + L * np.sqrt(abs(c2))
    bma = b - a

    def chi(k, c, d):
        if k == 0:
            return np.exp(d) - np.exp(c)
        w = k * np.pi / bma
        num = (np.exp(d) * (w * np.sin(w * (d - a)) + np.cos(w * (d - a)))
             - np.exp(c) * (w * np.sin(w * (c - a)) + np.cos(w * (c - a))))
        return num / (1.0 + w**2)

    def psi(k, c, d):
        if k == 0:
            return d - c
        w = k * np.pi / bma
        return (np.sin(w * (d - a)) - np.sin(w * (c - a))) / w

    lnK = np.log(K)
    price = 0.0
    for k in range(N):
        uk = k * np.pi / bma
        phi = cf(uk, S, T, r, v0, kappa, theta, xi, rho) * np.exp(-1j * uk * a)
        Vk = 2.0 / bma * (chi(k, lnK, b) - K * psi(k, lnK, b))
        w = 0.5 if k == 0 else 1.0
        price += w * np.real(phi) * Vk

    return np.exp(-r * T) * price


# --- Surface ---

def surface(S_min=50, S_max=150, T_min=0.05, T_max=2.0,
            nS=40, nT=40, K=100, r=0.05,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            cp="call"):
    """Generate V(S,T) grid. Keep nS, nT moderate (~40)."""
    Sv = np.linspace(S_min, S_max, nS)
    Tv = np.linspace(T_min, T_max, nT)
    Sg, Tg = np.meshgrid(Sv, Tv, indexing="ij")
    flat = np.array([
        call(float(s), K, float(t), r, v0, kappa, theta, xi, rho)
        for s, t in zip(Sg.ravel(), Tg.ravel())
    ])
    V = flat.reshape(Sg.shape)
    if cp == "put":
        V = V - Sg + K * np.exp(-r * Tg)
    return {"S": Sg, "T": Tg, "V": V,
            "params": dict(K=K, r=r, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)}


# --- Demo ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    from time import time

    plotdir = Path(__file__).resolve().parent.parent / "plots" / "analytical"
    plotdir.mkdir(parents=True, exist_ok=True)

    S0, K, T0, r = 100.0, 100.0, 1.0, 0.05
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7

    t0 = time()
    c = call(S0, K, T0, r, v0, kappa, theta, xi, rho)
    p = put(S0, K, T0, r, v0, kappa, theta, xi, rho)
    dt = time() - t0

    print(f"Heston  Call: {c:.6f}   Put: {p:.6f}  ({dt:.3f}s)")
    print(f"Parity: C-P={c-p:.6f}, S-Ke^(-rT)={S0 - K*np.exp(-r*T0):.6f}")

    c_cos = call_cos(S0, K, T0, r, v0, kappa, theta, xi, rho, N=256)
    print(f"COS call: {c_cos:.6f}")

    print("\nSurface (40x40)...")
    t0 = time()
    surf = surface(nS=40, nT=40)
    print(f"Done in {time()-t0:.1f}s")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))
    ax.plot_surface(surf["S"], surf["T"], surf["V"],
                    cmap="inferno", edgecolor="none", alpha=0.9)
    ax.set_xlabel("S"); ax.set_ylabel("T"); ax.set_zlabel("V(S,T)")
    ax.set_title("Heston Call Surface")
    plt.tight_layout()
    out = plotdir / "heston_surface.pdf"
    plt.savefig(out)
    print(f"\nSaved to {out}")
    plt.show()
