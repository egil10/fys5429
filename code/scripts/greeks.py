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
