"""calibrate.py
-----------
Inverse problem: calibrate BS / Heston parameters from observed prices.

  Phase 4 (due May 23).

  Usage:
    from calibrate import calibrate_bs, calibrate_heston
"""

import numpy as np
from scipy.optimize import minimize


# ── Black-Scholes calibration ────────────────────────────────────────────────

def calibrate_bs(prices, S, K, T, r):
    """Implied volatility per option via Newton-Raphson (wraps bs.iv).

    Args:
        prices: observed call prices, shape (n,)
        S, K, T, r: option parameters, shape (n,)

    Returns:
        iv: implied volatility array, shape (n,)
    """
    from bs import iv
    return np.array([iv(p, s, k, t, r)
                     for p, s, k, t in zip(prices, S, K, T)])


# ── Heston calibration ────────────────────────────────────────────────────────

def calibrate_heston(prices, S, K, T, r,
                     p0=(0.04, 2.0, 0.04, 0.3, -0.7),
                     bounds=((1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0),
                             (0.01, 1.0), (-0.99, 0.0)),
                     method="L-BFGS-B"):
    """Calibrate Heston (v0, κ, θ, ξ, ρ) by minimising mean squared error.

    Args:
        prices: observed call prices, shape (n,)
        S, K, T, r: option parameters, shape (n,)
        p0: initial guess (v0, kappa, theta, xi, rho)
        bounds: parameter bounds

    Returns:
        dict with keys v0, kappa, theta, xi, rho, mse, success
    """
    from heston import call as heston_call

    def obj(params):
        v0, kappa, theta, xi, rho = params
        try:
            phat = np.array([
                heston_call(s, k, t, r, v0, kappa, theta, xi, rho)
                for s, k, t in zip(S, K, T)
            ])
        except Exception:
            return 1e10
        return np.mean((phat - prices) ** 2)

    res = minimize(obj, p0, bounds=bounds, method=method,
                   options={"maxiter": 1000, "ftol": 1e-12})
    v0, kappa, theta, xi, rho = res.x
    return dict(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
                mse=res.fun, success=res.success)


# ── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Calibration module — Phase 4 (due May 23)")
