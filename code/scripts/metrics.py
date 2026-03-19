"""metrics.py
----------
Evaluation metrics for PINN validation.
"""

import numpy as np


def rmse(pred, exact):
    """Root mean squared error."""
    return np.sqrt(np.mean((pred - exact) ** 2))


def mae(pred, exact):
    """Mean absolute error."""
    return np.mean(np.abs(pred - exact))


def mape(pred, exact, eps=1e-8):
    """Mean absolute percentage error (%)."""
    return 100.0 * np.mean(np.abs((pred - exact) / (np.abs(exact) + eps)))


def rel_l2(pred, exact):
    """Relative L2 error: ||pred − exact||₂ / ||exact||₂."""
    return np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-14)


def max_err(pred, exact):
    """Maximum absolute error."""
    return np.max(np.abs(pred - exact))


def summary(pred, exact, label="") -> dict:
    """All metrics in one dict. Optionally print a table.

    Args:
        pred, exact: arrays of the same shape
        label: printed header (empty = no print)

    Returns:
        dict with keys rmse, mae, mape, rel_l2, max_err
    """
    m = dict(
        rmse   = rmse(pred, exact),
        mae    = mae(pred, exact),
        mape   = mape(pred, exact),
        rel_l2 = rel_l2(pred, exact),
        max_err= max_err(pred, exact),
    )
    if label:
        print(f"\n{'-'*40}")
        print(f"  {label}")
        print(f"{'-'*40}")
        for k, v in m.items():
            unit = "%" if k == "mape" else ""
            print(f"  {k:<10} {v:.6f}{unit}")
        print(f"{'-'*40}")
    return m
