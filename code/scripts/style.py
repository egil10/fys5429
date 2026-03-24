"""style.py
--------
Matplotlib style configuration. Import once at the top of any script.

  from fys5429.style import apply; apply()
  # or
  import fys5429.style as style; style.apply()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Palette ───────────────────────────────────────────────────────────────────

COLORS = {
    "exact":    "#111111",   # black  — analytical / reference
    "pinn":     "#c0392b",   # red    — PINN prediction
    "error":    "#2980b9",   # blue   — error / residual
    "tanh":     "#27ae60",   # green
    "swish":    "#8e44ad",   # purple
    "gelu":     "#e67e22",   # orange
    "softplus": "#16a085",   # teal
    "siren":    "#2c3e50",   # dark
}

CMAPS = {
    "surface": "viridis",
    "error":   "Reds",
    "heston":  "inferno",
}


# ── Apply ─────────────────────────────────────────────────────────────────────

def apply(fontsize=11, usetex=False):
    """Set global rcParams for a clean, publication-ready style."""
    mpl.rcParams.update({
        # Font
        "font.family":       "serif",
        "font.size":         fontsize,
        "axes.titlesize":    fontsize,
        "axes.labelsize":    fontsize,
        "xtick.labelsize":   fontsize - 1,
        "ytick.labelsize":   fontsize - 1,
        "legend.fontsize":   fontsize - 1,

        # Lines
        "lines.linewidth":   1.6,
        "lines.markersize":  5,

        # Axes
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",

        # Figure
        "figure.dpi":        150,
        "figure.autolayout": True,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",

        # TeX (optional)
        "text.usetex":       usetex,
    })
    if usetex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def savefig(path, fig=None, **kwargs):
    """Save current (or given) figure to path, creating dirs as needed."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    (fig or plt).savefig(path, **kwargs)


# Apply on import
apply()
