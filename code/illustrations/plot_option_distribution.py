"""
plot_option_distribution.py
----------------------------
Generates a publication-quality PDF illustrating how a call option's price
corresponds to the shaded area under the probability distribution to the
right of the strike price K.

Output: ../plots/option_as_distribution.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
S0    = 100      # current stock price (mean of distribution)
sigma = 15       # volatility (std dev of distribution)
K     = 118      # strike price — positioned in the right tail

# ── Build the distribution ────────────────────────────────────────────────────
x = np.linspace(S0 - 4 * sigma, S0 + 4 * sigma, 1000)
y = norm.pdf(x, loc=S0, scale=sigma)

# ── Colour palette (premium, not generic) ─────────────────────────────────────
BLUE_LINE   = "#3B82F6"       # distribution curve
GREEN_FILL  = "#34D399"       # shaded option-price area
GREEN_DARK  = "#059669"       # annotation text for price
RED_STRIKE  = "#EF4444"       # strike marker
BG_COLOR    = "#FFFFFF"       # clean white background
AXIS_COLOR  = "#1E293B"       # near-black axes
TEXT_COLOR  = "#334155"       # text / labels
GRID_COLOR  = "#E2E8F0"      # subtle grid

# ── Figure: 12 × 4 ratio ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Plot the bell curve ──────────────────────────────────────────────────────
ax.plot(x, y, color=BLUE_LINE, linewidth=2.8, zorder=3)

# ── Shade the area to the right of K (= option price) ────────────────────────
mask = x >= K
ax.fill_between(x[mask], y[mask], alpha=0.35, color=GREEN_FILL,
                edgecolor=GREEN_DARK, linewidth=1.2, zorder=2,
                label="Price of the option")

# ── Strike vertical line ─────────────────────────────────────────────────────
ax.axvline(K, color=RED_STRIKE, linewidth=1.8, linestyle="--", zorder=4)

# ── Annotations ──────────────────────────────────────────────────────────────

# "Strike (K)" — red arrow pointing UP from below the x-axis
ax.annotate(
    "Strike (K)",
    xy=(K, 0), xytext=(K, -0.0065),
    fontsize=14, fontweight="bold", color=RED_STRIKE,
    ha="center", va="top",
    arrowprops=dict(arrowstyle="-|>", color=RED_STRIKE, lw=2.2),
    annotation_clip=False
)

# "Price of the option" — green arrow pointing DOWN into the shaded region
shade_mid_x = K + (S0 + 2.5 * sigma - K) / 2    # keep label inside plot
shade_mid_y = norm.pdf(shade_mid_x, S0, sigma)   # height at that midpoint
ax.annotate(
    "Price of the option",
    xy=(shade_mid_x, shade_mid_y * 0.4),
    xytext=(K + 5, max(y) * 0.70),
    fontsize=14, fontweight="bold", color=GREEN_DARK,
    ha="left", va="bottom",
    arrowprops=dict(arrowstyle="-|>", color=GREEN_DARK, lw=2.2,
                    connectionstyle="arc3,rad=-0.2"),
)

# "Current price S₀" — small label at the peak
ax.annotate(
    f"Current price S\u2080 = {S0}",
    xy=(S0, max(y)),
    xytext=(S0 - 22, max(y) * 1.08),
    fontsize=10, color=TEXT_COLOR, fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1),
)

# ── Axis labels ──────────────────────────────────────────────────────────────
ax.set_ylabel("Probability", fontsize=14, fontweight="bold", color=AXIS_COLOR,
              labelpad=10)
ax.set_xlabel("Stock Price at Expiry", fontsize=13, color=TEXT_COLOR, labelpad=8)

# ── Axis styling ─────────────────────────────────────────────────────────────
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(AXIS_COLOR)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_color(AXIS_COLOR)
ax.spines["bottom"].set_linewidth(2)
ax.tick_params(axis="both", which="both", length=0, labelsize=10, colors=TEXT_COLOR)
ax.set_yticks([])                         # probability axis — no numeric ticks
ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.009, max(y) * 1.18)       # room for annotation below axis

# ── Subtle grid ──────────────────────────────────────────────────────────────
ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.6)

# ── Layout & save ────────────────────────────────────────────────────────────
fig.tight_layout()
for ext in ["pdf", "png"]:
    out_path = os.path.join(OUT_DIR, f"option_as_distribution.{ext}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved -> {out_path}")
plt.close(fig)
