# FYS5429 — Physics-Informed Neural Networks for Option Pricing

**Advanced Machine Learning and Data Analysis for the Physical Sciences**
University of Oslo, Spring 2026

---

## Overview

This project applies Physics-Informed Neural Networks (PINNs) to option pricing in quantitative finance. PINNs embed the governing PDE directly into the loss function, enabling the network to learn solutions that are consistent with the underlying financial mathematics — without requiring large labeled datasets.

The project progresses through two pricing models:

| Model | PDE | Highlights |
|-------|-----|------------|
| **Black-Scholes** | Parabolic PDE | Constant volatility, closed-form benchmark available |
| **Heston** | Coupled PDE system | Stochastic volatility with mean-reversion |

Both the **forward problem** (pricing given parameters) and the **inverse problem** (calibrating parameters from market prices) are explored.

---

## Repository Structure

```
fys5429/
├── README.md
├── requirements.txt
├── FYS5429 - PINNs for Option Pricing.pdf   # Project report
│
└── code/
    ├── scripts/
    │   ├── run.py            # Main entry point — trains and evaluates models
    │   ├── pinn-bs.py        # PINN solver for the Black-Scholes PDE
    │   ├── pinn-heston.py    # PINN solver for the Heston model
    │   ├── bs.py             # Analytical Black-Scholes reference solution
    │   ├── heston.py         # Heston model utilities
    │   ├── generate.py       # Synthetic data generation (GBM / Heston paths)
    │   ├── calibrate.py      # Inverse problem: parameter calibration
    │   └── utils.py          # Shared helpers (plotting, I/O, seeding)
    │
    ├── notebooks/            # Exploratory analysis and result visualisation
    ├── plots/                # Generated figures (PDF)
    ├── data/                 # Training / validation data (git-ignored)
    └── rsc/                  # Project proposal and supplementary resources
```

---

## Getting Started

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run experiments

```bash
cd code/scripts

# Train the Black-Scholes PINN
python run.py --model bs

# Train the Heston PINN
python run.py --model heston

# Set a custom random seed
python run.py --model bs --seed 123
```

---

## Method

### PINN Loss Function

The network is trained to minimise a composite loss:

```
L = λ_pde · L_pde  +  λ_bc · L_bc  +  λ_ic · L_ic
```

- **L_pde** — residual of the Black-Scholes / Heston PDE at collocation points, computed via automatic differentiation
- **L_bc** — boundary conditions (e.g. option payoff at expiry, put-call parity at domain edges)
- **L_ic** — initial/terminal condition (payoff at maturity)

PDE gradients are obtained via PyTorch autograd — no finite-difference approximations.

### Calibration (Inverse Problem)

`calibrate.py` treats model parameters (e.g. σ for Black-Scholes, κ, θ, ξ, ρ for Heston) as trainable variables and fits them by minimising the difference between PINN-predicted prices and observed market quotes.

---

## Prerequisites

- Completed **FYS-STK4155** (Applied Data Analysis and Machine Learning) or equivalent
- Comfort with PDEs and stochastic calculus
- Python with PyTorch

---

## Key References

- Raissi, Perdikaris & Karniadakis (2019) — [Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Black & Scholes (1973) — The pricing of options and corporate liabilities
- Heston (1993) — A closed-form solution for options with stochastic volatility
- [FYS5429 course page](https://www.uio.no/studier/emner/matnat/fys/FYS5429/)

---

**Department of Physics, University of Oslo**
