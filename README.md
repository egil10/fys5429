# FYS5429 — PINNs for Option Pricing

Physics-Informed Neural Networks applied to the Black-Scholes and Heston PDEs.
University of Oslo · Spring 2026 · Egil Furnes

---

## What this is

Standard neural networks learn from data. PINNs also satisfy a PDE — the governing equation is baked directly into the loss function via automatic differentiation.

This project uses PINNs to price European options under two models:

| Model | PDE | Status |
|---|---|---|
| Black-Scholes | Parabolic, 1D | Phase 1 — in progress |
| Heston | Coupled, 2D (S, v) | Phase 3 — pending |

Both the **forward problem** (price given params) and the **inverse problem** (calibrate params from prices) are covered.

---

## Structure

```
code/scripts/
  bs.py           analytical Black-Scholes (pricing + Greeks)
  heston.py       semi-analytical Heston (CF integration + COS)
  generate.py     synthetic data generation
  pinn_bs.py      PINN solver — Black-Scholes
  pinn_heston.py  PINN solver — Heston
  calibrate.py    parameter calibration (inverse problem)
  greeks.py       Greeks: analytical, numerical FD, PINN
  metrics.py      RMSE, MAE, MAPE, rel-L2, max error
  style.py        matplotlib style + colour palette
  utils.py        seeding, plotting, model I/O
  run.py          experiment entry point
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd code/scripts

python run.py --model bs --steps 5000        # train BS PINN
python pinn_bs.py                            # demo with plots
python greeks.py                             # plot all BS Greeks
python bs.py                                 # analytical surface
```

---

## PINN loss

```
L = λ_pde · L_pde  +  λ_ic · L_ic  +  λ_bc · L_bc
```

- `L_pde` — PDE residual at random collocation points (autograd)
- `L_ic` — terminal payoff V(S, 0) = max(S − K, 0)
- `L_bc` — boundary conditions at S = 0 and S = S_max

---

## Roadmap

| Phase | Deadline | Goal |
|---|---|---|
| 1 | Apr 4 | BS PINN — validate against analytical solution |
| 2 | Apr 18 | Activation study: tanh, Swish, GELU, Softplus, SIREN |
| 3 | May 9 | Heston PINN (2D PDE, mixed derivative) |
| 4 | May 23 | Heston calibration (inverse problem) |
| 5 | Jun 1 | Write-up and submission |

---

Department of Physics · University of Oslo
