# FYS5429 - Advanced Machine Learning and Data Analysis for the Physical Sciences

**Physics-Informed Neural Networks for Option Pricing**

---

## Project Overview

This repository contains coursework for FYS5429, focusing on the application of Physics-Informed Neural Networks (PINNs) to solve partial differential equations arising in quantitative finance. The primary objective is to develop and analyze PINN-based solvers for option pricing models, progressing from the classical Black-Scholes equation to the more complex Heston stochastic volatility model.

---

## Project Focus: PINNs for Option Pricing

**From Black-Scholes to Stochastic Volatility Calibration**

This project explores how deep learning can be combined with domain knowledge from mathematical finance to solve and calibrate option pricing PDEs:

- **Black-Scholes PINN**: Solving the classical Black-Scholes PDE using physics-informed loss functions that encode the governing equation, boundary conditions, and terminal payoff.

- **Heston PINN**: Extending the methodology to the Heston model, which captures stochastic volatility dynamics through a system of coupled PDEs.

- **Model Calibration**: Using PINNs for the inverse problem of calibrating model parameters to observed market data.

---

## Repository Structure

```
fys5429/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- FYS5429 - PINNs for Option Pricing [...].pdf
|
|-- code/
    |-- data/                       # Training and validation datasets (gitignored)
    |-- notebooks/                  # Jupyter notebooks for analysis
    |-- plots/                      # Generated figures (PDF only)
    |
    |-- scripts/
    |   |-- data_simulation.py      # Synthetic data generation (GBM, Heston paths)
    |   |-- pinn_black_scholes.py   # PINN implementation for Black-Scholes
    |   |-- pinn_heston.py          # PINN implementation for Heston model
    |   |-- calibration.py          # Model calibration utilities
    |   |-- utils.py                # Plotting, I/O, and helper functions
    |   |-- run.py                  # Main entry point and experiment runner
    |
    |-- books/
        |-- raschka/                # Notes from Raschka ML book
        |-- goodfellow/             # Notes from Goodfellow DL book
```

---

## Methodology

### Physics-Informed Neural Networks

PINNs incorporate physical laws directly into the neural network training process by adding PDE residual terms to the loss function. For option pricing:

1. **Forward Problem**: Given model parameters, solve for option prices across the (S, t) domain
2. **Inverse Problem**: Given observed option prices, calibrate model parameters

### Models Implemented

| Model | PDE Type | Key Features |
|-------|----------|--------------|
| Black-Scholes | Parabolic PDE | Constant volatility, closed-form benchmark |
| Heston | Coupled PDE system | Stochastic volatility, mean-reversion |

---

## Learning Objectives

- Understand the mathematical formulation of option pricing PDEs
- Implement physics-informed loss functions encoding PDE constraints
- Apply automatic differentiation for computing PDE residuals
- Compare PINN solutions against analytical benchmarks
- Explore calibration as an inverse problem

---

## Prerequisites

- **FYS-STK4155** - Applied Data Analysis and Machine Learning
- Strong background in calculus and differential equations
- Familiarity with Python and deep learning frameworks (PyTorch/TensorFlow)

---

## Resources

- [Course Page - University of Oslo](https://www.uio.no/studier/emner/matnat/fys/FYS5429/)
- [Physics-Informed Neural Networks - Raissi et al.](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Applied Data Analysis and Machine Learning - FYS-STK4155](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/)

---

## Course Information

- **Credits:** 10
- **Level:** Master
- **Teaching:** Spring semester
- **Language:** English

---

> *"The goal is to turn data into information, and information into insight."*
>
> -- Carly Fiorina

---

**Department of Physics, University of Oslo**