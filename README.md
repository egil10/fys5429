# FYS5429 — PINNs for Option Pricing

*Can a neural network learn to price options by understanding the physics?*

University of Oslo · Spring 2026 · Egil Furnes

---

## The idea

Option pricing is a PDE problem. The Black-Scholes equation is structurally identical to the heat equation — a fact Fischer Black and Myron Scholes exploited in 1973 to derive their celebrated closed form. Heston's 1993 extension adds a second stochastic dimension for volatility, breaking the closed form but preserving the PDE structure.

Physics-Informed Neural Networks (PINNs) turn this into a learning problem: instead of discretising the domain on a mesh, a neural network is trained to satisfy the PDE everywhere simultaneously. The PDE residual — computed via automatic differentiation — enters directly into the loss function. No labelled data required.

---

## Why it's interesting

**It connects two fields.** Financial PDEs and deep learning rarely meet this cleanly. The Black-Scholes equation under a log-price change *is* a diffusion equation — the same one that governs heat flow, Brownian motion, and quantum mechanics in imaginary time.

**The inverse problem is the real prize.** Markets give you prices, not parameters. Calibrating the Heston model — recovering κ, θ, ξ, ρ from observed option prices — is an ill-posed nonlinear optimisation problem. A PINN that has internalised the Heston PDE can be repurposed as a differentiable pricer, making gradient-based calibration natural.

**Mesh-free scales.** Classical finite-difference methods on a 2D (S, v) grid are expensive. PINNs sample collocation points randomly and scale to higher dimensions without a grid — relevant for multi-asset and path-dependent extensions.

**Activation functions matter.** The smoothness of the solution — and the sharpness of the terminal payoff — make the choice of activation non-trivial. This project benchmarks tanh, Swish, GELU, Softplus, and SIREN against each other.

---

## The models

**Black-Scholes** (Phase 1)

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV, \qquad \tau = T - t$$

Closed-form solution available — used as ground truth to validate the PINN.

**Heston** (Phase 3)

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}vS^2 V_{SS} + \rho\xi v S\, V_{Sv} + \frac{1}{2}\xi^2 v\, V_{vv} + rS\, V_S + \kappa(\theta - v)\, V_v - rV$$

Five parameters: mean-reversion speed κ, long-run variance θ, vol-of-vol ξ, spot-vol correlation ρ, initial variance v₀. No closed form — numerical integration (characteristic functions) serves as benchmark.

---

## Loss function

$$\mathcal{L} = \lambda_\text{pde}\,\mathcal{L}_\text{pde} + \lambda_\text{ic}\,\mathcal{L}_\text{ic} + \lambda_\text{bc}\,\mathcal{L}_\text{bc}$$

All three terms computed via PyTorch autograd — no finite differences anywhere in the training loop.

---

## Roadmap

| Phase | Deadline | |
|---|---|---|
| 1 | Apr 4 | BS PINN — validate against closed-form solution |
| 2 | Apr 18 | Activation study: tanh · Swish · GELU · Softplus · SIREN |
| 3 | May 9 | Heston PINN — 2D PDE, mixed derivative |
| 4 | May 23 | Inverse problem — Heston calibration from prices |
| 5 | Jun 1 | Write-up and submission |

---

Department of Physics · University of Oslo
