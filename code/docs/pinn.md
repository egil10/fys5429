# PINN Overview

## Black-Scholes PDE

In τ = T − t (time to maturity):

```
∂V/∂τ = ½σ²S²·V_SS + rS·V_S − rV
```

**IC**: V(S, 0) = max(S − K, 0)
**BC lower**: V(0, τ) = 0
**BC upper**: V(S_max, τ) ≈ S_max − K·e^{−rτ}

## Loss Function

```
L = λ_pde · L_pde  +  λ_ic · L_ic  +  λ_bc · L_bc
```

Default weights: `lam = (1.0, 10.0, 5.0)` — IC weighted heavily because the terminal payoff is the sharpest feature.

## Network

```
Input  (2): (S/K, τ)
Hidden (4): 64 neurons each, Tanh
Output (1): V
```

Total params: ~16k (fast to train, ~2 min CPU for 5k steps).

## Training

```python
from pinn_bs import BSPINN
model = BSPINN(K=100, r=0.05, sig=0.20)
model.train(n_steps=5000, log=500)
```

## Validation

```python
from bs import call as bs_call
from metrics import summary
import numpy as np

S   = np.linspace(50, 200, 200)
tau = 1.0
summary(model.predict(S, tau), bs_call(S, 100, tau, 0.05, 0.2), label="BS PINN τ=1")
```

## Heston PDE

In τ = T − t, inputs (S, v, τ):

```
∂V/∂τ = ½vS²·V_SS + ρξvS·V_Sv + ½ξ²v·V_vv
        + rS·V_S + κ(θ−v)·V_v − rV
```

Network input becomes (S/K, v, τ) — 3D, 5-param model.
See `pinn_heston.py`, Phase 3 (due May 9).
