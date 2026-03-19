"""pinn_heston.py
---------------
PINN solver for the Heston PDE (Phase 3, due May 9).

  Heston PDE (τ = T−t, inputs: S, v, τ):
    ∂V/∂τ = ½vS²·V_SS + ρξvS·V_Sv + ½ξ²v·V_vv
            + rS·V_S + κ(θ−v)·V_v − rV
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ── Network ──────────────────────────────────────────────────────────────────

class Net(nn.Module):
    """Fully-connected network: (S/K, v, τ) → V."""

    def __init__(self, layers=(3, 64, 64, 64, 64, 1)):
        super().__init__()
        act = nn.Tanh()
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq.append(act)
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# ── PINN ─────────────────────────────────────────────────────────────────────

class HestonPINN:
    """PINN for the Heston stochastic-volatility equation."""

    def __init__(self, K=100.0, r=0.05,
                 kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                 layers=(3, 64, 64, 64, 64, 1),
                 lam=(1.0, 10.0, 5.0),
                 device=None):
        self.K     = K
        self.r     = r
        self.kappa = kappa
        self.theta = theta
        self.xi    = xi
        self.rho   = rho
        self.lam   = lam  # (pde, ic, bc)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device  = torch.device(dev)
        self.net     = Net(layers).to(self.device)
        self.opt     = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.history = {"pde": [], "ic": [], "bc": []}

    # ── forward ──────────────────────────────────────────────────────────────

    def _fwd(self, S, v, tau):
        return self.net(torch.stack([S / self.K, v, tau], dim=1)).squeeze(-1)

    def _residual(self, S, v, tau):
        """Heston PDE residual."""
        S.requires_grad_(True)
        v.requires_grad_(True)
        tau.requires_grad_(True)
        V = self._fwd(S, v, tau)
        (V_tau,) = torch.autograd.grad(V.sum(), tau, create_graph=True)
        (V_S,)   = torch.autograd.grad(V.sum(), S,   create_graph=True)
        (V_SS,)  = torch.autograd.grad(V_S.sum(), S, create_graph=True)
        (V_v,)   = torch.autograd.grad(V.sum(), v,   create_graph=True)
        (V_vv,)  = torch.autograd.grad(V_v.sum(), v, create_graph=True)
        (V_Sv,)  = torch.autograd.grad(V_S.sum(), v, create_graph=True)
        return (V_tau
                - 0.5 * v * S**2 * V_SS
                - self.rho * self.xi * v * S * V_Sv
                - 0.5 * self.xi**2 * v * V_vv
                - self.r * S * V_S
                - self.kappa * (self.theta - v) * V_v
                + self.r * V)

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, n_steps=8000, n_col=2000, n_ic=500, n_bc=200,
              S_min=10.0, S_max=300.0, v_min=1e-4, v_max=0.5, tau_max=2.0,
              log=500, lr_decay=True):
        """Train the PINN."""
        dev   = self.device
        opt   = self.opt
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
                 if lr_decay else None)

        def rS(n): return torch.rand(n, device=dev) * (S_max - S_min) + S_min
        def rv(n): return torch.rand(n, device=dev) * (v_max - v_min) + v_min
        def rT(n): return torch.rand(n, device=dev) * tau_max

        for step in range(1, n_steps + 1):
            # PDE interior
            l_pde = self._residual(rS(n_col), rv(n_col), rT(n_col)).pow(2).mean()

            # IC: V(S, v, 0) = max(S − K, 0)
            S_ic = rS(n_ic); v_ic = rv(n_ic)
            l_ic = (self._fwd(S_ic, v_ic, torch.zeros(n_ic, device=dev))
                    - (S_ic - self.K).clamp(min=0)).pow(2).mean()

            # BC lower S: V ≈ 0
            tau_bc = rT(n_bc); v_bc = rv(n_bc)
            l_bc   = self._fwd(torch.full((n_bc,), S_min, device=dev), v_bc, tau_bc).pow(2).mean()

            # BC upper S: V ≈ S_max − K·e^{−rτ}
            S_hi = torch.full((n_bc,), S_max, device=dev)
            l_bc = l_bc + (self._fwd(S_hi, v_bc, tau_bc)
                           - (S_hi - self.K * torch.exp(-self.r * tau_bc))).pow(2).mean()

            loss = self.lam[0] * l_pde + self.lam[1] * l_ic + self.lam[2] * l_bc
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched:
                sched.step()

            self.history["pde"].append(l_pde.item())
            self.history["ic"].append(l_ic.item())
            self.history["bc"].append(l_bc.item())

            if log and step % log == 0:
                print(f"[{step:5d}]  pde={l_pde:.3e}  ic={l_ic:.3e}  bc={l_bc:.3e}")

        return self

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, S, v, tau):
        """V(S, v, τ) — arrays or scalars → numpy array."""
        self.net.eval()
        with torch.no_grad():
            t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32, device=self.device).flatten()
            V = self._fwd(t(S), t(v), t(tau))
        self.net.train()
        return V.cpu().numpy()

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net":    self.net.state_dict(),
            "history": self.history,
            "params": dict(K=self.K, r=self.r, kappa=self.kappa,
                           theta=self.theta, xi=self.xi, rho=self.rho),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        self.history = ckpt.get("history", self.history)
        return self


# ── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Heston PINN — Phase 3 (due May 9)")
    model = HestonPINN()
    print(f"  Network: {sum(p.numel() for p in model.net.parameters()):,} params")
    print(f"  Device:  {model.device}")
