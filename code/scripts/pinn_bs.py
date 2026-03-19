"""pinn_bs.py
-----------
PINN solver for the Black-Scholes PDE.

  PDE (τ = T−t):  ∂V/∂τ = ½σ²S²·V_SS + rS·V_S − rV
  IC  (τ = 0):    V(S,0) = max(S−K, 0)
  BC  lower:      V(S→0, τ) = 0
  BC  upper:      V(S_max, τ) ≈ S_max − K·e^{−rτ}

  Loss:  L = λ_pde·L_pde + λ_ic·L_ic + λ_bc·L_bc
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


# ── Network ──────────────────────────────────────────────────────────────────

class Net(nn.Module):
    """Fully-connected network: (S/K, τ) → V."""

    def __init__(self, layers=(2, 64, 64, 64, 64, 1)):
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

class BSPINN:
    """PINN for the Black-Scholes equation."""

    def __init__(self, K=100.0, r=0.05, sig=0.20,
                 layers=(2, 64, 64, 64, 64, 1),
                 lam=(1.0, 10.0, 5.0),
                 device=None):
        self.K   = K
        self.r   = r
        self.sig = sig
        self.lam = lam  # (pde, ic, bc)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device  = torch.device(dev)
        self.net     = Net(layers).to(self.device)
        self.opt     = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.history = {"pde": [], "ic": [], "bc": []}

    # ── forward ──────────────────────────────────────────────────────────────

    def _fwd(self, S, tau):
        """V(S, τ) — raw network pass."""
        return self.net(torch.stack([S / self.K, tau], dim=1)).squeeze(-1)

    def _residual(self, S, tau):
        """BS PDE residual."""
        S.requires_grad_(True)
        tau.requires_grad_(True)
        V = self._fwd(S, tau)
        (V_tau,) = torch.autograd.grad(V.sum(), tau, create_graph=True)
        (V_S,)   = torch.autograd.grad(V.sum(), S,   create_graph=True)
        (V_SS,)  = torch.autograd.grad(V_S.sum(), S, create_graph=True)
        return V_tau - 0.5 * self.sig**2 * S**2 * V_SS - self.r * S * V_S + self.r * V

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, n_steps=5000, n_col=2000, n_ic=500, n_bc=200,
              S_min=10.0, S_max=300.0, tau_max=2.0,
              log=500, lr_decay=True):
        """Train the PINN."""
        dev   = self.device
        opt   = self.opt
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
                 if lr_decay else None)

        def rS(n): return torch.rand(n, device=dev) * (S_max - S_min) + S_min
        def rT(n): return torch.rand(n, device=dev) * tau_max

        for step in range(1, n_steps + 1):
            # PDE interior
            l_pde = self._residual(rS(n_col), rT(n_col)).pow(2).mean()

            # IC: V(S, 0) = max(S − K, 0)
            S_ic = rS(n_ic)
            l_ic = (self._fwd(S_ic, torch.zeros(n_ic, device=dev))
                    - (S_ic - self.K).clamp(min=0)).pow(2).mean()

            # BC lower: V(S_min, τ) ≈ 0
            tau_bc = rT(n_bc)
            l_bc   = self._fwd(torch.full((n_bc,), S_min, device=dev), tau_bc).pow(2).mean()

            # BC upper: V(S_max, τ) ≈ S_max − K·e^{−rτ}
            S_hi = torch.full((n_bc,), S_max, device=dev)
            l_bc = l_bc + (self._fwd(S_hi, tau_bc)
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

    def predict(self, S, tau):
        """V(S, τ) — arrays or scalars → numpy array."""
        self.net.eval()
        with torch.no_grad():
            S_t   = torch.as_tensor(np.asarray(S,   float), dtype=torch.float32, device=self.device).flatten()
            tau_t = torch.as_tensor(np.asarray(tau, float), dtype=torch.float32, device=self.device).flatten()
            V = self._fwd(S_t, tau_t)
        self.net.train()
        return V.cpu().numpy()

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net":    self.net.state_dict(),
            "history": self.history,
            "params": dict(K=self.K, r=self.r, sig=self.sig),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        self.history = ckpt.get("history", self.history)
        return self


# ── demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bs import call as bs_call

    K, r, sig = 100.0, 0.05, 0.20
    model = BSPINN(K=K, r=r, sig=sig)
    model.train(n_steps=5000, log=500)

    S    = np.linspace(50, 200, 100)
    tau  = 1.0
    V_nn = model.predict(S, np.full_like(S, tau))
    V_ex = bs_call(S, K, tau, r, sig)
    rmse = np.sqrt(np.mean((V_nn - V_ex) ** 2))
    print(f"\nRMSE vs analytical (τ=1): {rmse:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(S, V_ex, "k-",  label="exact")
    ax1.plot(S, V_nn, "r--", label="PINN")
    ax1.set(xlabel="S", ylabel="V", title="BS PINN vs Exact  (τ=1)")
    ax1.legend()

    ax2.plot(S, V_nn - V_ex)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set(xlabel="S", ylabel="error", title="Pointwise Error")

    plt.tight_layout()
    out = Path(__file__).parent.parent / "plots" / "pinn" / "bs_pinn.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"Saved → {out}")
    plt.show()
