"""run.py
-------
Entry point for PINN option pricing experiments.

  python run.py --model bs --steps 5000 --seed 42
"""

import argparse
import numpy as np
from pathlib import Path

from fys5429.utils import set_seed


def _run_bs(args):
    from fys5429.pinn_bs import BSPINN
    from fys5429.bs import call as bs_call
    from fys5429.utils import plot_loss

    model = BSPINN(K=args.K, r=args.r, sig=args.sig)
    model.train(n_steps=args.steps, log=args.log)

    out = Path(__file__).parent.parent / "plots" / "pinn"
    model.save(out / "bs_pinn.pt")
    plot_loss(model.history, path=out / "bs_loss.pdf")

    # Validation slice at τ=1
    S    = np.linspace(50, 200, 200)
    tau  = 1.0
    V_nn = model.predict(S, np.full_like(S, tau))
    V_ex = bs_call(S, args.K, tau, args.r, args.sig)
    rmse = np.sqrt(np.mean((V_nn - V_ex) ** 2))
    print(f"\nRMSE vs analytical (τ=1): {rmse:.6f}")


def _run_heston(args):
    raise NotImplementedError("Heston PINN — Phase 3 (due May 9)")


def main():
    p = argparse.ArgumentParser(description="PINN option pricing")
    p.add_argument("--model",  choices=["bs", "heston"], default="bs")
    p.add_argument("--steps",  type=int,   default=5000)
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--log",    type=int,   default=500,   help="print every N steps")
    p.add_argument("--K",      type=float, default=100.0, help="strike")
    p.add_argument("--r",      type=float, default=0.05,  help="risk-free rate")
    p.add_argument("--sig",    type=float, default=0.20,  help="volatility (BS only)")
    args = p.parse_args()

    set_seed(args.seed)
    print(f"model={args.model}  steps={args.steps}  seed={args.seed}")

    {"bs": _run_bs, "heston": _run_heston}[args.model](args)


if __name__ == "__main__":
    main()
