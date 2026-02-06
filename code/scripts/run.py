"""
run.py
------
Main entry point for running PINN option pricing experiments.

This script orchestrates:
- Data generation/loading
- Model training (Black-Scholes and/or Heston)
- Calibration
- Results visualization
"""

import argparse

from data_simulation import generate_training_data
from pinn_black_scholes import BlackScholesPINN
from pinn_heston import HestonPINN
from calibration import calibrate_black_scholes, calibrate_heston
from utils import set_seed, plot_solution


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="PINN Option Pricing")
    parser.add_argument("--model", type=str, default="bs", choices=["bs", "heston"],
                        help="Model type: 'bs' for Black-Scholes, 'heston' for Heston")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Running PINN with model: {args.model}")
    print("This is a placeholder - implementation pending")

    # TODO: Implement full pipeline
    raise NotImplementedError("Full pipeline not yet implemented")


if __name__ == "__main__":
    main()
