# CLAUDE.md

Project context for Claude Code.

## Project

FYS5429 — Physics-Informed Neural Networks for Option Pricing.
University of Oslo, Spring 2026. Submission: June 1, 2026.

Author: Egil Furnes (egilsf@uio.no)

## Stack

- Python 3.11+ / PyTorch (>= 2.10)
- scipy, numpy, pandas, matplotlib, joblib
- LaTeX (Overleaf, mirrored in `.article/`) — pdflatex + biber

## Scripts (`code/scripts/`)

| File | Purpose | Status |
|---|---|---|
| `bs.py` | Analytical BS pricing + Greeks | done |
| `heston.py` | Heston CF integration + COS | done |
| `generate.py` | Synthetic data generation | done |
| `pinn_bs.py` | BSPINN class | Phase 1 |
| `pinn_heston.py` | HestonPINN class | Phase 3 |
| `calibrate.py` | BS IV + Heston calibration | Phase 4 |
| `greeks.py` | Greeks: exact, FD, PINN | utility |
| `metrics.py` | rmse, mae, mape, rel_l2 | utility |
| `style.py` | matplotlib style + palette | utility |
| `utils.py` | seeding, plotting, model I/O | utility |
| `run.py` | CLI entry point | always |

## Running

Install once (venv activated, repo root):

```bash
pip install -r requirements.txt
```

Then use module mode from repo root (or any cwd):

```bash
python -m fys5429.run --model bs --steps 5000
python -m fys5429.pinn_bs          # demo + plots
python -m fys5429.greeks           # Greek plots
python -m fys5429.generate         # regenerate data
```

Notebooks: `from fys5429.bm import bm` (no `sys.path` hack after install).

## Conventions

- Short names: `BSPINN`, `pinn_bs.py` — not `BlackScholesPINN`, `pinn_black_scholes.py`
- All scripts have a `if __name__ == "__main__":` demo block
- Plots save to `code/plots/<subfolder>/` as PDF
- Models save to `code/plots/pinn/` as `.pt`
- Generated data lives in `code/data/generated/`

## Phases

| Phase | Deadline | Goal |
|---|---|---|
| 1 | 2026-04-04 | BS-PINN forward, validate vs analytical |
| 2 | 2026-04-18 | Activation study: tanh, Swish, GELU, Softplus, SIREN |
| 3 | 2026-05-09 | Heston PINN |
| 4 | 2026-05-23 | Heston calibration |
| 5 | 2026-06-01 | Write-up + submission |

## Article (`.article/`)

- Root: `00 MAIN.tex` — compiles all sections via `\input{}`
- Cite: `\textcite{}` inline, `\parencite{}` parenthetical
- Cross-ref: `\cref{}`
- `.article/` is git-ignored (synced to Overleaf separately)
- Current state: INTRO + METHOD written; RESULT + CONCLUSION pending
