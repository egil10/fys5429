# Preliminary Results — 2026-03-19

Training steps per PINN: 5000


## pipeline

```
Loaded BS surface: 10,000 rows  cols=['S', 'T', 'K', 'r', 'sig', 'call', 'put']

Training BS PINN  (5000 steps, device=cpu)
[  500]  pde=5.476e-03  ic=1.107e-03  bc=5.010e-04
[ 1000]  pde=2.239e-03  ic=5.210e-04  bc=1.434e-04
[ 1500]  pde=1.020e-03  ic=2.651e-04  bc=1.580e-04
[ 2000]  pde=4.677e-04  ic=1.068e-04  bc=7.036e-05
[ 2500]  pde=3.066e-04  ic=9.658e-05  bc=1.677e-05
[ 3000]  pde=2.560e-04  ic=7.279e-05  bc=7.642e-06
[ 3500]  pde=2.366e-04  ic=8.413e-05  bc=9.943e-06
[ 4000]  pde=2.231e-04  ic=6.155e-05  bc=5.870e-06
[ 4500]  pde=2.253e-04  ic=7.674e-05  bc=5.961e-06
[ 5000]  pde=2.311e-04  ic=5.783e-05  bc=5.601e-06

Metrics on full 100x100 surface

----------------------------------------
  BS PINN vs analytical
----------------------------------------
  rmse       0.488537
  mae        0.380367
  mape       147903788.138090%
  rel_l2     0.019934
  max_err    3.099970
----------------------------------------

Saving plots...
  saved: bs_slices.pdf
  saved: bs_surface_comparison.pdf
  saved: bs_error_map.pdf
  saved: bs_loss.pdf
  saved: bs_greeks.pdf
  saved: bs_greeks_all.pdf

All plots saved to C:\Users\ofurn\Dokumenter\Github\fys5429\code\plots\pinn/
```


## pipeline_heston

```
Loaded Heston surface: 1,600 rows  cols=['S', 'T', 'K', 'r', 'v0', 'kappa', 'theta', 'xi', 'rho', 'call', 'put']

Training Heston PINN  (5000 steps, device=cpu)
[  500]  pde=1.756e+02  ic=2.851e+01  bc=5.899e+02
[ 1000]  pde=1.145e+02  ic=1.706e+00  bc=7.959e+01
[ 1500]  pde=2.431e+01  ic=8.951e-01  bc=1.997e+01
[ 2000]  pde=1.382e+01  ic=8.728e-01  bc=2.993e+00
[ 2500]  pde=1.854e+01  ic=6.885e-01  bc=1.243e+00
[ 3000]  pde=9.737e+00  ic=7.447e-01  bc=1.038e+00
[ 3500]  pde=1.036e+01  ic=5.587e-01  bc=5.543e-01
[ 4000]  pde=8.952e+00  ic=4.759e-01  bc=5.147e-01
[ 4500]  pde=8.910e+00  ic=5.281e-01  bc=5.074e-01
[ 5000]  pde=7.175e+00  ic=5.944e-01  bc=5.756e-01

Metrics on full Heston surface

----------------------------------------
  Heston PINN vs analytical
----------------------------------------
  rmse       2.275677
  mae        1.823841
  mape       22883955.457198%
  rel_l2     0.090580
  max_err    4.104863
----------------------------------------

Saving plots...
  saved: heston_slices.pdf
  saved: heston_surface_comparison.pdf
  saved: heston_error_map.pdf
  saved: heston_loss.pdf

All plots saved to C:\Users\ofurn\Dokumenter\Github\fys5429\code\plots\pinn/
```


## pipeline_calibrate

```
Loaded BS surface: 10,000 rows

BS implied vol calibration
  liquid options: 9,889 / 10,000
  true sigma  = 0.2000
  recovered   = 0.2009 +/- 0.013906
  max abs err = 1.27e+00
  saved: calibrate_iv_heatmap.pdf

Loaded Heston surface: 1,600 rows

Heston parameter calibration
  points: 30  noise: 1%  initial guess: (0.06, 1.5, 0.05, 0.4, -0.5)
  converged: True   MSE: 5.5176e-02

     param        true   recovered      err%
        v0      0.0400      0.0378     5.47%
     kappa      2.0000      1.9978     0.11%
     theta      0.0400      0.0417     4.20%
        xi      0.3000      0.3794    26.45%
       rho     -0.7000     -0.5591    20.13%
  saved: calibrate_heston_params.pdf

All plots saved to C:\Users\ofurn\Dokumenter\Github\fys5429\code\plots\calibrate/
```
