import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(x):
    """Inverse of softplus: y such that softplus(y) = x."""
    return x + torch.log(-torch.expm1(-x))


def _inv_tanh(x):
    """Inverse of tanh (atanh): y such that tanh(y) = x.  |x| < 1."""
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


class INVPINN(nn.Module):
    """
    Physics-Informed Neural Network for the INVERSE Heston problem.

    Key fix: Heston parameters are reparameterised so the optimiser
    always sees *unconstrained* raw values, while the model exposes
    *physically valid* parameters via properties:

        kappa = softplus(raw_kappa)          -> kappa > 0
        theta = softplus(raw_theta)          -> theta > 0
        xi    = softplus(raw_xi)             -> xi    > 0
        rho   = tanh(raw_rho)               -> rho in (-1, 1)
    """

    def __init__(self, hidden_layers=3, neurons_per_layer=256,
                 S_scale=300.0, v_scale=1.0, tau_scale=1.0, activation='tanh',
                 kappa_init=1.0, theta_init=0.1, xi_init=0.5, rho_init=0.0):
        super().__init__()

        self.S_scale = S_scale
        self.v_scale = v_scale
        self.tau_scale = tau_scale

        # ---------- constrained Heston parameters ----------
        # Store *raw* (unconstrained) values; the properties below
        # apply softplus / tanh so the physics always gets valid numbers.
        self._raw_kappa = nn.Parameter(
            _inv_softplus(torch.tensor([float(kappa_init)])))
        self._raw_theta = nn.Parameter(
            _inv_softplus(torch.tensor([float(theta_init)])))
        self._raw_xi = nn.Parameter(
            _inv_softplus(torch.tensor([float(xi_init)])))
        # clamp rho_init into (-1,1) before computing atanh
        rho_clamped = max(-0.999, min(0.999, float(rho_init)))
        self._raw_rho = nn.Parameter(
            _inv_tanh(torch.tensor([rho_clamped])))

        # ---------- network architecture (same as HSPINN) ----------
        layers = [nn.Linear(3, neurons_per_layer),
                  nn.Softplus() if activation == 'softplus' else nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Softplus() if activation == 'softplus' else nn.Tanh())
        layers.append(nn.Linear(neurons_per_layer, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ----- constrained properties -----
    @property
    def kappa(self):
        return F.softplus(self._raw_kappa)

    @property
    def theta(self):
        return F.softplus(self._raw_theta)

    @property
    def xi(self):
        return F.softplus(self._raw_xi)

    @property
    def rho(self):
        return torch.tanh(self._raw_rho)

    # ----- helpers -----
    def heston_params(self):
        """Return a dict of the current (constrained) Heston parameters."""
        return {
            'kappa': self.kappa.item(),
            'theta': self.theta.item(),
            'xi':    self.xi.item(),
            'rho':   self.rho.item(),
        }

    def param_groups(self, lr_nn=5e-3, lr_heston=1e-3):
        """
        Return two parameter groups with different learning rates:
          - NN weights/biases  -> lr_nn
          - Heston raw params  -> lr_heston  (typically much smaller)
        """
        heston_names = {'_raw_kappa', '_raw_theta', '_raw_xi', '_raw_rho'}
        nn_params, heston_params = [], []
        for name, p in self.named_parameters():
            if name in heston_names:
                heston_params.append(p)
            else:
                nn_params.append(p)
        return [
            {'params': nn_params,     'lr': lr_nn},
            {'params': heston_params, 'lr': lr_heston},
        ]

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, v, tau):
        S_norm = S / self.S_scale
        v_norm = v / self.v_scale
        tau_norm = tau / self.tau_scale
        x = torch.cat([S_norm, v_norm, tau_norm], dim=1)
        return self.net(x)
