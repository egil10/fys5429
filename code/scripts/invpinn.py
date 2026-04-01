import torch
import torch.nn as nn


class InvPINN(nn.Module):
    def __init__(self, hidden_layers=3, neurons_per_layer=256,
                 S_scale=300.0, v_scale=1.0, tau_scale=1.0, activation='tanh',
                 kappa_init=1.0, theta_init=0.1, xi_init=0.5, rho_init=0.0):
        super().__init__()

        self.S_scale = S_scale
        self.v_scale = v_scale
        self.tau_scale = tau_scale

        # Trainable Heston parameters (intentionally wrong initial guesses)
        self.kappa = nn.Parameter(torch.tensor([kappa_init]))
        self.theta = nn.Parameter(torch.tensor([theta_init]))
        self.xi    = nn.Parameter(torch.tensor([xi_init]))
        self.rho   = nn.Parameter(torch.tensor([rho_init]))

        # Network architecture (same as HSPINN)
        layers = [nn.Linear(3, neurons_per_layer),
                  nn.Softplus() if activation == 'softplus' else nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Softplus() if activation == 'softplus' else nn.Tanh())
        layers.append(nn.Linear(neurons_per_layer, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

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
