import torch
import torch.nn as nn
import torch.optim as optim
from invpinn import INVPINN


def train_inv_pinn(S_in, v_in, tau_in,
                   S_ic, v_ic, tau_ic,
                   S_bc, v_bc, tau_bc,
                   S_data, v_data, tau_data, V_data,
                   r, K,
                   device,
                   lambda_pde, lambda_ic, lambda_bc, lambda_data, epochs,
                   lr=5e-3, lr_heston=1e-3,
                   lambda_feller=1.0,
                   hidden_layers=3, neurons=256, activation='tanh',
                   kappa_init=1.0, theta_init=0.1, xi_init=0.5, rho_init=0.0):

    model = INVPINN(hidden_layers, neurons, activation=activation,
                    kappa_init=kappa_init, theta_init=theta_init,
                    xi_init=xi_init, rho_init=rho_init).to(device)

    # Use separate learning rates: slower for Heston params
    optimizer = optim.Adam(model.param_groups(lr_nn=lr, lr_heston=lr_heston))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    grad_ones = torch.ones_like(S_in)
    history = {
        'epoch': [], 'pde': [], 'ic': [], 'bc': [], 'data': [], 'feller': [], 'total': [],
        'kappa': [], 'theta': [], 'xi': [], 'rho': []
    }
    sample_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Read trainable (constrained) params from model
        kappa = model.kappa
        theta = model.theta
        xi    = model.xi
        rho   = model.rho

        # --- PDE residual (uses trainable params) ---
        V_pred = model(S_in, v_in, tau_in)

        V_S   = torch.autograd.grad(V_pred, S_in,   grad_outputs=grad_ones, create_graph=True)[0]
        V_v   = torch.autograd.grad(V_pred, v_in,   grad_outputs=grad_ones, create_graph=True)[0]
        V_tau = torch.autograd.grad(V_pred, tau_in,  grad_outputs=grad_ones, create_graph=True)[0]

        V_SS  = torch.autograd.grad(V_S, S_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_vv  = torch.autograd.grad(V_v, v_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_Sv  = torch.autograd.grad(V_S, v_in, grad_outputs=grad_ones, create_graph=True)[0]

        pde_residual = V_tau - (
            0.5 * v_in * S_in**2 * V_SS
            + rho * xi * v_in * S_in * V_Sv
            + 0.5 * xi**2 * v_in * V_vv
            + r * S_in * V_S
            + kappa * (theta - v_in) * V_v
            - r * V_pred
        )
        loss_pde = torch.mean(pde_residual**2)

        # --- IC loss (tau=0): payoff = max(S - K, 0) ---
        V_ic_pred = model(S_ic, v_ic, tau_ic)
        V_ic_true = torch.relu(S_ic - K)
        loss_ic = torch.mean((V_ic_pred - V_ic_true)**2)

        # --- BC loss (S=0): V(0, v, tau) = 0 ---
        V_bc_pred = model(S_bc, v_bc, tau_bc)
        loss_bc = torch.mean((V_bc_pred - 0.0)**2)

        # --- Data loss (fit to market observations) ---
        V_data_pred = model(S_data, v_data, tau_data)
        loss_data = torch.mean((V_data_pred - V_data)**2)

        # --- Feller condition penalty: 2*kappa*theta > xi^2 ---
        # Penalise violation of the Feller condition
        feller_violation = torch.relu(xi**2 - 2.0 * kappa * theta)
        loss_feller = feller_violation**2

        # --- Total loss ---
        loss = (lambda_pde * loss_pde
                + lambda_ic * loss_ic
                + lambda_bc * loss_bc
                + lambda_data * loss_data
                + lambda_feller * loss_feller)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % sample_interval == 0 or epoch == epochs - 1:
            history['epoch'].append(epoch)
            history['pde'].append(loss_pde.item())
            history['ic'].append(loss_ic.item())
            history['bc'].append(loss_bc.item())
            history['data'].append(loss_data.item())
            history['feller'].append(loss_feller.item())
            history['total'].append(loss.item())
            history['kappa'].append(kappa.item())
            history['theta'].append(theta.item())
            history['xi'].append(xi.item())
            history['rho'].append(rho.item())

    return {
        'model': model,
        'history': history,
        'final_pde': loss_pde.item(),
        'final_ic': loss_ic.item(),
        'final_bc': loss_bc.item(),
        'final_data': loss_data.item(),
        'final_feller': loss_feller.item(),
        'final_total': loss.item(),
        'final_kappa': kappa.item(),
        'final_theta': theta.item(),
        'final_xi': xi.item(),
        'final_rho': rho.item(),
    }
