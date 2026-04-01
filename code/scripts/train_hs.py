import torch
import torch.nn as nn
import torch.optim as optim
from hspinn import HSPINN


def train_pinn(S_in, v_in, tau_in,
               S_ic, v_ic, tau_ic,
               S_bc, v_bc, tau_bc,
               r, K, kappa, theta, xi, rho,
               device,
               lambda_pde, lambda_ic, lambda_bc, epochs,
               lr=0.001, hidden_layers=4, neurons=256, activation='tanh'):

    model = HSPINN(hidden_layers, neurons, activation=activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    grad_ones = torch.ones_like(S_in)
    history = {'epoch': [], 'pde': [], 'ic': [], 'bc': [], 'total': []}
    sample_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- PDE residual ---
        V_pred = model(S_in, v_in, tau_in)

        # First-order derivatives
        V_S   = torch.autograd.grad(V_pred, S_in,   grad_outputs=grad_ones, create_graph=True)[0]
        V_v   = torch.autograd.grad(V_pred, v_in,   grad_outputs=grad_ones, create_graph=True)[0]
        V_tau = torch.autograd.grad(V_pred, tau_in,  grad_outputs=grad_ones, create_graph=True)[0]

        # Second-order derivatives
        V_SS  = torch.autograd.grad(V_S, S_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_vv  = torch.autograd.grad(V_v, v_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_Sv  = torch.autograd.grad(V_S, v_in, grad_outputs=grad_ones, create_graph=True)[0]

        # Heston PDE: V_tau = 0.5*v*S^2*V_SS + rho*xi*v*S*V_Sv + 0.5*xi^2*v*V_vv
        #                     + r*S*V_S + kappa*(theta-v)*V_v - r*V
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

        # --- Total loss ---
        loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % sample_interval == 0 or epoch == epochs - 1:
            history['epoch'].append(epoch)
            history['pde'].append(loss_pde.item())
            history['ic'].append(loss_ic.item())
            history['bc'].append(loss_bc.item())
            history['total'].append(loss.item())

        if epoch > 500 and epoch % sample_interval == 0:
            if len(history['total']) >= 2 and abs(history['total'][-1] - history['total'][-2]) / (history['total'][-2] + 1e-12) < 1e-5:
                break

    return {
        'model': model,
        'history': history,
        'final_pde': loss_pde.item(),
        'final_ic': loss_ic.item(),
        'final_bc': loss_bc.item(),
        'final_total': loss.item()
    }
