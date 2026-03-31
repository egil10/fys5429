import torch
import torch.nn as nn
import torch.optim as optim
from bspinn import BSPINN


def train_pinn(S_in, tau_in, S_ic, tau_ic, S_bc, tau_bc,
               sigma, r, K, device,
               lambda_pde, lambda_ic, lambda_bc, epochs,
               lr=0.001, hidden_layers=3, neurons=128, activation='softplus'):

    model = BSPINN(hidden_layers, neurons, activation=activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    grad_ones = torch.ones_like(S_in)
    history = {'epoch': [], 'pde': [], 'ic': [], 'bc': [], 'total': []}
    sample_interval = max(1, epochs // 100)

    for epoch in range(epochs):
        optimizer.zero_grad()

        V_pred = model(S_in, tau_in)
        V_S = torch.autograd.grad(V_pred, S_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_tau = torch.autograd.grad(V_pred, tau_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_SS = torch.autograd.grad(V_S, S_in, grad_outputs=grad_ones, create_graph=True)[0]

        pde_residual = V_tau - (0.5 * sigma**2 * S_in**2 * V_SS + r * S_in * V_S - r * V_pred)
        loss_pde = torch.mean(pde_residual**2)

        V_ic_pred = model(S_ic, tau_ic)
        V_ic_true = torch.relu(S_ic - K)
        loss_ic = torch.mean((V_ic_pred - V_ic_true)**2)

        V_bc_pred = model(S_bc, tau_bc)
        loss_bc = torch.mean((V_bc_pred - 0.0)**2)

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