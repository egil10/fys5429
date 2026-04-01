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
                   lambda_pde, lambda_ic, lambda_bc, lambda_data, 
                   epochs_adam=25000,          # PHASE 1
                   epochs_lbfgs=5000,          # PHASE 2
                   lr=5e-3, lr_heston=1e-3,
                   lambda_feller=1.0,
                   hidden_layers=3, neurons=256, activation='siren', # Takes siren/gelu natively now!
                   kappa_init=1.0, theta_init=0.1, xi_init=0.5, rho_init=0.0):

    model = INVPINN(hidden_layers, neurons, activation=activation,
                    kappa_init=kappa_init, theta_init=theta_init,
                    xi_init=xi_init, rho_init=rho_init).to(device)

    # ---------------- OPTIMIZERS ----------------
    optimizer_adam = optim.Adam(model.param_groups(lr_nn=lr, lr_heston=lr_heston))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=epochs_adam, eta_min=1e-6)
    
    # L-BFGS uses a strong wolfe line search to find the exact mathematical bottom of the basin
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=1, history_size=50, 
                                  tolerance_grad=1e-5, tolerance_change=1e-9, 
                                  line_search_fn="strong_wolfe")

    grad_ones = torch.ones_like(S_in)
    history = {
        'epoch': [], 'pde': [], 'ic': [], 'bc': [], 'data': [], 'feller': [], 'total': [],
        'kappa': [], 'theta': [], 'xi': [], 'rho': []
    }
    total_epochs = epochs_adam + epochs_lbfgs
    sample_interval = max(1, total_epochs // 100)

    # =========================================================================
    # ENCAPSULATED LOSS FUNCTION (Required for L-BFGS closures)
    # =========================================================================
    def compute_loss():
        kappa, theta, xi, rho = model.kappa, model.theta, model.xi, model.rho

        # PDE residual 
        V_pred = model(S_in, v_in, tau_in)
        V_S   = torch.autograd.grad(V_pred, S_in,   grad_outputs=grad_ones, create_graph=True)[0]
        V_v   = torch.autograd.grad(V_pred, v_in,   grad_outputs=grad_ones, create_graph=True)[0]
        V_tau = torch.autograd.grad(V_pred, tau_in, grad_outputs=grad_ones, create_graph=True)[0]
        V_SS  = torch.autograd.grad(V_S, S_in,      grad_outputs=grad_ones, create_graph=True)[0]
        V_vv  = torch.autograd.grad(V_v, v_in,      grad_outputs=grad_ones, create_graph=True)[0]
        V_Sv  = torch.autograd.grad(V_S, v_in,      grad_outputs=grad_ones, create_graph=True)[0]

        pde_residual = V_tau - (
            0.5 * v_in * S_in**2 * V_SS + rho * xi * v_in * S_in * V_Sv + 0.5 * xi**2 * v_in * V_vv
            + r * S_in * V_S + kappa * (theta - v_in) * V_v - r * V_pred
        )
        loss_pde = torch.mean(pde_residual**2)

        # IC / BC / Data Losses
        loss_ic = torch.mean((model(S_ic, v_ic, tau_ic) - torch.relu(S_ic - K))**2)
        loss_bc = torch.mean((model(S_bc, v_bc, tau_bc) - 0.0)**2)
        loss_data = torch.mean((model(S_data, v_data, tau_data) - V_data)**2)

        # Feller
        loss_feller = torch.relu(xi**2 - 2.0 * kappa * theta)**2

        loss = (lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc 
                + lambda_data * loss_data + lambda_feller * loss_feller)
                
        return loss, loss_pde, loss_ic, loss_bc, loss_data, loss_feller, kappa, theta, xi, rho

    def log_history(epoch_num):
        with torch.no_grad():
            loss, lp, lic, lbc, ldata, lf, k, t, x, r_corr = compute_loss()
            history['epoch'].append(epoch_num)
            history['pde'].append(lp.item()); history['ic'].append(lic.item())
            history['bc'].append(lbc.item()); history['data'].append(ldata.item())
            history['feller'].append(lf.item()); history['total'].append(loss.item())
            history['kappa'].append(k.item()); history['theta'].append(t.item())
            history['xi'].append(x.item()); history['rho'].append(r_corr.item())

    # =========================================================================
    # PHASE 1: ADAM (Broad Basin Search)
    # =========================================================================
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        loss, *_ = compute_loss()
        loss.backward()
        
        # Clip Gradients for stability ONLY in Phase 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step()

        if epoch % sample_interval == 0: log_history(epoch)

    # =========================================================================
    # PHASE 2: L-BFGS (Precision Fine-Tuning)
    # =========================================================================
    for epoch in range(epochs_adam, total_epochs):
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, *_ = compute_loss()
            loss.backward()
            return loss # Note: No clip_grad_norm_ allowed here; it breaks L-BFGS line-search math!

        optimizer_lbfgs.step(closure)

        if epoch % sample_interval == 0 or epoch == total_epochs - 1:
            log_history(epoch)

    # Grab final state to return
    with torch.no_grad():
        loss, lp, lic, lbc, ldata, lf, k, t, x, r_corr = compute_loss()

    return {
        'model': model,
        'history': history,
        'final_pde': lp.item(), 'final_ic': lic.item(), 'final_bc': lbc.item(),
        'final_data': ldata.item(), 'final_feller': lf.item(), 'final_total': loss.item(),
        'final_kappa': k.item(), 'final_theta': t.item(), 'final_xi': x.item(), 'final_rho': r_corr.item(),
    }
