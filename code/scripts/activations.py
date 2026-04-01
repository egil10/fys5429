import torch
import torch.nn as nn
import math

class Sine(nn.Module):
    """
    Sine activation function for SIREN networks.
    f(x) = sin(w0 * x)
    """
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def get_activation(name):
    """
    Returns the appropriate PyTorch activation function.
    Supports: 'tanh', 'relu', 'softplus', 'gelu', 'silu' (swish), 'siren' (sine)
    """
    name = name.lower()
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'gelu':
        return nn.GELU()
    elif name in ['silu', 'swish']:
        return nn.SiLU()
    elif name in ['siren', 'sine']:
        # SIREN notoriously requires a sine wave multiplier (w0) to hit high frequencies
        return Sine(w0=1.0) 
    else:
        raise ValueError(f"Unknown activation function: {name}. Use 'tanh', 'gelu', 'silu', or 'siren'.")


def init_weights(m, activation_name):
    """
    Applies mathematically proven weight initializations based on the activation.
    - Tanh/GELU/SiLU: Xavier Normal
    - ReLU/Softplus: Kaiming Normal (He Init)
    - SIREN: Sitzmann initialization (crucial for Sine waves)
    """
    if isinstance(m, nn.Linear):
        name = activation_name.lower()
        if name in ['siren', 'sine']:
            # Sitzmann et al. (2020) initialization specifically for SIREN
            in_features = m.weight.shape[1]
            w_std = math.sqrt(6.0 / in_features) / 1.0  # w0 = 1.0
            nn.init.uniform_(m.weight, -w_std, w_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif name in ['relu', 'softplus']:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else: 
            # Default for Tanh, GELU, SiLU
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
