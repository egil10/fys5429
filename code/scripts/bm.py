
import numpy as np

def bm(S0, mu, sigma, T, n_steps, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    dt = T / n_steps
    S = np.empty(n_steps + 1)
    S[0] = S0
    for i in range(n_steps):
        z = rng.standard_normal()
        dW = np.sqrt(dt) * z
        S[i+1] = S[i] + mu*S[i]*dt + sigma*S[i]*dW
    return S

