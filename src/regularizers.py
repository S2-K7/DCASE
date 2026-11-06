"""Nuclear-norm and spectral-norm regularisers.

NOTE (BatchNorm caveat): a conv/linear layer followed by BatchNorm is scale-invariant, so a
penalty on its weight norm can be reduced by rescaling weights without changing the function.
That makes weight-space penalties on such layers act like weight decay. Hence the default
scope is 'linear' (embedding + classifier layers, which are not followed by BN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def nuclear_norm_penalty(emb, center=True):
    """Nuclear norm of the (batch x dim) embedding matrix, divided by sqrt(batch size) so the
    value (and therefore lambda) is roughly invariant to batch size."""
    x = emb.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    return s.sum() / (x.shape[0] ** 0.5)


def _target_layers(model, scope):
    types = (nn.Linear,) if scope == "linear" else (nn.Linear, nn.Conv2d)
    return [m for m in model.modules() if isinstance(m, types)]


def nuclear_norm_weights(model, scope="linear"):
    """Mean over layers of (nuclear norm of reshaped weight) / sqrt(rows * cols)."""
    vals = []
    for m in _target_layers(model, scope):
        w = m.weight.reshape(m.weight.shape[0], -1)
        vals.append(torch.linalg.svdvals(w).sum() / (w.shape[0] * w.shape[1]) ** 0.5)
    return torch.stack(vals).mean()


class SpectralNormPenalty:
    """Mean spectral norm (largest singular value, via power iteration) over target layers."""

    def __init__(self, model, scope="linear", n_iter=1):
        self.layers = _target_layers(model, scope)
        self.n_iter = n_iter
        self.u = {}

    def __call__(self):
        total = 0.0
        for m in self.layers:
            W = m.weight.reshape(m.weight.shape[0], -1)
            key = id(m)
            u = self.u.get(key)
            if u is None:
                u = F.normalize(torch.randn(W.shape[0], device=W.device), dim=0)
            with torch.no_grad():
                for _ in range(self.n_iter):
                    v = F.normalize(W.t() @ u, dim=0)
                    u = F.normalize(W @ v, dim=0)
                v = F.normalize(W.t() @ u, dim=0)
                self.u[key] = u
            total = total + u @ (W @ v)  # gradient flows through W only
        return total / len(self.layers)


def apply_spectral_norm(model, scope="linear"):
    """Hard constraint: wrap target layers with torch's spectral-norm parametrisation."""
    from torch.nn.utils.parametrizations import spectral_norm
    for m in _target_layers(model, scope):
        spectral_norm(m)
    return model
