"""SpecAugment, mixup and soft-target cross-entropy (label smoothing + mixup in one loss)."""
import random

import torch
import torch.nn.functional as F


def spec_augment(x, freq_mask=16, time_mask=32, n_masks=2):
    """x: (B, C, F, T). Same mask for all channels of a clip. Masks are filled with 0
    (= the mean after normalisation)."""
    out = x.clone()
    B, _, n_f, n_t = x.shape
    for b in range(B):
        for _ in range(n_masks):
            f = random.randint(0, min(freq_mask, n_f))
            f0 = random.randint(0, max(0, n_f - f))
            out[b, :, f0:f0 + f, :] = 0
            t = random.randint(0, min(time_mask, n_t))
            t0 = random.randint(0, max(0, n_t - t))
            out[b, :, :, t0:t0 + t] = 0
    return out


def smooth_onehot(y, num_classes, eps=0.0):
    oh = F.one_hot(y, num_classes).float()
    return oh * (1 - eps) + eps / num_classes


def mixup(x, y_soft, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_soft + (1 - lam) * y_soft[perm]


def soft_cross_entropy(logits, y_soft):
    return -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
