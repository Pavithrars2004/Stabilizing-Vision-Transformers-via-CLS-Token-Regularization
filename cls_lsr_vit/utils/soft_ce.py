import torch
import torch.nn as F

def soft_cross_entropy(preds, soft_targets):
    """
    preds: [B, C]
    soft_targets: [B, C]  (from mixup)
    """
    log_probs = F.log_softmax(preds, dim=1)
    return torch.mean(torch.sum(-soft_targets * log_probs, dim=1))
