"""Explicit reconstruction objectives; defaults preserve the original pilot."""
import torch
import torch.nn.functional as F


def occupancy_loss(logits, labels, negative_weight=1.0):
    """Mean weighted BCE; weight >1 penalizes predicted background occupancy.

    This changes the training objective, not the meaning of pseudo-radius targets.
    Threshold calibration is performed independently on validation people only.
    """
    if negative_weight <= 0 or not torch.isfinite(torch.tensor(negative_weight)):
        raise ValueError('negative_weight must be positive and finite')
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    weights = torch.where(labels > .5, 1., float(negative_weight))
    return (loss * weights).mean()
