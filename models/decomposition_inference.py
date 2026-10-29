"""Unseen-subject rigid fitting with all population network weights frozen."""
import torch
from torch import nn


def adapt_subject(model, coords, conditions, targets_mm, steps=100, lr=.01,
                  registration_weight=1e-3):
    if len(coords) == 0:
        raise ValueError('Adaptation requires observed SDF samples')
    states = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    transform = nn.Parameter(torch.zeros(1, 6, device=coords.device))
    parameters = []
    if model.cfg.get('registration_enabled', True):
        parameters.append(transform)
    history = []
    try:
        if not parameters:
            return {'transform': transform.detach(), 'history': history}
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            output = model(coords, conditions, transforms=transform.expand(len(coords), -1))
            tau = model.shared.tau
            loss = (output['sdf_mm'] - targets_mm.clamp(-tau, tau)).abs().mean() / tau
            loss = loss + registration_weight * (transform[:, :3].square().mean() +
                    (transform[:, 3:] / model.registration.half_bbox).square().mean())
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
    finally:
        for p, state in zip(model.parameters(), states):
            p.requires_grad_(state)
        model.train(was_training)
    return {'transform': transform.detach(), 'history': history}
