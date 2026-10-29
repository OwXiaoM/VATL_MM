"""Losses in physical mm; finite differences avoid grid_sample double backward."""
import torch
import torch.nn.functional as F


def difference_samples(coords, conditions, indices, sdf_mm, half_bbox, cfg):
    """Return six offset clouds and their base indices, restricted to a surface band."""
    candidates = torch.where(sdf_mm.abs() < float(cfg.get('gradient_band_mm', 2.)))[0]
    n = min(len(candidates), int(cfg.get('gradient_points', 256)))
    chosen = candidates[torch.randperm(len(candidates), device=coords.device)[:n]]
    eps = float(cfg.get('difference_step_mm', .25))
    if eps <= 0:
        raise ValueError('difference_step_mm must be positive')
    if n == 0:
        return None
    offsets = torch.eye(3, device=coords.device) * eps / half_bbox
    offset_clouds = [coords[chosen] + sign * offsets[axis] for axis in range(3) for sign in (1, -1)]
    return torch.cat(offset_clouds), conditions[chosen].repeat(6, 1), indices[chosen].repeat(6), chosen


def decomposition_losses(model, components, target_mm, indices, cfg, differences=None):
    pred = components['sdf_mm']
    tau = float(cfg.get('trunc_mm', model.shared.tau))
    if tau <= 0 or not torch.isfinite(target_mm).all():
        raise ValueError('Positive truncation and finite SDF targets required')
    target = target_mm.clamp(-tau, tau)
    # Do not clamp prediction: saturated wrong predictions must retain gradients.
    near = torch.exp(-target_mm.abs() / float(cfg.get('surface_band_mm', 1.)))
    raw = {'sdf': ((pred - target).abs() / tau * (1 + float(cfg.get('near_surface_boost', 1.)) * near)).mean()}
    band = target_mm.abs() <= float(cfg.get('surface_band_mm', 1.))
    raw['surface'] = (pred[band] - target_mm[band]).abs().mean() if band.any() else pred.sum() * 0
    raw['registration'] = components['registration_penalty'].mean()
    raw['eikonal'] = pred.sum() * 0
    if differences is not None:
        eps = float(cfg.get('difference_step_mm', .25))
        def gradient(field):
            clouds = field.reshape(3, 2, -1)
            return ((clouds[:, 0] - clouds[:, 1]) / (2 * eps)).T
        grad = gradient(differences['sdf_mm'])
        raw['eikonal'] = (torch.linalg.vector_norm(grad, dim=-1) - 1).square().mean()
    weights = cfg.get('weights', {})
    weighted = {name: value * float(weights.get(name, 0.)) for name, value in raw.items()}
    return {**{f'{k}_raw': v for k, v in raw.items()}, **weighted, 'total': sum(weighted.values())}
