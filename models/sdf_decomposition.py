"""Population SDF plus physical rigid registration.

All public coordinates use the existing centered world/world_bbox convention.
Distances returned in component dictionaries are millimetres. Registration stores
the inverse sampling map psi = phi^-1, from subject space to atlas space.
"""
import copy
import math
import torch
from torch import nn
from models.inr_decoder import INR_Decoder


class SharedAtlasSDF(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.decoder = INR_Decoder(args, device)
        self.grid = nn.Parameter(torch.randn(1, *args['inr_decoder']['latent_dim']) * .01)
        self.tau = float(args['optimizer'].get('sdf_trunc_mm', 5.))
        if self.decoder.sdf_dim != 1 or self.tau <= 0:
            raise ValueError('Decomposition requires one SDF channel and positive truncation')

    def forward(self, coords, conditions):
        # Crucially: there is no subject index or subject modulation grid here.
        indices = torch.zeros(len(coords), dtype=torch.long, device=coords.device)
        return self.decoder(coords, self.grid, conditions, idcs_df=indices)


    def sdf_mm(self, coords, conditions):
        output, _ = self(coords, conditions)
        return output[:, self.decoder.seg_end] * self.tau


class SubjectRegistration(nn.Module):
    """Six Euler/translation parameters for psi(x)=R*x+t in centered mm.

    This is a rigid finite-dimensional transform, not a learned deformation.
    The interface exposes both maps for a future registration implementation.
    Translation is in mm, angles in radians; inverse is analytic and exact.
    """
    def __init__(self, world_bbox, rotation='legacy_euler'):
        super().__init__()
        if rotation not in ('legacy_euler', 'rotvec'):
            raise ValueError('rotation must be legacy_euler or rotvec')
        self.rotation = rotation
        self.register_buffer('half_bbox', torch.as_tensor(world_bbox, dtype=torch.float32) / 2)
        if (self.half_bbox <= 0).any():
            raise ValueError('world_bbox must be positive')

    def subject_to_atlas(self, coords, parameters):
        if self.rotation == 'legacy_euler':
            return INR_Decoder.transform(coords * self.half_bbox, parameters) / self.half_bbox
        matrix = self.rotation_matrix(parameters[:, :3])
        return (torch.einsum('nij,nj->ni', matrix, coords * self.half_bbox) + parameters[:, 3:]) / self.half_bbox

    def atlas_to_subject(self, coords, parameters):
        if self.rotation == 'legacy_euler':
            return INR_Decoder.transform(coords * self.half_bbox, parameters, inverse=True) / self.half_bbox
        matrix = self.rotation_matrix(parameters[:, :3])
        return torch.einsum('nji,nj->ni', matrix, coords * self.half_bbox - parameters[:, 3:]) / self.half_bbox

    @staticmethod
    def rotation_matrix(vector):
        """Rodrigues map with finite derivatives at the identity."""
        x, y, z = vector.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
        angle = torch.linalg.vector_norm(vector, dim=-1)
        a = torch.sinc(angle / math.pi)[:, None, None]
        b = (.5 * torch.sinc(angle / (2 * math.pi)).square())[:, None, None]
        return torch.eye(3, device=vector.device, dtype=vector.dtype) + a * skew + b * (skew @ skew)


class AtlasSubjectSDF(nn.Module):
    def __init__(self, args, subject_ids, device='cpu'):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.cfg = args.get('decomposition', {})
        obsolete = {'residual_enabled', 'latent_enabled', 'alpha', 'lr_residual', 'lr_latent', 'fourier_bands'} & self.cfg.keys()
        if obsolete:
            raise ValueError(f'Residual-era configuration is retired: {sorted(obsolete)}. Prepare a fresh atlas-rigid v2 study.')
        self.subject_ids = [str(s) for s in subject_ids]
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError('Subject IDs must be nonempty and unique')
        self.shared = SharedAtlasSDF(args, device)
        self.registration = SubjectRegistration(args['dataset']['world_bbox'], self.cfg.get('rotation', 'legacy_euler'))
        self.transforms = nn.Embedding(len(subject_ids), 6)
        nn.init.zeros_(self.transforms.weight)
        if self.cfg.get('registration_enabled', True) and args['inr_decoder'].get('tf_dim', 6) not in (0, 6):
            raise ValueError('Decomposition supports physical rigid registration only (tf_dim 6); affine baseline remains available')
        self.stage = 'C'

    def set_stage(self, epoch):
        # Residual-only freezing stages no longer apply. Train atlas and rigid jointly.
        self.stage = 'C'
        for p in self.shared.parameters():
            p.requires_grad_(True)
        self.transforms.weight.requires_grad_(self.cfg.get('registration_enabled', True))
        return self.stage

    def forward(self, coords, conditions, subject_indices=None, mode='subject_reconstruction',
                transforms=None):
        if mode not in ('atlas_only', 'subject_reconstruction', 'atlas_and_components'):
            raise ValueError(f'Unknown inference mode: {mode}')
        atlas_only = mode == 'atlas_only'
        if not atlas_only and transforms is None:
            if subject_indices is None:
                raise ValueError('Subject indices or explicit transforms required')
            transforms = self.transforms(subject_indices)
        warped = coords
        if not atlas_only and self.cfg.get('registration_enabled', True):
            warped = self.registration.subject_to_atlas(coords, transforms)
        output, moe = self.shared(warped, conditions)
        atlas_mm = output[:, self.shared.decoder.seg_end] * self.shared.tau
        return {'sdf_mm': atlas_mm, 'warped_atlas_mm': atlas_mm,
                'output': output, 'moe': moe,
                'registration_penalty': (transforms[:, :3].square().mean(-1) +
                    (transforms[:, 3:] / self.registration.half_bbox).square().mean(-1))
                    if not atlas_only else torch.zeros_like(atlas_mm)}

    def initialize_baseline(self, checkpoint):
        """Initialize decoder and population grid; deliberately discard subject identity.

        Baseline normalized rotations cannot generally be converted to physical rigid
        rotations on an anisotropic box, so registrations start at identity.
        """
        state = {k.removeprefix('module.'): v for k, v in checkpoint['inr_decoder'].items()}
        self.shared.decoder.load_state_dict(state, strict=True)
        if 'latents' in checkpoint:
            with torch.no_grad():
                self.shared.grid.copy_(checkpoint['latents'].to(self.shared.grid).mean(0, keepdim=True))


def sdf_to_mask(sdf_mm):
    """Repository convention: negative inside; zero belongs to the exterior."""
    return sdf_mm < 0
