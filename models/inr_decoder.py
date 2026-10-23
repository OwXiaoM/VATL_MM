import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.siren import Siren
from utils import embed2affine
from scipy.ndimage import label
import scipy.ndimage as ndi


class INR_Decoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        args_inr = args["inr_decoder"]
        self.device = device
        self.sr_dims = sum(args_inr["out_dim"][:-1])
        self.seg_dims = args_inr["out_dim"][-1]
        self.seg_end = self.sr_dims + self.seg_dims
        self.sdf_dim = args_inr.get("sdf_dim", 0)
        if self.sdf_dim not in (0, 1):
            raise ValueError("inr_decoder.sdf_dim must be 0 or 1")
        self.out_dim = self.seg_end + self.sdf_dim
        self.mod_dim = args_inr["hidden_size"] * len(args_inr["modulated_layers"]) * 2
        self.modulator = Modulator(
            args_inr["latent_dim"], kernel_size=args_inr["cnn_kernel_size"]
        )

        use_moe = args_inr.get("use_moe", False)
        num_experts = args_inr.get("num_experts", 4)
        moe_k = args_inr.get("moe_k", 2)

        self.sr_net = Siren(
            in_size=args_inr["in_dim"],
            lat_size=args_inr["latent_dim"][0] + args_inr["cond_dims"],
            out_size=self.out_dim,
            hidden_size=args_inr["hidden_size"],
            num_layers=args_inr["num_hidden_layers"],
            f_om=args_inr["omega"][0],
            h_om=args_inr["omega"][1],
            outermost_linear=True,
            modulated_layers=args_inr["modulated_layers"],
            use_moe=use_moe,
            num_experts=num_experts,
            moe_k=moe_k,
        )

    def forward(self, coords, latent_vecs, condition_vecs, tfs=None, idcs_df=None):
        """
        Args:
            coords: (N, 3)
            latent_vecs: (N, l_channels, l_x, l_y, l_z)
            condition_vecs: (N, c_channels)
            tfs: (N, 6)
        Returns:
            output: (N, out_dim), ordered as images, tissue logits, optional SDF.
                SDF is linear and unbounded, with no intensity clamp or softmax.
            aux_loss: scalar tensor (MoE load balancing loss)
        """
        coords = self.transform(coords, tfs) if tfs is not None else coords
        modulations = self.modulator(latent_vecs)[idcs_df]
        modulations_interp = self.spatial_interpolation(
            coords, modulations, condition_vecs
        )

        output = self.sr_net((coords, modulations_interp))

        aux_loss = self.sr_net.get_moe_loss()

        return output, aux_loss

    def inference(
        self,
        coords,
        latent_vec,
        condition_vec,
        img_shape,
        tfs=None,
        step_size=100000,
        return_sdf=False,
        include_sdf=False,
    ):
        """
        Inference of the INR decoder for volume generation.

        The default return retains images/segmentation/tissue probabilities.
        include_sdf=True appends [SDF_mm, VesselMask] after those channels.
        SDF_mm = raw_sdf * optimizer.sdf_trunc_mm; VesselMask = SDF_mm < 0.
        Neither SDF nor its vessel mask is modified by reconstruction masking.
        return_sdf=True additionally returns (reconstruction, raw_sdf), where
        raw_sdf has shape (*img_shape, 1) and retains normalized training units.
        """
        if (return_sdf or include_sdf) and not self.sdf_dim:
            raise ValueError("SDF output is disabled; set inr_decoder.sdf_dim to 1")

        output = torch.empty((coords.shape[0], self.out_dim)).to(device=self.device)

        coords = (
            self.transform(coords, tfs.expand(coords.shape[0], -1))
            if tfs is not None
            else coords
        )

        for i in range(0, coords.shape[0], step_size):
            c = coords[i : i + step_size]

            idcs_df = torch.zeros(c.shape[0], dtype=torch.long, device=self.device)

            cv = condition_vec.expand(c.shape[0], -1)

            batch_out, _ = self.forward(c, latent_vec, cv, idcs_df=idcs_df)

            output[i : i + step_size] = batch_out

        raw_imgs = output[..., : self.sr_dims]

        imgs = torch.clamp(raw_imgs, 0, 1)

        imgs = imgs.reshape(img_shape + [-1])

        seg_hard = torch.argmax(
            output[..., self.sr_dims : self.seg_end], dim=-1
        ).reshape(img_shape + [-1])
        seg_soft = torch.nn.functional.softmax(
            output[..., self.sr_dims : self.seg_end], dim=-1
        ).reshape(img_shape + [-1])

        modalities_rec = torch.cat((imgs, seg_hard, seg_soft), dim=-1)

        if self.args["mask_reconstruction"]:
            modalities_rec = self.mask_reconstruction(modalities_rec, seg_hard)
        if return_sdf or include_sdf:
            sdf = output[..., self.seg_end :].reshape(list(img_shape) + [self.sdf_dim])
        # Append SDF after reconstruction masking so distances and vessel branches are retained.
        if include_sdf:
            tau_mm = float(self.args["optimizer"].get("sdf_trunc_mm", 5.0))
            if not np.isfinite(tau_mm) or tau_mm <= 0:
                raise ValueError("optimizer.sdf_trunc_mm must be finite and positive")
            sdf_mm = sdf * tau_mm
            vessel_mask = (sdf_mm < 0).to(sdf_mm.dtype)
            modalities_rec = torch.cat((modalities_rec, sdf_mm, vessel_mask), dim=-1)
        if return_sdf:
            return modalities_rec, sdf
        return modalities_rec

    @staticmethod
    def transform(coords, tfs, inverse=False):
        R, t = embed2affine(tfs)
        if inverse:
            R = R.inverse()
            t = -torch.einsum("nij,nj->ni", R, t)
        coords = torch.einsum("nxy,ny->nx", R, coords) + t
        return coords

    @staticmethod
    def spatial_interpolation(coords, latents, condition_vecs=None):
        """Interpolate latent grids at normalized 3D coordinates.

        coords: (N, 3); latents: (N, C, X, Y, Z). Optional conditions are
        concatenated to the interpolated features.
        """
        coords = coords[:, None, None, None, :]
        latents = F.grid_sample(
            latents, coords, mode="bilinear", align_corners=True, padding_mode="border"
        ).squeeze()
        if condition_vecs is not None:
            latents = torch.concat((latents, condition_vecs), dim=-1)
        return latents

    def mask_reconstruction(self, recs, seg):
        mask = self.connected_components(seg)
        mask = mask.expand_as(recs)
        return recs * mask

    def connected_components(self, seg, bg_label_str="Background"):
        bg_label = self.args["dataset"]["label_names"].index(bg_label_str)
        mask = ((seg > 0) & (seg != bg_label)).detach().cpu().numpy()
        shp = np.array(list(mask.shape[:-1]))
        ps = ((shp * 0.1) // 2).astype(int)
        cp = shp // 2
        labeled_mask, num_labels = label(mask)
        center_label = labeled_mask[
            cp[0] - ps[0] : cp[0] + ps[0],
            cp[1] - ps[1] : cp[1] + ps[1],
            cp[2] - ps[2] : cp[2] + ps[2],
        ].flatten()
        majority_label = np.argmax(np.bincount(center_label))
        mask = mask * (labeled_mask == majority_label)
        mask_blur = (
            ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0) > 0.001
        ).astype(np.uint8)

        return torch.from_numpy(mask_blur).to(self.device, torch.float)


class Modulator(nn.Module):
    """Produce spatial modulation grids from latent tensors (C, X, Y, Z)."""

    def __init__(self, latent_dims, kernel_size=3):
        super().__init__()
        if kernel_size > 0:
            self.conv = nn.Conv3d(
                latent_dims[0], latent_dims[0], kernel_size, padding="same"
            )
        else:
            self.conv = nn.Identity()

    def forward(self, latent_vecs):
        return self.conv(latent_vecs)


class LatentRegressor(nn.Module):
    """
    Latent regressor to regress information from the latent vector, e.g., birth_age.
    """

    def __init__(self, latent_dims):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dims[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent_vecs):
        return self.sequence(latent_vecs)
