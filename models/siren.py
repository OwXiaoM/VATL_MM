import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    def __init__(
        self, in_feat, lat_feat, out_feat, bias=True, is_first=False, omega=30
    ):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_feat
        self.out_features = out_feat
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.linear_lats = (
            nn.Linear(lat_feat, out_feat * 2, bias=bias) if lat_feat > 0 else None
        )
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega,
                    np.sqrt(6 / self.in_features) / self.omega,
                )

    def forward(self, input):
        """
        input: (coords, latent_vec)
        """
        intermed = self.linear(input[0])
        if self.linear_lats is not None:
            lats = self.linear_lats(input[1])
            out = torch.sin(
                (self.omega * intermed * lats[..., : self.out_features])
                + lats[..., self.out_features :]
            )
        else:
            out = torch.sin(self.omega * intermed)
        return out, input[1]


class MoESineLayer(nn.Module):
    def __init__(self, in_feat, lat_feat, out_feat, num_experts=4, k=2, omega=30):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(in_feat, num_experts)

        self.experts = nn.ModuleList(
            [
                SineLayer(in_feat, lat_feat, out_feat, is_first=False, omega=omega)
                for _ in range(num_experts)
            ]
        )
        self.latest_loss = 0.0

    def forward(self, input_tuple):
        """
        input_tuple: (coords, latent_vec)
        """
        x, latents = input_tuple

        gate_logits = self.gate(x)

        weights, indices = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        soft_gates = F.softmax(gate_logits, dim=-1)
        importance = soft_gates.sum(0)
        # Penalize unequal expert importance with a differentiable balancing term.
        target = x.shape[0] / self.num_experts
        loss = torch.sum((importance - target) ** 2) / (target**2)
        self.latest_loss = loss

        final_output = torch.zeros(
            x.shape[0], self.experts[0].out_features, device=x.device
        )

        # Accumulate top-k expert outputs, weighted by routing probabilities.
        for i in range(self.k):
            idx = indices[:, i]
            w = weights[:, i].unsqueeze(1)

            for e_idx in range(self.num_experts):
                mask = idx == e_idx
                if mask.any():
                    masked_x = x[mask]
                    masked_latents = latents[mask] if latents is not None else None

                    expert_out, _ = self.experts[e_idx]((masked_x, masked_latents))

                    final_output[mask] += w[mask] * expert_out

        return final_output, latents


class Siren(nn.Module):
    def __init__(
        self,
        in_size,
        lat_size,
        out_size,
        hidden_size,
        num_layers,
        f_om,
        h_om,
        outermost_linear,
        modulated_layers,
        use_moe=False,
        num_experts=4,
        moe_k=2,
    ):
        super().__init__()
        l_in_mod = 0 in modulated_layers

        self.net = [
            SineLayer(
                in_size, lat_size * l_in_mod, hidden_size, is_first=True, omega=f_om
            )
        ]
        self.hidden_size = hidden_size

        for i in range(num_layers):
            l_in_mod = (i + 1) in modulated_layers

            is_moe_layer = use_moe and (i > 0 and i < num_layers - 1)

            if is_moe_layer:
                self.net.append(
                    MoESineLayer(
                        hidden_size,
                        lat_size * l_in_mod,
                        hidden_size,
                        num_experts=num_experts,
                        k=moe_k,
                        omega=h_om,
                    )
                )
            else:
                self.net.append(
                    SineLayer(
                        hidden_size,
                        lat_size * l_in_mod,
                        hidden_size,
                        is_first=False,
                        omega=h_om,
                    )
                )

        if outermost_linear:
            self.final_linear = nn.Linear(hidden_size + lat_size, out_size, bias=True)
            with torch.no_grad():
                self.final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_size) / h_om, np.sqrt(6 / hidden_size) / h_om
                )
        else:
            self.final_linear = SineLayer(
                hidden_size, 0, out_size, is_first=False, omega=h_om
            )
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        return self.final_linear(torch.cat([x[0], x[1]], dim=-1))

    def get_moe_loss(self):
        loss = 0.0
        count = 0
        for module in self.net:
            if isinstance(module, MoESineLayer):
                loss += module.latest_loss
                count += 1
        return loss / max(count, 1)
