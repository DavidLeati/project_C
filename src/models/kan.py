import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_pade_grid(
    in_features,
    grid_size,
    spline_order,
    grid_range=(-1.0, 1.0),
    dtype=torch.float32,
):
    h = (grid_range[1] - grid_range[0]) / grid_size
    grid = (
        torch.arange(-spline_order, grid_size + spline_order + 1, dtype=dtype) * h
        + grid_range[0]
    )
    return grid.view(1, -1).expand(in_features, -1).contiguous()


def compute_b_splines(x, grid, spline_order):
    x_uns = x.unsqueeze(-1)
    grid_view = grid
    while grid_view.ndim < x_uns.ndim:
        grid_view = grid_view.unsqueeze(0)

    bases = ((x_uns >= grid_view[..., :-1]) & (x_uns < grid_view[..., 1:])).to(
        dtype=x.dtype
    )
    for order_idx in range(1, spline_order + 1):
        left_term = (
            x_uns - grid_view[..., : -(order_idx + 1)]
        ) / (
            grid_view[..., order_idx:-1]
            - grid_view[..., : -(order_idx + 1)]
            + 1e-8
        )
        right_term = (
            grid_view[..., order_idx + 1 :] - x_uns
        ) / (
            grid_view[..., order_idx + 1 :]
            - grid_view[..., 1:(-order_idx)]
            + 1e-8
        )
        bases = left_term * bases[..., :-1] + right_term * bases[..., 1:]
    return bases


def decode_pade_dna(dna, out_features, in_features, spline_terms):
    component_width = 2 * spline_terms + 5
    expected_dim = out_features * in_features * component_width
    if dna.shape[-1] != expected_dim:
        raise ValueError(
            f"Divergência no tamanho da estrutura do DNA. O tamanho {dna.shape[-1]} difere de {expected_dim}."
        )

    dna_view = dna.view(*dna.shape[:-1], out_features, in_features, component_width)
    dna_chunks = torch.split(
        dna_view,
        [1, spline_terms, spline_terms, 1, 1, 1, 1],
        dim=-1,
    )
    return {
        "base_weight": dna_chunks[0].squeeze(-1),
        "num_spline_weight": dna_chunks[1],
        "den_spline_weight": dna_chunks[2],
        "heaviside_bias": dna_chunks[3].squeeze(-1),
        "heaviside_weight": dna_chunks[4].squeeze(-1),
        "l2_den_weight": dna_chunks[5].squeeze(-1),
        "prod_den_weight": dna_chunks[6].squeeze(-1),
    }


def _broadcast_param_to(delta_tensor, base_tensor):
    while base_tensor.ndim < delta_tensor.ndim:
        base_tensor = base_tensor.unsqueeze(0)
    return base_tensor


def resolve_pade_parameters(
    *,
    base_weight,
    num_spline_weight,
    den_spline_weight,
    heaviside_bias,
    heaviside_weight,
    l2_den_weight,
    prod_den_weight,
    dna=None,
):
    if dna is None:
        return {
            "base_weight": base_weight,
            "num_spline_weight": num_spline_weight,
            "den_spline_weight": den_spline_weight,
            "heaviside_bias": heaviside_bias,
            "heaviside_weight": heaviside_weight,
            "l2_den_weight": l2_den_weight,
            "prod_den_weight": prod_den_weight,
        }

    spline_terms = num_spline_weight.shape[-1]
    decoded = decode_pade_dna(
        dna,
        out_features=base_weight.shape[-2],
        in_features=base_weight.shape[-1],
        spline_terms=spline_terms,
    )
    return {
        name: _broadcast_param_to(delta, tensor) + delta
        for name, tensor, delta in [
            ("base_weight", base_weight, decoded["base_weight"]),
            ("num_spline_weight", num_spline_weight, decoded["num_spline_weight"]),
            ("den_spline_weight", den_spline_weight, decoded["den_spline_weight"]),
            ("heaviside_bias", heaviside_bias, decoded["heaviside_bias"]),
            ("heaviside_weight", heaviside_weight, decoded["heaviside_weight"]),
            ("l2_den_weight", l2_den_weight, decoded["l2_den_weight"]),
            ("prod_den_weight", prod_den_weight, decoded["prod_den_weight"]),
        ]
    }


def pade_kan_forward(
    x,
    *,
    base_weight,
    num_spline_weight,
    den_spline_weight,
    heaviside_bias,
    heaviside_weight,
    l2_den_weight,
    prod_den_weight,
    grid,
    spline_order,
    pade_eps=1e-4,
    heaviside_sharpness=50.0,
    base_activation=None,
    dna=None,
):
    params = resolve_pade_parameters(
        base_weight=base_weight,
        num_spline_weight=num_spline_weight,
        den_spline_weight=den_spline_weight,
        heaviside_bias=heaviside_bias,
        heaviside_weight=heaviside_weight,
        l2_den_weight=l2_den_weight,
        prod_den_weight=prod_den_weight,
        dna=dna,
    )

    activation = F.silu(x) if base_activation is None else base_activation(x)
    base_output = torch.einsum("...i,...oi->...o", activation, params["base_weight"])

    spline_basis = compute_b_splines(x, grid, spline_order)
    num_spline = torch.einsum(
        "...ik,...oik->...oi", spline_basis, params["num_spline_weight"]
    )

    x_diff = x.unsqueeze(-2) - params["heaviside_bias"]
    heaviside_trigger = torch.sigmoid(heaviside_sharpness * x_diff)
    heaviside_out = heaviside_trigger * params["heaviside_weight"]
    numerator = num_spline + heaviside_out

    den_spline = torch.einsum(
        "...ik,...oik->...oi", spline_basis, params["den_spline_weight"]
    )
    l2_norm_sq = torch.sum(x**2, dim=-1, keepdim=True).unsqueeze(-2)
    prod_val = torch.prod(x, dim=-1, keepdim=True).unsqueeze(-2)

    catalog_l2 = l2_norm_sq * params["l2_den_weight"]
    catalog_prod = prod_val * params["prod_den_weight"]

    denominator = 1.0 + den_spline + catalog_l2 + catalog_prod
    denominator_safe = F.softplus(denominator) + pade_eps
    pade_output = torch.sum(numerator / denominator_safe, dim=-1)
    return base_output + pade_output


class PadeKANLayer(nn.Module):
    """
    Camada de Rede Padé Kolmogorov-Arnold.

    Uma camada sofisticada que aplica aproximações racionais de Padé com B-Splines,
    aumentada por catálogos simbólicos explícitos e ativações de Heaviside.
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
        pade_eps=1e-4,
        heaviside_sharpness=50.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()
        self.pade_eps = pade_eps
        self.heaviside_sharpness = heaviside_sharpness

        self.register_buffer(
            "grid",
            build_pade_grid(
                in_features=in_features,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            ),
        )

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.num_spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.den_spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.l2_den_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.prod_den_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.heaviside_bias = nn.Parameter(torch.empty(out_features, in_features))
        self.heaviside_weight = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps

        self.reset_parameters()

    @property
    def dna_dim(self):
        spline_terms = self.grid_size + self.spline_order
        return self.out_features * self.in_features * (2 * spline_terms + 5)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.base_weight.data.mul_(self.scale_base)

        noise_num = (
            (
                torch.rand(
                    self.out_features,
                    self.in_features,
                    self.grid_size + self.spline_order,
                )
                - 0.5
            )
            * self.scale_noise
            / self.grid_size
        )
        self.num_spline_weight.data.copy_(
            self.scale_spline
            * torch.randn(
                self.out_features,
                self.in_features,
                self.grid_size + self.spline_order,
            )
            / math.sqrt(self.grid_size)
            + noise_num
        )

        self.den_spline_weight.data.fill_(0.0)
        self.l2_den_weight.data.fill_(0.0)
        self.prod_den_weight.data.fill_(0.0)
        nn.init.zeros_(self.heaviside_bias)
        nn.init.zeros_(self.heaviside_weight)

    def b_splines(self, x):
        return compute_b_splines(x, self.grid, self.spline_order)

    def forward(self, x, dna=None):
        return pade_kan_forward(
            x,
            base_weight=self.base_weight,
            num_spline_weight=self.num_spline_weight,
            den_spline_weight=self.den_spline_weight,
            heaviside_bias=self.heaviside_bias,
            heaviside_weight=self.heaviside_weight,
            l2_den_weight=self.l2_den_weight,
            prod_den_weight=self.prod_den_weight,
            grid=self.grid,
            spline_order=self.spline_order,
            pade_eps=self.pade_eps,
            heaviside_sharpness=self.heaviside_sharpness,
            base_activation=self.base_activation,
            dna=dna,
        )


class PadeKAN(nn.Module):
    """
    Núcleo da Rede Padé Kolmogorov-Arnold.

    Uma rede sequencial arquitetada iterativamente usando camadas de expressão matemática funcional.
    """

    def __init__(self, layers_hidden, grid_size=5, spline_order=3, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PadeKANLayer(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    **kwargs,
                )
                for in_features, out_features in zip(
                    layers_hidden[:-1], layers_hidden[1:]
                )
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
