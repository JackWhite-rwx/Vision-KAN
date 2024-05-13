import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

from functools import partial
from einops.layers.torch import Rearrange, Reduce


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

# 直接加有点问题，建议把这个改成可学习权重相加
class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))

def Permutator(*, image_size, patch_size, input_dim, dim, depth, num_classes, segments, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
    height = width = image_size // patch_size
    s = segments# channel group segments 这个参数把channel分割为原来的几倍

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                ParallelSum(
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b w c (h s)', s = s),
                        nn.Linear(height * s, height * s),
                        Rearrange('b w c (h s) -> b h w (c s)', s = s),
                    ),
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b h c (w s)', s = s),
                        nn.Linear(width * s, width * s),
                        Rearrange('b h c (w s) -> b h w (c s)', s = s),
                    ),
                    nn.Linear(dim, dim)
                ),
                nn.Linear(dim, dim)
            )),
            # PreNormResidual(dim, nn.Sequential(
            #     nn.Linear(dim, dim * expansion_factor),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(dim * expansion_factor, dim),
            #     nn.Dropout(dropout)
            # ))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b h w c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

def KanPermutator(*, image_size, patch_size, input_dim, dim, depth, segments, num_classes):# expansion_factor = 4, dropout = 0.
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
    height = width = image_size // patch_size
    s = segments# channel group segments 这个参数把channel分割为原来的几倍

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        Rearrange('b h w (p1 p2 c) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        FastKANLayer((patch_size ** 2) * input_dim, dim),#在channel维度做隐式函数嵌入
        Rearrange('b (h w) (p1 p2 c) -> b h w (p1 p2 c)', p1=patch_size, p2=patch_size, h=height, w=width),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                ParallelSum(
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b (w c) (h s)', s = s),
                        FastKANLayer(height * s, height * s),
                        Rearrange('b (w c) (h s) -> b h w (c s)', s = s, h=height, w=width),
                    ),
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b (h c) (w s)', s = s),
                        FastKANLayer(width * s, width * s),
                        Rearrange('b (h c) (w s) -> b h w (c s)', s = s, h=height, w=width),
                    ),
                    nn.Sequential(
                        Rearrange('b h w c -> b (h w) c'),
                        FastKANLayer(dim, dim),
                        Rearrange('b (h w) c -> b h w c', h=height, w=width),
                    )
                ),
                nn.Sequential(
                    Rearrange('b h w c -> b (h w) c'),
                    FastKANLayer(dim, dim),
                    Rearrange('b (h w) c -> b h w c', h=height, w=width),
                )
            )),
            # PreNormResidual(dim, nn.Sequential(
            #     FastKAN([dim, dim * expansion_factor]),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            #     FastKAN([dim * expansion_factor, dim]),
            #     nn.Dropout(dropout)
            # ))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b h w c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
        FastKANLayer(dim, num_classes),
        # Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size)
    )



#上采样 nn.PixelShuffle

if __name__ == "__main__":
    model = Permutator(image_size=64, patch_size=4, input_dim=3, dim=48, depth=4, segments=2, num_classes=10).cuda()# patch_size*patch_size*input_dim==dim，时，输入输出shape一致，否则dim最好是其倍数
    y = model(torch.rand(4,3,64,64).cuda())
    print(y.shape)
