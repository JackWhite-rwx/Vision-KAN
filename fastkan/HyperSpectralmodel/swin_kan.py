import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


import math
from typing import *

from functools import partial
from einops.layers.torch import Rearrange, Reduce

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinMLPBlock(nn.Module):
    r""" Swin MLP Block.
        与标准版不一致，修改删减了一些东西，仅保留了swin的主要结构
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion. 输入图像的尺寸，这里为初始尺寸，与原始swin transformer不同，本模型不要求输入尺寸与设置尺寸必须一致，但应当为window size的整数倍，
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=32, input_resolution=[64, 64], num_heads=2, window_size=8, shift_size=0,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = FastKANLayer(dim, mlp_hidden_dim)

    def forward(self, x):
        '''

        :param x: B C H W
        :param input_resolution:
        :return: feature
        '''
        H, W = self.input_resolution
        B, C, H_IN, W_IN = x.shape
        if H != H_IN or W != W_IN:
            assert H <= H_IN
            assert W <= W_IN
            assert H_IN % self.window_size == 0
            assert W_IN % self.window_size == 0
            H, W = H_IN, W_IN

        # shortcut = x

        x = rearrange(x, 'b c h w -> b h w c')


        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Window/Shifted-Window Spatial MLP
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads,
                                         C // self.num_heads)# nW*B, window_size*window_size, num_Head, C//nH
        x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)  # nW*B, nH*window_size*window_size, C//nH
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        # merge windows
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x

        x = rearrange(x, 'b h w c -> b c h w')

        # FFN
        # x = shortcut + self.drop_path(x)

        return x




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


class SwinPermutatorKanBlock(nn.Module):
    r""" Swin KAN Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion. 输入图像的尺寸，这里为初始尺寸，与原始swin transformer不同，本模型不要求输入尺寸与设置尺寸必须一致，但应当为window size的整数倍，
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_dim=32, dim=32, input_resolution=[64, 64], num_heads=2, window_size=8, shift_size=0,depth=1
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.depth = depth

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        # use group convolution to implement multi-head MLP
        # self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
        #                              self.num_heads * self.window_size ** 2,
        #                              kernel_size=1,
        #                              groups=self.num_heads)

        # self.embedding = nn.Sequential(
        #             Rearrange('b h w c -> b (h w) c'),
        #             FastKANLayer(input_dim, dim),
        #             Rearrange('b (h w) c -> b h w c', h=self.window_size, w=self.window_size),
        #         )
        self.body = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim,
                    nn.Sequential(
                        ParallelSum(
                            nn.Sequential(
                                Rearrange('b h w (c s) -> b (w c) (h s)', s=self.num_heads),
                                FastKANLayer(self.window_size * self.num_heads, self.window_size * self.num_heads),
                                Rearrange('b (w c) (h s) -> b h w (c s)', s=self.num_heads, h=self.window_size, w=self.window_size),
                            ),
                            nn.Sequential(
                                Rearrange('b h w (c s) -> b (h c) (w s)', s=self.num_heads),
                                FastKANLayer(self.window_size * self.num_heads, self.window_size * self.num_heads),
                                Rearrange('b (h c) (w s) -> b h w (c s)', s=self.num_heads, h=self.window_size, w=self.window_size),
                            ),
                            nn.Sequential(
                                Rearrange('b h w c -> b (h w) c'),
                                FastKANLayer(dim, dim),
                                Rearrange('b (h w) c -> b h w c', h=self.window_size, w=self.window_size),
                            )
                        ),
                        nn.Sequential(
                            Rearrange('b h w c -> b (h w) c'),
                            FastKANLayer(dim, dim),
                            Rearrange('b (h w) c -> b h w c', h=self.window_size, w=self.window_size),
                        ),
                    ),
                ),
                # PreNormResidual(dim, nn.Sequential(
                #     # FastKAN([dim, dim * expansion_factor]),
                #     # nn.GELU(),
                #     # nn.Dropout(dropout),
                #     # FastKAN([dim * expansion_factor, dim]),
                #     # nn.Dropout(dropout)
                # ))
            ) for _ in range(depth)],
        )

    def forward(self, x):
        '''
        :param x: B C H W
        :return: feature B C H W
        '''
        H, W = self.input_resolution
        B, C, H_IN, W_IN = x.shape
        if H != H_IN or W != W_IN:
            assert H <= H_IN
            assert W <= W_IN
            assert H_IN % self.window_size == 0
            assert W_IN % self.window_size == 0
            H, W = H_IN, W_IN

        # shortcut = x

        x = rearrange(x, 'b c h w -> b h w c')


        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        mlp_windows = self.body(x_windows)

        # merge windows

        shifted_x = window_reverse(mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x

        x = rearrange(x, 'b h w c -> b c h w')

        return x


class SwinConvKanBlock(nn.Module):
    r""" Swin KAN Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion. 输入图像的尺寸，这里为初始尺寸，与原始swin transformer不同，本模型不要求输入尺寸与设置尺寸必须一致，但应当为window size的整数倍，
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_dim=32, dim=32, input_resolution=[64, 64], num_heads=2, window_size=8, shift_size=0,depth=1,expansion_factor=2
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.depth = depth

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        # use group convolution to implement multi-head MLP
        # self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
        #                              self.num_heads * self.window_size ** 2,
        #                              kernel_size=1,
        #                              groups=self.num_heads)

        # self.embedding = nn.Sequential(
        #             Rearrange('b h w c -> b (h w) c'),
        #             FastKANLayer(input_dim, dim),
        #             Rearrange('b (h w) c -> b h w c', h=self.window_size, w=self.window_size),
        #         )
        self.body = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim,
                    nn.Sequential(
                        ParallelSum(
                            nn.Sequential(
                                Rearrange('b h w c -> b c h w'),
                                nn.Conv2d(dim, dim*expansion_factor, kernel_size=3, padding=(3 - 1) // 2, bias=False, groups=num_heads),
                                nn.GELU(),
                                nn.Conv2d(dim*expansion_factor, dim, kernel_size=3, padding=(3 - 1) // 2, bias=False, groups=num_heads),
                                Rearrange('b c h w -> b h w c'),
                            ),
                            nn.Sequential(
                                Rearrange('b h w c -> b (h w) c'),
                                FastKANLayer(dim, dim),
                                Rearrange('b (h w) c -> b h w c', h=self.window_size, w=self.window_size),
                            )
                        ),
                        nn.Sequential(
                            Rearrange('b h w c -> b (h w) c'),
                            FastKANLayer(dim, dim),
                            Rearrange('b (h w) c -> b h w c', h=self.window_size, w=self.window_size),
                        ),
                    ),
                ),
                # PreNormResidual(dim, nn.Sequential(
                #     # FastKAN([dim, dim * expansion_factor]),
                #     # nn.GELU(),
                #     # nn.Dropout(dropout),
                #     # FastKAN([dim * expansion_factor, dim]),
                #     # nn.Dropout(dropout)
                # ))
            ) for _ in range(depth)],
        )

    def forward(self, x):
        '''
        :param x: B C H W
        :return: feature B C H W
        '''
        H, W = self.input_resolution
        B, C, H_IN, W_IN = x.shape
        if H != H_IN or W != W_IN:
            assert H <= H_IN
            assert W <= W_IN
            assert H_IN % self.window_size == 0
            assert W_IN % self.window_size == 0
            H, W = H_IN, W_IN

        # shortcut = x

        x = rearrange(x, 'b c h w -> b h w c')


        # shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        mlp_windows = self.body(x_windows)

        # merge windows

        shifted_x = window_reverse(mlp_windows, self.window_size, _H, _W)  # B H' W' C

        # reverse shift
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x

        x = rearrange(x, 'b h w c -> b c h w')

        return x

class SwinConvKan(nn.Module):
    """ A Swin Permutator Kan model.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, input_dim=3, dim=32, output_dim=31, input_resolution=[64, 64], depth=4, num_heads=2, window_size=8,
                 # mlp_ratio=4., drop=0., drop_path=0.,
                 # norm_layer=nn.LayerNorm,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size

        # self.conv_in = nn.Conv2d(input_dim, dim, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv_in = SwinConvKanBlock(input_dim=input_dim, dim=dim, input_resolution=input_resolution,
                         num_heads=1, window_size=window_size,
                         shift_size=0,
                         )
        # build blocks
        self.blocks = nn.ModuleList([
            SwinConvKanBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         )#input_dim=32, dim=32, input_resolution=[64, 64], num_heads=2, window_size=8, shift_size=0,depth=1
            for i in range(depth)])

        # self.conv_out = nn.Conv2d(dim, output_dim, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv_out = SwinConvKanBlock(input_dim=dim, dim=output_dim, input_resolution=input_resolution,
                         num_heads=1, window_size=window_size,
                         shift_size=0,
                         )


    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = self.window_size, self.window_size
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.conv_out(x)
        return x[:, :, :h_inp, :w_inp]

class SwinPermutatorKan(nn.Module):
    """ A Swin Permutator Kan model.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, input_dim=3, dim=32, output_dim=31, input_resolution=[64, 64], depth=4, num_heads=2, window_size=8,
                 # mlp_ratio=4., drop=0., drop_path=0.,
                 # norm_layer=nn.LayerNorm,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size

        self.conv_in = nn.Conv2d(input_dim, dim, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        # self.conv_in = SwinPermutatorKanBlock(input_dim=input_dim, dim=dim, input_resolution=input_resolution,
        #                  num_heads=1, window_size=window_size,
        #                  shift_size=0,
        #                  )
        # build blocks
        self.blocks = nn.ModuleList([
            SwinPermutatorKanBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         )#input_dim=32, dim=32, input_resolution=[64, 64], num_heads=2, window_size=8, shift_size=0,depth=1
            for i in range(depth)])

        self.conv_out = nn.Conv2d(dim, output_dim, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        # self.conv_out = SwinPermutatorKanBlock(input_dim=dim, dim=output_dim, input_resolution=input_resolution,
        #                  num_heads=1, window_size=window_size,
        #                  shift_size=0,
        #                  )


    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = self.window_size, self.window_size
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.conv_out(x)
        return x[:, :, :h_inp, :w_inp]


if __name__ == "__main__":
    model = SwinConvKan(input_dim=3, dim=32,output_dim=31, input_resolution=[64, 64], num_heads=2, window_size=16).cuda()
    y = model(torch.rand(4,3,64,64).cuda())
    print(y.shape)
