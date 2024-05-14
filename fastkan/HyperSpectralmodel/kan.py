import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from einops import rearrange
import numpy as np

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

#https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
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

#https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py
# code modified from https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fourier_kan.py
# https://github.com/GistNoesis/FourierKAN/

# This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
# It should be easier to optimize as fourier are more dense than spline (global vs local)
# Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
# The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
# Avoiding the issues of going out of grid
#
class FourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):
        super(FourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                                (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        # We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if (self.addbias):
            y += self.bias
        # End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, outshape)
        return y

# code modified from https://github.com/lif314/X-KANeRF/blob/main/xKANeRF/xKAN/fourier_kan.py
# 这玩意比fast Kan要费显存的多
class Fourier_KAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            grid_size: int = 8,
            spline_order: int = 0,  # placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FourierKANLayer(
                inputdim=in_dim,
                outdim=out_dim,
                gridsize=grid_size,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class kanSSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31):
        super(kanSSR, self).__init__()

        # self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        # modules_body = [GAU_VQ_HSI(dim=31, stage=2) for _ in range(stage)]
        # self.body = nn.ModuleList(modules_body)
        # self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

        self.body = Fourier_KAN([in_channels, n_feat, out_channels])

        self.out_channels = out_channels

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 2, 2
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        h = rearrange(x, 'b c h w -> b (h w) c')

        h = self.body(h)

        h = rearrange(h, 'b (h w) c -> b c h w', h=h_inp+pad_h, w=w_inp+pad_w, c=self.out_channels)

        return h[:, :, :h_inp, :w_inp]

#kanSSR with convin
# class kanSSR(nn.Module):
#     def __init__(self, in_channels=3, out_channels=31, n_feat=31):
#         super(kanSSR, self).__init__()
#
#         self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
#         # modules_body = [GAU_VQ_HSI(dim=31, stage=2) for _ in range(stage)]
#         # self.body = nn.ModuleList(modules_body)
#         # self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
#
#         self.body = FastKAN([n_feat, n_feat, n_feat, n_feat, out_channels])
#
#         self.out_channels = out_channels
#
#     def forward(self, x, loss=None):
#         """
#         x: [b,c,h,w]
#         return out:[b,c,h,w]
#         """
#         b, c, h_inp, w_inp = x.shape
#         hb, wb = 2, 2
#         pad_h = (hb - h_inp % hb) % hb
#         pad_w = (wb - w_inp % wb) % wb
#         x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
#
#         x = self.conv_in(x)
#
#         h = rearrange(x, 'b c h w -> b (h w) c')
#
#         h = self.body(h)
#
#         h = rearrange(h, 'b (h w) c -> b c h w', h=h_inp+pad_h, w=w_inp+pad_w, c=self.out_channels)


        return h[:, :, :h_inp, :w_inp]
if __name__ == "__main__":

    model=kanSSR().cuda()
    with torch.no_grad():
        x = torch.randn(1, 3, 128, 128).cuda()
        out = model(x)#
    print(out.shape)


