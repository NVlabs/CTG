import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .diffuser_helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

import tbsim.utils.tensor_utils as TensorUtils

class ResidualTemporalMapBlockConcat(nn.Module):

    def __init__(self, inp_channels, out_channels, time_embed_dim, horizon, kernel_size=5):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)



class TemporalMapUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        output_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        diffuser_building_block='concat'
    ):
        super().__init__()

        if diffuser_building_block == 'concat':
            ResidualTemporalMapBlock = ResidualTemporalMapBlockConcat
        else:
            raise NotImplementedError
        
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        cond_dim = cond_dim + time_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalMapBlock(dim_in, dim_out, time_embed_dim=cond_dim, horizon=horizon),
                ResidualTemporalMapBlock(dim_out, dim_out, time_embed_dim=cond_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalMapBlock(mid_dim, mid_dim, time_embed_dim=cond_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalMapBlock(mid_dim, mid_dim, time_embed_dim=cond_dim, horizon=horizon)

        final_up_dim = None
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalMapBlock(dim_out * 2, dim_in, time_embed_dim=cond_dim, horizon=horizon),
                ResidualTemporalMapBlock(dim_in, dim_in, time_embed_dim=cond_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            final_up_dim = dim_in

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(final_up_dim, final_up_dim, kernel_size=5),
            nn.Conv1d(final_up_dim, output_dim, 1),
        )

    def forward(self, x, aux_info, time):
        '''
            x : [  B*N, T, D ] or [ B*N, M, T, D ]
            aux_info['cond_feat'] : [B*N, C] or [ B*N, M, C ]
        '''
        len_x_shape = len(x.shape)
        if len_x_shape == 4:
            BN, M, T, _ = x.shape
            # [ B*N, M, T, D ] -> [ B*N*M, T, D ]
            x = x.reshape(BN * M, T, -1)
            # [ B*N, M, C ] -> [ B*N*M, C ]
            cond_feat = aux_info["cond_feat"].reshape(BN * M, -1)
            # [ B*N ] -> [ B*N*M ]
            time = TensorUtils.repeat_by_expand_at(time, repeats=M, dim=0)
        else:
            cond_feat = aux_info['cond_feat']
        x = einops.rearrange(x, 'b h t -> b t h')
        # print('rearrange x.size()', x.size())
        # print('time', time)
        # print('time.size()', time.size())
        # (B*N) -> (B*N, K_d)
        t = self.time_mlp(time)
        # print('t.size()', t.size())
        t = torch.cat([t, cond_feat], dim=-1)
        # raise
        h = []
        for resnet, resnet2, downsample in self.downs:
            # print('down1 x.size()', x.size())
            x = resnet(x, t)
            # print('down2 x.size()', x.size())
            x = resnet2(x, t)
            # print('down3 x.size()', x.size())
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        # print('mid1 x.size()', x.size())
        x = self.mid_block2(x, t)
        # print('mid2 x.size()', x.size())
        for resnet, resnet2, upsample in self.ups:
            # print('x.shape', x.shape)
            # print('h[-1].shape', h[-1].shape)
            x = torch.cat((x, h.pop()), dim=1)
            # print('up1 x.size()', x.size())
            x = resnet(x, t)
            # print('up2 x.size()', x.size())
            x = resnet2(x, t)
            # print('up3 x.size()', x.size())
            x = upsample(x)
            

        x = self.final_conv(x)
        # print('final conv x.size()', x.size())

        x = einops.rearrange(x, 'b t h -> b h t')
        # print('output x.size()', x.size())
        if len_x_shape == 4:
            x = x.reshape(BN, M, T, -1)
        return x