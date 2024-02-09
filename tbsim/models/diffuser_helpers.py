from typing import Dict, Union, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision.models.feature_extraction import create_feature_extractor
import tbsim.models.base_models as base_models
from tbsim.dynamics.base import Dynamics

from tbsim.utils.diffuser_utils.arrays import to_np


#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

def prepare_hist_in(pos, yaw, speed, extent, avail, norm_info):
    '''
    INPUTS:
    - pos : (B, T, 2)
    - yaw : (B, T, 1)
    - speed : (B, T)
    - extent: (B, 3)
    - avail: (B, T)
    - norm_info : tuple of (add_coeffs, div_coeffs) torch tensors in order of [x,y,vel,len,width]
    OUTPUTS:
    - hist_in : (B, T*7)
    '''
    B, T, _ = pos.size()
    # convert yaw to heading vec
    hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1) # (B, T, 2)
    # only need length, width for pred
    lw = extent[:,:2].unsqueeze(1).expand((B, T, 2)) # (B, 1, 2)

    # normalize everything
    #  note: don't normalize hvec since already unit vector
    add_coeffs = norm_info[0]
    div_coeffs = norm_info[1]
    pos = (pos + add_coeffs[:2][None,None]) / div_coeffs[:2][None,None]
    speed = (speed.unsqueeze(-1) + add_coeffs[2]) / div_coeffs[2]
    lw = (lw + add_coeffs[3:][None,None]) / div_coeffs[3:][None,None]

    # combine to get full input
    hist_in = torch.cat([pos, hvec, speed, lw, avail.unsqueeze(-1)], dim=-1)

    # zero out values we don't have data for
    hist_in[~avail] = 0.0

    # flatten so trajectory is one long vector
    hist_in = hist_in.reshape((B, -1))

    return hist_in

class AgentHistoryEncoder(nn.Module):
    '''
    MLP encodes past state history.

    Assumes inputs will be (x,y,hx,hy,s,l,w,avail)
    '''
    def __init__(self, num_steps,
                       out_dim=128,
                       use_norm=True, # layernorm
                       norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]), # add coeffs and div coeffs to standardize input data to model [x,y,vel,len,width]
                       ):
        super().__init__()
        assert len(norm_info) == 2, norm_info
        self.add_coeffs = torch.tensor(norm_info[0])
        self.div_coeffs = torch.tensor(norm_info[1])
        self.state_dim = 8 # (x,y,hx,hy,s,l,w,avail)
        input_dim = num_steps * self.state_dim
        layer_dims = (input_dim, input_dim, out_dim, out_dim)
        self.traj_mlp = base_models.MLP(input_dim, out_dim, layer_dims, normalization=use_norm)

    def forward(self, pos, yaw, speed, extent, avail):
        '''
        - pos : (B, T, 2)
        - yaw : (B, T, 1)
        - speed : (B, T)
        - extent: (B, 3)
        - avail: (B, T)
        '''
        net_in = prepare_hist_in(pos, yaw, speed, extent, avail, (self.add_coeffs.to(pos.device), self.div_coeffs.to(pos.device)))
        net_out = self.traj_mlp(net_in)
        return net_out

    def scene_forward(self, net_in):
        '''
        added for scenediffuser
        - net_in : (B, M, T, 8)

        - net_out : (B*M, d_k)
        '''
        B, M, _, _ = net_in.shape
        net_out = self.traj_mlp(net_in.reshape(B*M, -1))
        return net_out
    
class SimpleAgentHistoryEncoder(nn.Module):
    '''
    MLP encodes past state history.

    Assumes inputs will be (x,y,hx,hy,s,l,w,avail)
    '''
    def __init__(self, input_dim=8,
                       out_dim=128,
                       use_norm=True, # layernorm
                       ):
        super().__init__()
        layer_dims = (input_dim, out_dim)
        self.traj_mlp = base_models.MLP(input_dim, out_dim, layer_dims, normalization=use_norm)

    def forward(self, net_in):
        '''
        - net_in : (B, M, T, 7)

        - net_out : (B, M, T, d_k)
        '''
        net_out = self.traj_mlp(net_in)
        return net_out


class NeighborHistoryEncoder(nn.Module):
    '''
    MLP encodes past state history.
    '''
    def __init__(self, num_steps,
                       out_dim=128,
                       use_norm=True,
                       norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])):
        super().__init__()
        self.agt_hist_encoder = AgentHistoryEncoder(num_steps, out_dim, use_norm, norm_info)

    def forward(self, pos, yaw, speed, extent, avail):
        '''
        - pos : (B, Q, T, 2)
        - yaw : (B, Q, T, 1)
        - speed : (B, Q, T)
        - extent: (B, Q, 3)
        - avail: (B, Q, T)
        '''
        # get encoding for each neighbor's traj
        B, Q, T, _ = pos.size()
        neighbor_enc = self.agt_hist_encoder(pos.reshape((B*Q, T, 2)),
                                             yaw.reshape((B*Q, T, 1)),
                                             speed.reshape((B*Q, T)),
                                             extent.reshape((B*Q, 3)),
                                             avail.reshape((B*Q, T)))
        # set unavailable neighbors (i.e. padding) to be -inf so it doesn't affect max pool
        neighbor_enc = neighbor_enc.reshape((B, Q, -1))
        padded_neighbor_mask = torch.sum(avail, dim=-1) == 0.0 # all timesteps are unavailable
        neighbor_enc = torch.where(padded_neighbor_mask.unsqueeze(-1).expand_as(neighbor_enc),
                                   -torch.inf * torch.ones_like(neighbor_enc),
                                   neighbor_enc)

        # pool over neighbors
        neighbor_pool = torch.amax(neighbor_enc, dim=1)

        # if an agent has no neighbors, just replace with zeros
        no_neighbors_mask = torch.sum(padded_neighbor_mask, dim=-1) == Q # B
        neighbor_pool = torch.where(no_neighbors_mask.unsqueeze(-1).expand_as(neighbor_pool),
                                    torch.zeros_like(neighbor_pool),
                                    neighbor_pool)
        return neighbor_pool
    
    def scene_forward(self, net_in, avail):
        '''
        added for scenediffuser
        - net_in : (B, M, Q, T, 8)
        - avail : (B, M, Q, T)

        - net_out : (B*M*Q, d_k)
        '''
        B, M, Q, T, _ = net_in.shape
        # (B*M*Q, T*8) -> (B*M*Q, d_k)
        neighbor_enc = self.agt_hist_encoder.traj_mlp(net_in.reshape(B*M*Q, -1))

        # set unavailable neighbors (i.e. padding) to be -inf so it doesn't affect max pool
        # (B*M*Q, d_k) -> (B*M, Q, d_k)
        neighbor_enc = neighbor_enc.reshape((B*M, Q, -1))

        avail = avail.reshape((B*M, Q, T))
        padded_neighbor_mask = torch.sum(avail, dim=-1) == 0.0 # all timesteps are unavailable
        neighbor_enc = torch.where(padded_neighbor_mask.unsqueeze(-1).expand_as(neighbor_enc),
                                   -torch.inf * torch.ones_like(neighbor_enc),
                                   neighbor_enc)

        # pool over neighbors
        # (B*M, Q, d_k) -> (B*M, d_k)
        neighbor_pool = torch.amax(neighbor_enc, dim=1)

        # if an agent has no neighbors, just replace with zeros
        # (B*M)
        no_neighbors_mask = torch.sum(padded_neighbor_mask, dim=-1) == Q
        neighbor_pool = torch.where(no_neighbors_mask.unsqueeze(-1).expand_as(neighbor_pool),
                                    torch.zeros_like(neighbor_pool),
                                    neighbor_pool)
        return neighbor_pool

class SimpleNeighborHistoryEncoder(nn.Module):
    '''
    MLP encodes past state history.

    Assumes inputs will be (x,y,hx,hy,relvx,relvy,l,w,avail)
    '''
    def __init__(self, input_dim=9,
                       out_dim=32,
                       use_norm=True, # layernorm
                       ):
        super().__init__()
        layer_dims = (input_dim, input_dim, out_dim, out_dim)
        self.traj_mlp = base_models.MLP(input_dim, out_dim, layer_dims, normalization=use_norm)

    def forward(self, net_in):
        '''
        -param net_in : (B*N*M*T_fut, Q, K_neigh)
        -output net_out_pool : (B*N*M*T_fut, K_d)
        '''
        BNMT, Q, _ = net_in.shape
        # (B*N*M*T_fut, Q, K_neigh) -> (B*N*M*T_fut*Q, K_d)
        net_out = self.traj_mlp(net_in.reshape(BNMT*Q, -1))
        
        # (B*N*M*T_fut*Q, K_d) -> (B*N*M*T_fut, Q, K_d)
        net_out = net_out.view((BNMT, Q, -1))

        # set unavailable neighbors (i.e. padding) to be -inf so it doesn't affect max pool
        # (B*N*M*T_fut, Q)
        invalid_mask = net_in[..., -1]==0
        net_out = torch.where(invalid_mask.unsqueeze(-1).expand_as(net_out),
                                   -torch.inf * torch.ones_like(net_out),
                                   net_out)

        # pool over neighbors
        # (B*N*M*T_fut, Q, K_d) -> (B*N*M*T_fut, K_d)
        net_out_pool = torch.amax(net_out, dim=1)

        # if an agent has no neighbors, just replace with zeros
        # (B*N*M*T_fut)
        no_neighbors_mask = torch.sum(invalid_mask, dim=-1) == Q
        net_out_pool = torch.where(no_neighbors_mask.unsqueeze(-1).expand_as(net_out_pool),
                                    torch.zeros_like(net_out_pool),
                                    net_out_pool)
        
        return net_out_pool


class MapEncoder(nn.Module):
    """Encodes map, may output a global feature, feature map, or both."""
    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            global_feature_dim=None,
            grid_feature_dim=None,
    ) -> None:
        super(MapEncoder, self).__init__()
        self.return_global_feat = global_feature_dim is not None
        self.return_grid_feat = grid_feature_dim is not None
        encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=global_feature_dim
        )
        self.input_image_shape = input_image_shape
        # build graph for extracting intermediate features
        feat_nodes = {
            'map_model.layer1': 'layer1',
            'map_model.layer2': 'layer2',
            'map_model.layer3': 'layer3',
            'map_model.layer4': 'layer4',
            'map_model.fc' : 'fc',
        }
        self.encoder_heads = create_feature_extractor(encoder, feat_nodes)
        if self.return_grid_feat:
            encoder_channels = list(encoder.feature_channels().values())
            input_shape_scale = encoder.feature_scales()["layer4"]
            self.decoder = MapGridDecoder(
                input_shape=(encoder_channels[-1], input_image_shape[1]*input_shape_scale, input_image_shape[2]*input_shape_scale),
                encoder_channels=encoder_channels[:-1],
                output_channel=grid_feature_dim,
                batchnorm=True,
            )
        self.encoder_feat_scales = list(encoder.feature_scales().values())

    def feat_map_out_dim(self, H, W):
        dim_scale = self.encoder_feat_scales[-4] # decoder has 3 upsampling
        return (H * dim_scale, W * dim_scale )

    def forward(self, map_inputs, encoder_feats=None):
        if encoder_feats is None:
            encoder_feats = self.encoder_heads(map_inputs)
        fc_out = encoder_feats['fc'] if self.return_global_feat else None
        encoder_feats = [encoder_feats[k] for k in ["layer1", "layer2", "layer3", "layer4"]]
        feat_map_out = None
        if self.return_grid_feat:
            feat_map_out = self.decoder.forward(feat_to_decode=encoder_feats[-1],
                                                encoder_feats=encoder_feats[:-1])
        return fc_out, feat_map_out

from tbsim.models.base_models import Up, ConvBlock, IdentityBlock

class MapGridDecoder(nn.Module):
    """
    Has 3 doubling up-samples.
    UNet part based on https://github.com/milesial/Pytorch-UNet/tree/master/unet
    """

    def __init__(self, input_shape, output_channel, encoder_channels, bilinear=True, batchnorm=True):
        super(MapGridDecoder, self).__init__()
        input_channel = input_shape[0]
        input_hw = np.array(input_shape[1:])
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(512 + encoder_channels[-1], 256, bilinear)
        input_hw = input_hw * 2

        self.up2 = Up(256 + encoder_channels[-2], 128, bilinear)
        input_hw = input_hw * 2

        self.up3 = Up(128 + encoder_channels[-3], 64, bilinear)
        input_hw = input_hw * 2

        self.conv2 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm)
        self.conv3 = nn.Conv2d(64, output_channel, kernel_size=1)
        self.out_norm = nn.LayerNorm((output_channel, int(input_hw[0]), int(input_hw[1])))

    def forward(self, feat_to_decode: torch.Tensor, encoder_feats: List[torch.Tensor]):
        assert len(encoder_feats) >= 3
        x = self.conv1(feat_to_decode)
        x = self.up1(x, encoder_feats[-1])
        x = self.up2(x, encoder_feats[-2])
        x = self.up3(x, encoder_feats[-3])
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out_norm(x)

def query_feature_grid(pos, feat_grid):
    '''
    Bilinearly interpolates given positions in feature grid.
    - pos : (B x T x 2) float
    - feat_grid : (B x C x H x W)

    Returns:
    - (B x T x C) feature interpolated to each position
    '''
    B, T, _ = pos.size()
    x, y = pos[...,0], pos[...,1]
    x = x.reshape((-1))
    y = y.reshape((-1))

    eps = 1e-3
    x = torch.clamp(x, 0, feat_grid.shape[-1] - 1 - eps)
    y = torch.clamp(y, 0, feat_grid.shape[-2] - 1 - eps)

    x0 = torch.floor(x).to(torch.long)
    y0 = torch.floor(y).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    feat_grid = torch.permute(feat_grid, (0, 2, 3, 1)) # B x H x W x C
    bdim = torch.arange(feat_grid.size(0))[:,None].expand((B, T)).reshape((-1))

    Ia = feat_grid[bdim, y0, x0]
    Ib = feat_grid[bdim, y1, x0]
    Ic = feat_grid[bdim, y0, x1]
    Id = feat_grid[bdim, y1, x1]

    x0 = x0.to(torch.float)
    x1 = x1.to(torch.float)
    y0 = y0.to(torch.float)
    y1 = y1.to(torch.float)

    norm_const = 1. / ((x1 - x0) * (y1 - y0)) # always dealing with discrete pixels so no numerical issues (should always be 1)
    wa = (x1 - x) * (y1 - y) * norm_const
    wb = (x1 - x) * (y - y0) * norm_const
    wc = (x - x0) * (y1 - y) * norm_const
    wd = (x - x0) * (y - y0) * norm_const

    interp_feats = wa.unsqueeze(1) * Ia + \
                   wb.unsqueeze(1) * Ib + \
                   wc.unsqueeze(1) * Ic + \
                   wd.unsqueeze(1) * Id

    interp_feats = interp_feats.reshape((B, T, -1))

    return interp_feats

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

# CHANGE: specify both start and end dim for flexible handling of
# if action or state is used as conditions
def apply_conditioning(x, conditions, start_dim, end_dim):
    for t, val in conditions.items():
        x[:, t, start_dim:end_dim] = val.clone()
    return x


def get_scene_obs(pos_pred, yaw_pred, data_batch):
    '''
    Generate local coordinates to scene coordinates matrix
    Input:
        pos_pred: (num_agents, time_steps, 2)
        yaw_pred: (num_agents, time_steps, 1)
        data_batch: dict
    Output:
        agents_fut_pos: (num_agents, time_steps, 2)
        agents_fut_yaw: (num_agents, time_steps, 1)
    '''
    import tbsim.utils.geometry_utils as GeoUtils

    n = data_batch['agent_from_world'].size()[0]
    ego_idx = [0]
    agent_idx = [i for i in range(1, n)]

    center_from_agent = data_batch["agent_from_world"][ego_idx] @ data_batch["world_from_agent"][agent_idx]
    center_from_agents = torch.cat((torch.eye(3,device=center_from_agent.device).unsqueeze(0),center_from_agent),0)

    # agents_hist_pos = GeoUtils.transform_points_tensor(data_batch["history_positions"], center_from_agents) * data_batch["history_availabilities"].unsqueeze(-1)    
    # agents_hist_yaw = (data_batch["history_yaws"] + data_batch["yaw"][:,None,None] - data_batch["yaw"][0]) * data_batch["history_availabilities"].unsqueeze(-1)

    #target_availabilities = data_batch['target_availabilities']
    #t = target_availabilities.size()[1]
    #pos_pred = pos_pred[:, :t, :]
    #yaw_pred = yaw_pred[:, :t, :]

    agents_fut_pos = GeoUtils.transform_points_tensor(pos_pred,center_from_agents) #* target_availabilities.unsqueeze(-1)
    agents_fut_yaw = (yaw_pred+data_batch['yaw'][:,None,None]-data_batch['yaw'][0]) #* target_availabilities.unsqueeze(-1)

    # curr_yaw = agents_hist_yaw[:,-1]
    # curr_pos = agents_hist_pos[:,-1]

    # agents_from_center = GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros_like(curr_pos)) @ GeoUtils.transform_matrices(torch.zeros_like(curr_yaw).flatten(),-curr_pos)
    
    return agents_fut_pos, agents_fut_yaw


# max_steer = 0.5
# max_yawvel = math.pi * 2.0
# acce_bound = (-10, 8)
# vbound = [-10, 30]

@torch.no_grad()
def ubound(dyn_model, v):
    yawbound = torch.minimum(
        dyn_model.max_steer * torch.abs(v),
        dyn_model.max_yawvel / torch.clip(torch.abs(v), min=0.1),
    )
    yawbound = torch.clip(yawbound, min=0.1)
    acce_lb = torch.clip(
        torch.clip(dyn_model.vbound[0] - v, max=dyn_model.acce_bound[1]),
        min=dyn_model.acce_bound[0],
    )
    acce_ub = torch.clip(
        torch.clip(dyn_model.vbound[1] - v, min=dyn_model.acce_bound[0]),
        max=dyn_model.acce_bound[1],
    )
    lb = torch.cat((acce_lb, -yawbound), dim=-1)
    ub = torch.cat((acce_ub, yawbound), dim=-1)
    return lb, ub

def torch_bmm(a, b):
    if a.ndim == 3:
        return torch.bmm(a, b)
    elif a.ndim == 4:
        return torch.einsum('bijk,bikl->bijl', a, b)

def unicyle_forward_dynamics(
    dyn_model: Dynamics,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    step_time: float,
    mode: str = 'parallel',
):
    """
    Integrate the state forward with initial state x0, action u
    Note:
        Support both agent-centric and scene-centric (extra dimension for the inputs).
    Args:
        dyn_model (dynamics.Dynamics): dynamics model
        initial_states (Torch.tensor): state tensor of size [B, (A), 4]
        actions (Torch.tensor): action tensor of size [B, (A), T, 2]
        step_time (float): delta time between steps
        mode (str): 'parallel' or 'partial_parallel' or 'chain'. 'parallel' is the fastet
        but it generates different results from 'partial_parallel' and 'chain' when the
        velocity is out of bounds.
        
        When running one (three) inner loop gradient update, the network related time for each are:
        parallel: 1.2s (2.5s)
        partial_parallel: 2.9s (7.0s)
        chain: 4.4s (10.4s)
        original implementation: 5.8s (14.6s)

    Returns:
        state tensor of size [B, (A), T, 4]
    """
    

    # ------------------------------------------------------------ #
    if mode in ['parallel', 'partial_parallel']:
        with torch.no_grad():
            num_steps = actions.shape[-2]
            bm = actions.shape[:-2]
            device = initial_states.device

            mat = torch.ones(num_steps+1, num_steps+1, device=device)
            mat = torch.tril(mat)
            mat = mat.repeat(*bm, 1, 1)
            
            mat2 = torch.ones(num_steps, num_steps+1, device=device)
            mat2_h = torch.tril(mat2, diagonal=1)
            mat2_l = torch.tril(mat2, diagonal=-1)
            mat2 = torch.logical_xor(mat2_h, mat2_l).float()*0.5
            mat2 = mat2.repeat(*bm, 1, 1)

        acc = actions[..., :1]
        yawvel = actions[..., 1:]
        
        acc_clipped = torch.clip(acc, dyn_model.acce_bound[0], dyn_model.acce_bound[1])
        
        if mode == 'parallel':
            acc_paded = torch.cat((initial_states[..., -2:-1].unsqueeze(-2), acc_clipped*step_time), dim=-2)
            v_raw = torch_bmm(mat, acc_paded)
            v_clipped = torch.clip(v_raw, dyn_model.vbound[0], dyn_model.vbound[1])
        else:
            v_clipped = [initial_states[..., 2:3]] + [None] * num_steps
            for t in range(num_steps):
                vt = v_clipped[t]
                acc_clipped_t = torch.clip(acc_clipped[..., t], dyn_model.vbound[0] - vt, dyn_model.vbound[1] - vt)
                v_clipped[t+1] = vt + acc_clipped_t * step_time
            v_clipped = torch.stack(v_clipped, dim=-2)
            
        v_avg = torch_bmm(mat2, v_clipped)
        
        v = v_clipped[..., 1:, :]

        with torch.no_grad():
            v_earlier = v_clipped[..., :-1, :]
            yawbound = torch.minimum(
                dyn_model.max_steer * torch.abs(v_earlier),
                dyn_model.max_yawvel / torch.clip(torch.abs(v_earlier), min=0.1),
            )
            yawbound_clipped = torch.clip(yawbound, min=0.1)
        
        yawvel_clipped = torch.clip(yawvel, -yawbound_clipped, yawbound_clipped)

        yawvel_paded = torch.cat((initial_states[..., -1:].unsqueeze(-2), yawvel_clipped*step_time), dim=-2)
        yaw_full = torch_bmm(mat, yawvel_paded)
        yaw = yaw_full[..., 1:, :]

        # print('before clip', torch.cat((acc[0], yawvel[0]), dim=-1))
        # print('after clip', torch.cat((acc_clipped[0], yawvel_clipped[0]), dim=-1))

        yaw_earlier = yaw_full[..., :-1, :]
        vx = v_avg * torch.cos(yaw_earlier)
        vy = v_avg * torch.sin(yaw_earlier)
        v_all = torch.cat((vx, vy), dim=-1)

        # print('initial_states[0, -2:]', initial_states[0, -2:])
        # print('vx[0, :5]', vx[0, :5])

        v_all_paded = torch.cat((initial_states[..., :2].unsqueeze(-2), v_all*step_time), dim=-2)
        x_and_y = torch_bmm(mat, v_all_paded)
        x_and_y = x_and_y[..., 1:, :]

        x_all = torch.cat((x_and_y, v, yaw), dim=-1)
    
    # ------------------------------------------------------------ #
    elif mode == 'chain':
        num_steps = actions.shape[-2]
        x_all = [initial_states] + [None] * num_steps
        for t in range(num_steps):
            x = x_all[t]
            u = actions[..., t, :]
            dt = step_time
            
            with torch.no_grad():
                lb, ub = ubound(dyn_model, x[..., 2:3])
            # print('chain before clip u[0]', u[0])
            u = torch.clip(u, lb, ub)
            # print('chain after clip u[0]', u[0])
            theta = x[..., 3:4]
            dxdt = torch.cat(
                (
                    torch.cos(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    torch.sin(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    u,
                ),
                dim=-1,
            )
            # print('x[0, :3]', x[0, :3])
            # print(t, 'dxdt[0, 0]', dxdt[0, 0])
            x_all[t + 1] = x + dxdt * dt
        x_all = torch.stack(x_all[1:], dim=-2)
    # ------------------------------------------------------------ #
    else:
        raise

    return x_all

def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (..., 1)
    :param theta2: angle 2 (..., 1)
    :return diff: smallest angle difference between angles (..., 1)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

def convert_state_to_state_and_action(traj_state, vel_init, dt, data_type='torch'):
    '''
    Infer vel and action (acc, yawvel) from state (x, y, yaw) based on Unicycle.
    Note:
        Support both agent-centric and scene-centric (extra dimension for the inputs).
    Input:
        traj_state: (batch_size, [num_agents], num_steps, 3)
        vel_init: (batch_size, [num_agents],)
        dt: float
        data_type: ['torch', 'numpy']
    Output:
        traj_state_and_action: (batch_size, [num_agents], num_steps, 6)
    '''
    BM = traj_state.shape[:-2]
    if data_type == 'torch':
        sin = torch.sin
        cos = torch.cos

        device = traj_state.get_device()
        pos_init = torch.zeros(*BM, 1, 2, device=device)
        yaw_init = torch.zeros(*BM, 1, 1, device=device)
    elif data_type == 'numpy':
        sin = np.sin
        cos = np.cos

        pos_init = np.zeros((*BM, 1, 2))
        yaw_init = np.zeros((*BM, 1, 1))
    else:
        raise
    def cat(arr, dim):
        if data_type == 'torch':
            return torch.cat(arr, dim=dim)
        elif data_type == 'numpy':
            return np.concatenate(arr, axis=dim)

    target_pos = traj_state[..., :2]
    traj_yaw = traj_state[..., 2:]    

    # pre-pad with zero pos and yaw
    pos = cat((pos_init, target_pos), dim=-2)
    yaw = cat((yaw_init, traj_yaw), dim=-2)

    # estimate speed from position and orientation
    vel_init = vel_init[..., None, None]
    vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * cos(
        yaw[..., 1:, :]
    ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * sin(
        yaw[..., 1:, :]
    )
    vel = cat((vel_init, vel), dim=-2)
    
    # m/s^2
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    # rad/s
    # yawvel = (yaw[..., 1:, :] - yaw[..., :-1, :]) / dt
    yawdiff = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :])
    yawvel = yawdiff / dt
    
    pos, yaw, vel = pos[..., 1:, :], yaw[..., 1:, :], vel[..., 1:, :]

    traj_state_and_action = cat((pos, vel, yaw, acc, yawvel), dim=-1)

    return traj_state_and_action

def state_grad_general_transform(x_guidance, data_batch, transform_params, bsize, num_samp=1):
    '''
    Transform state to state and action using curr_speed.
    x_guidance: (B*N, T, 3)
    '''
    expand_speed = data_batch['curr_speed'].unsqueeze(1).expand((bsize, num_samp)).reshape((bsize*num_samp))
    x_all = convert_state_to_state_and_action(x_guidance, expand_speed, dt=transform_params['dt'])
    return x_all

def choose_act_using_guide_loss(guide_losses, guide_configs, act_idx):
    '''
    Choose actions based on the guide losses

    -act_idx: (B, ) 
    '''
    accum_guide_loss = torch.stack([v for _,v in guide_losses.items()], dim=2) # (B, N, guide total numbers over all scenes)
    # each scene separately since may contain different guidance
    scount = 0
    for sidx in range(len(guide_configs)):
        scene_guide_cfg = guide_configs[sidx]

        ends = scount + len(scene_guide_cfg)
        scene_guide_loss = accum_guide_loss[..., scount:ends] # (B, N, guide numbers / scene)
        scount = ends
        
        scene_mask = ~torch.isnan(torch.sum(scene_guide_loss, dim=[1,2]))
        scene_guide_loss = scene_guide_loss[scene_mask].cpu()

        weights = torch.as_tensor([c.weight for c in scene_guide_cfg], device=scene_guide_loss.device)[None, None, :]

        scene_guide_loss = torch.nansum(scene_guide_loss*weights, dim=-1) # (B[non-nan], N)
        
        # print('scene_guide_loss', scene_guide_loss)
        is_scene_level = np.array([guide_cfg.name in ['agent_collision', 'social_group'] for guide_cfg in scene_guide_cfg])
        if np.sum(is_scene_level) > 0: 
            # choose which sample minimizes at the scene level (where each sample is a "scene")
            scene_act_idx = torch.argmin(torch.sum(scene_guide_loss, dim=0))
        else:
            # each agent can choose the sample that minimizes guidance loss independently
            scene_act_idx = torch.argmin(scene_guide_loss, dim=-1)

        act_idx[scene_mask] = scene_act_idx.to(act_idx.device)

    return act_idx

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim, loss_decay_rates={}):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ, data_batch={}):
        '''
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
            pred, targ : tensor
                [ B, (M), T, transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[..., 0, -self.action_dim:] / self.weights[0, -self.action_dim:]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ, data_batch={}):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                to_np(pred).squeeze(),
                to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')
    
class WeightedL2WithCollision(WeightedLoss):
    def __init__(self, weights, action_dim, loss_decay_rates):
        super().__init__(weights, action_dim)
        from tbsim.utils.guidance_loss import AgentCollisionLoss, MapCollisionLoss
        
        self.agent_collision_loss_fn = AgentCollisionLoss(num_disks=2, buffer_dist=0.2, decay_rate=loss_decay_rates['collision_decay_rate'])
        self.map_collision_loss_fn = MapCollisionLoss(num_points_lw=(10, 10), decay_rate=loss_decay_rates['offroad_decay_rate'])

    def forward(self, pred, targ, data_batch={}):
        '''
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
            pred, targ : tensor
                [ B, (M), T, transition_dim ]
        '''
        
        # (B, M, T, transition_dim) -> (B*M, T, transition_dim) -> (B*M, 1, T, transition_dim)
        B, M, T, _ = pred.shape
        x_loss = pred.view(B*M, T, -1).unsqueeze(1)

        if data_batch != {}:
            agent_collision_loss = self.agent_collision_loss_fn(x_loss, data_batch)
            map_collision_loss = self.map_collision_loss_fn(x_loss, data_batch)
            agent_collision_loss_mean = torch.mean(agent_collision_loss)
            map_collision_loss_mean = torch.mean(map_collision_loss)
            loss_info = {'collision_loss': agent_collision_loss_mean, 'offroad_loss': map_collision_loss_mean}
        else:
            loss_info = {}
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        # a0_loss = (loss[..., 0, -self.action_dim:] / self.weights[0, -self.action_dim:]).mean()
        
        return weighted_loss, loss_info
    
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'l2_with_agent_collision_and_map_collision': WeightedL2WithCollision,
}

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        with torch.no_grad():
            ema_state_dict = ma_model.state_dict()
            for key, value in current_model.state_dict().items():
                ema_value = ema_state_dict[key]
                ema_value.copy_(self.beta * ema_value + (1. - self.beta) * value)

def angle_wrap_torch(radians):
    pi = torch.tensor(np.pi, device=radians.device)
    return (radians + pi) % (2 * pi) - pi

