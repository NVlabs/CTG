from typing import Dict, List
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.models.base_models as base_models
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.geometry_utils as GeoUtils

from tbsim.utils.guidance_loss import PerturbationGuidance

import numpy as np
from tbsim.utils.diffuser_utils.progress import Progress, Silent
from .diffuser_helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
    query_feature_grid,
    convert_state_to_state_and_action,
    unicyle_forward_dynamics,
    AgentHistoryEncoder,
    NeighborHistoryEncoder,
    SimpleNeighborHistoryEncoder,
    MapEncoder,
    angle_wrap_torch,
)
from .scenetemporal import SceneTransformer
from .temporal import TemporalMapUnet
import tbsim.dynamics as dynamics
from tbsim.utils.guidance_loss import verify_guidance_config_list, verify_constraint_config, apply_constraints, DiffuserGuidance, PerturbationGuidance
from trajdata.utils.arr_utils import angle_wrap
from tbsim.utils.trajdata_utils import extract_data_batch_for_guidance

class SceneDiffuserModel(nn.Module):
    """SceneDiffuser model for planning.
    
    data usage from data_batch
    # "history_positions" (coord), "history_yaws" (coord), "history_speeds", "extent", "history_availabilities"
    # "curr_speed"
    # "target_positions" (coord), "target_yaws" (coord), "target_availabilities"
    
    # If modeling map
    # ("image" and "maps") or "closest_lane_point" (coord)

    # If modeling neighbors:
    # "all_other_agents_history_positions" (coord), "all_other_agents_history_yaws" (coord), "all_other_agents_history_speeds", "all_other_agents_extents", "all_other_agents_history_availabilities"
    """

    def __init__(
            self,
            map_encoder_model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            map_grid_feature_dim: int,
            
            diffuser_model_arch: str,
            horizon: int,

            observation_dim: int, 
            action_dim: int,

            output_dim: int,

            # curr_state_feature_dim = 64,
            rasterized_map = True,
            use_map_feat_global = True,
            use_map_feat_grid = False,
            rasterized_hist = True,
            hist_num_frames = 31,
            hist_feature_dim = 128,

            n_timesteps=100,
            
            loss_type='l2', 
            clip_denoised=False, 
            
            predict_epsilon=True,
            action_weight=1.0, 
            loss_discount=1.0, 
            loss_weights=None,
            loss_decay_rates={},

            dynamics_type=None,
            dynamics_kwargs={},

            action_loss_only=False,

            diffuser_input_mode='state_and_action',
            use_reconstructed_state=False,

            use_conditioning=True,
            cond_fill_value=-1.0,

            # norm info is ([add_coeffs, div_coeffs])
            diffuser_norm_info=([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4]),
            agent_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            neighbor_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            neighbor_fut_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),

            agent_hist_embed_method='mlp',
            neigh_hist_embed_method=None,
            map_embed_method='transformer',
            interaction_edge_speed_repr='abs_speed',
            mask_social_interaction=False,
            mask_edge=False,
            neighbor_inds=[0,1,2,3],
            edge_attr_separation=[],
            social_attn_radius=50,
            use_last_hist_step=False,
            use_noisy_fut_edge=False,
            use_const_speed_edge=False,
            normalize_rel_states=True,
            all_interactive_social=False,
            mask_time=True,
            layer_num_per_edge_decoder=1,
            attn_combination='sum',
            single_cond_feat=False,
            disable_control_on_stationary=False,
            coordinate='agent_centric',
            
            dt=0.1,
    ) -> None:
        print('disable_control_on_stationary', disable_control_on_stationary)
        super().__init__()

        # ------ misc ------
        self.global_t = 0

        # ------ basic dimensions ------
        self.dt = dt
        self.horizon = horizon
        self.hist_num_frames = hist_num_frames
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.output_dim = output_dim

        self.hist_feature_dim = hist_feature_dim

        # ------ architectures ------
        # ['TemporalMapUnet', 'SceneTransformer']
        self.diffuser_model_arch = diffuser_model_arch
        # decide how the context is integrated
        self.agent_hist_embed_method = agent_hist_embed_method
        self.neigh_hist_embed_method = neigh_hist_embed_method
        self.map_embed_method = map_embed_method
        self.interaction_edge_speed_repr = interaction_edge_speed_repr
        self.normalize_rel_states = normalize_rel_states
        # this field is used only when SceneTransformer is used
        self.mask_social_interaction = mask_social_interaction
        # this field is used only when agent_hist_embed_method == 'mlp' or map_embed_method == 'cnn'
        self.single_cond_feat = single_cond_feat
        self.mask_edge = mask_edge
        self.neighbor_inds = neighbor_inds
        self.edge_attr_separation = edge_attr_separation
        self.social_attn_radius = social_attn_radius
        self.use_last_hist_step = use_last_hist_step
        self.use_noisy_fut_edge = use_noisy_fut_edge
        self.use_const_speed_edge = use_const_speed_edge
        self.all_interactive_social = all_interactive_social
        self.mask_time = mask_time
        self.layer_num_per_edge_decoder = layer_num_per_edge_decoder
        self.attn_combination = attn_combination

        self.disable_control_on_stationary = disable_control_on_stationary
        self.stationary_mask = None
        self.coordinate = coordinate

        assert not self.single_cond_feat or (self.single_cond_feat and (self.agent_hist_embed_method == 'mlp' or self.map_embed_method == 'cnn'))
        if self.single_cond_feat:
            cond_in_feat_size = 0

        # this applies to map and past NEIGHBOR conditioning only
        #       curr state or past ego trajecotry are always given
        self.use_conditioning = use_conditioning
        # for test-time classifier-free guidance, if desired
        self.cond_fill_value = cond_fill_value 

        self.rasterized_map = rasterized_map
        self.rasterized_hist = rasterized_hist

        self.agent_hist_norm_info = agent_hist_norm_info
        self.neighbor_hist_norm_info = neighbor_hist_norm_info
        self.neighbor_fut_norm_info = neighbor_fut_norm_info

        # agent history encoding (mlp)
        if self.agent_hist_embed_method == 'mlp':
            self.agent_hist_encoder = None
            if not self.rasterized_hist:
                # ego history is ALWAYS used as conditioning
                self.agent_hist_encoder = AgentHistoryEncoder(hist_num_frames,
                                                            out_dim=self.hist_feature_dim,
                                                            use_norm=True,
                                                            norm_info=agent_hist_norm_info)
            if self.single_cond_feat:
                cond_in_feat_size += self.hist_feature_dim

        # neighbor history encoding (mlp)
        if self.neigh_hist_embed_method == 'mlp':
            if self.use_conditioning and not self.rasterized_hist:
                self.neighbor_hist_encoder = NeighborHistoryEncoder(hist_num_frames, out_dim=self.hist_feature_dim, use_norm=True, norm_info=self.neighbor_hist_norm_info)
                if self.single_cond_feat:
                    cond_in_feat_size += self.hist_feature_dim
        # neighbor info is encoded through edge and fut input
        elif self.neigh_hist_embed_method == 'interaction_edge_and_input':
            # for encoding neighbor future
            # the input dimension assumes relvel
            self.neighbor_fut_encoder = SimpleNeighborHistoryEncoder(input_dim=9, out_dim=32, use_norm=True)



        # map encoding (cnn)
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            self.map_encoder = None
            self.use_map_feat_global = use_map_feat_global
            self.use_map_feat_grid = use_map_feat_grid
            self.input_image_shape = input_image_shape
            if self.use_conditioning and (self.rasterized_map or self.rasterized_hist):
                self.map_encoder = MapEncoder(
                    model_arch=map_encoder_model_arch,
                    input_image_shape=input_image_shape,
                    global_feature_dim=map_feature_dim if self.use_map_feat_global else None,
                    grid_feature_dim=map_grid_feature_dim if self.use_map_feat_grid else None,
                )
                if self.single_cond_feat:
                    cond_in_feat_size += map_feature_dim

        # if apply mlp to process all the conditions
        if self.single_cond_feat:
            # hacky way to get the output feature size
            if self.diffuser_model_arch == 'SceneTransformer':
               cond_out_feat_size = 128
            elif self.diffuser_model_arch == 'TemporalMapUnet':
                cond_out_feat_size = 256
            else:
                raise NotImplementedError
            
            combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)
            self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                    cond_out_feat_size,
                                                    combine_layer_dims,
                                                    normalization=True)


        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics()
        
        # ----- diffuser -----

        # TBD: include in algo_config
        self.use_target_availabilities = True

        self.action_loss_only = action_loss_only
        
        # x, y, vel, yaw, acc, yawvel
        assert len(diffuser_norm_info) == 2
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        assert len(norm_add_coeffs) == 6
        assert len(norm_div_coeffs) == 6
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')       

        print('self.add_coeffs', self.add_coeffs)
        print('self.div_coeffs', self.div_coeffs)


        self.use_reconstructed_state = use_reconstructed_state
        self.diffuser_input_mode = diffuser_input_mode
        
        if diffuser_input_mode == 'state':
            self.default_chosen_inds = [0, 1, 3]
        elif diffuser_input_mode == 'action':
            self.default_chosen_inds = [4, 5]
        elif diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:
            self.default_chosen_inds = [0, 1, 2, 3, 4, 5]
        else:
            raise

        if self.diffuser_model_arch == "SceneTransformer":
            # transition_in_dim = self.transition_dim
            # if self.use_map_feat_grid and self.map_encoder is not None:
            #     # will be appending map features to each step of trajectory
            #     transition_in_dim += map_grid_feature_dim
            self.model = SceneTransformer(
                                    d_k=128,
                                    d_edge=self.hist_feature_dim, 
                                    L_enc=2, 
                                    dropout=0.0, 
                                    k_attr=7,
                                    neighbor_inds=self.neighbor_inds,
                                    edge_attr_separation=self.edge_attr_separation,
                                    map_attr=2, 
                                    num_heads=16, 
                                    L_dec=2,
                                    tx_hidden_size=384, 
                                    use_map_lanes=True,

                                    transition_dim=self.transition_dim,
                                    output_dim=output_dim,
                                    agent_hist_embed_method=self.agent_hist_embed_method,
                                    neigh_hist_embed_method=self.neigh_hist_embed_method,
                                    map_embed_method=self.map_embed_method,
                                    interaction_edge_speed_repr=self.interaction_edge_speed_repr,
                                    single_cond_feat=self.single_cond_feat,
                                    mask_social_interaction=self.mask_social_interaction,
                                    mask_edge=self.mask_edge,
                                    social_attn_radius=self.social_attn_radius,
                                    all_interactive_social=self.all_interactive_social,
                                    mask_time=self.mask_time,
                                    layer_num_per_edge_decoder=self.layer_num_per_edge_decoder,
                                    attn_combination=self.attn_combination,
                                    )
        elif self.diffuser_model_arch == "TemporalMapUnet":
            self.model = TemporalMapUnet(horizon=horizon,
                                      transition_dim=self.transition_dim,
                                      cond_dim=256,
                                      output_dim=self.output_dim,
                                      dim=32,
                                      dim_mults=(2,4,8),
                                      diffuser_building_block='concat')
        else:
            print('unknown diffuser_model_arch:', self.diffuser_model_arch)
            raise
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # calculations for class-free guidance
        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - alphas_cumprod)
        # self.register_buffer('sqrt_alphas_over_one_minus_alphas_cumprod', torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod)))
        # self.register_buffer('sqrt_recip_one_minus_alphas_cumprod', 1.0 / torch.sqrt(1. - alphas_cumprod))

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        recon_loss_weights = self.get_loss_weights(action_weight, loss_discount)
        self.loss_fn = Losses[loss_type](recon_loss_weights, self.action_dim, loss_decay_rates)
        self.loss_decay_rates = loss_decay_rates

        # for guided sampling
        # This is 1 by default
        self.stride = 1
        self.apply_guidance_output = False
        self.guidance_optimization_params = None
        if self.diffuser_input_mode in ['action', 'state_and_action']:
            self.transform_params = {'scaled_input':True, 'scaled_output':True}
            self.transform = self.state_action_grad_inner_transform
        elif self.diffuser_input_mode in ['state_and_action_no_dyn']:
            self.transform_params = {'scaled_input':True, 'scaled_output':True}
            self.transform = self.state_action_no_dyn_grad_inner_transform
        else:
            self.transform_params = {}
            self.transform = self.state_grad_inner_transform
        
        self.current_constraints = None
        # wrapper for optimization using current_guidance
        self.current_perturbation_guidance = PerturbationGuidance(self.transform, self.transform_params, self.scale_traj, self.descale_traj)
        # --------------------------------------------------------------------------------------

    #------------------------------------------ guidance utils ------------------------------------------#
    
    def set_guidance(self, guidance_config_list, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config_list is not None:
            if len(guidance_config_list) > 0 and verify_guidance_config_list(guidance_config_list):
                print('Instantiating test-time guidance with configs:')
                print(guidance_config_list)
                self.current_perturbation_guidance.set_guidance(guidance_config_list, example_batch)

    def set_constraints(self, constraint_config):
        '''
        Instantiates test-time hard constraints using the config (dict) passed in.
        '''
        if constraint_config is not None and len(constraint_config) > 0:
            verify_constraint_config(constraint_config)
            print('Instantiating test-time constraints with config:')
            print(constraint_config)
            self.current_constraints = constraint_config
    
    def update_guidance(self, **kwargs):
        if self.current_perturbation_guidance.current_guidance is not None:
            self.current_perturbation_guidance.update(**kwargs)

    def clear_guidance(self):
        self.current_perturbation_guidance.clear_guidance()

    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.guidance_optimization_params = guidance_optimization_params

    def set_diffusion_specific_params(self, diffusion_specific_params):
        self.apply_guidance_intermediate = diffusion_specific_params['apply_guidance_intermediate']
        self.apply_guidance_output = diffusion_specific_params['apply_guidance_output']
        self.final_step_opt_params = diffusion_specific_params['final_step_opt_params']
        self.stride = diffusion_specific_params['stride']

    #------------------------------------------ utility ------------------------------------------#
    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None

    def prepare_scene_agent_hist(self, pos, yaw, speed, extent, avail, norm_info, scale=True, speed_repr='abs_speed'):
        '''
        Input:
        - pos : (B, M, (Q), T, 2)
        - yaw : (B, M, (Q), T, 1)
        - speed : (B, M, (Q), T)
        - extent: (B, M, (Q), 3)
        - avail: (B, M, (Q), T)
        - norm_info: [2, 5]
        - speed_repr: 'abs_speed' or 'rel_vel'
        Output:
        - hist_in: [B, M, (Q), T, 8] (x,y,cos,sin,v,l,w,avail)
        '''
        M = pos.shape[1]
        T = pos.shape[-2]

        if speed_repr == 'rel_vel_per_step':
            # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
            yaw_self = torch.diagonal(yaw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(yaw)
            # (B, M, M, T, 1) -> (B*M*M*T)
            yaw_self_ = yaw_self.reshape(-1)
            # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2) -> (B*M*M*T, 2)
            pos_self_ = torch.diagonal(pos, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(pos).reshape(-1, 2)

            # (B, M1, M2, T, 2) -> (B*M1*M2*T, 2)
            pos_ = pos.view(-1, 2)
            
            # self_from_self_per_timestep
            # (B*M1*M2*T, 3, 3)
            i_from_i_per_time = torch.stack(
                [
                    torch.stack([torch.cos(yaw_self_), -torch.sin(yaw_self_), pos_self_[..., 0]], dim=-1),
                    torch.stack([torch.sin(yaw_self_), torch.cos(yaw_self_), pos_self_[..., 1]], dim=-1),
                    torch.stack([0.0*torch.ones_like(yaw_self_), 0.0*torch.ones_like(yaw_self_), 1.0*torch.ones_like(yaw_self_)], dim=-1)
                ], dim=-2
            )
            i_per_time_from_i = torch.linalg.inv(i_from_i_per_time)
            
            # transform coord
            pos_transformed = torch.einsum("...jk,...k->...j", i_per_time_from_i[..., :-1, :-1], pos_)
            pos_transformed += i_per_time_from_i[..., :-1, -1]
            # (B*M1*M2*T, 2) -> (B, M1, M2, T, 2)
            pos = pos_transformed.view(pos.shape)

            # transform angle
            yaw = angle_wrap_torch(yaw - yaw_self)

            # print('pos.shape', pos.shape)
            # print('yaw.shape', yaw.shape)
            # print(pos[0, 2])
            # print(yaw[0, 2])
            # raise


        # convert yaw to heading vec
        hvec = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=-1) # (B, M, (Q), T, 2)
        # only need length, width for pred
        lw = extent[..., :2].unsqueeze(-2).expand(pos.shape) # (B, M, (Q), T, 2)

        # Get time to collision
        if speed_repr in ['rel_vel', 'rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']:
            d_th = 20
            t_to_col_th = 20

            # print('pos.shape', pos.shape)
            # print('pos[0,0]', pos[0,0])
            # raise

            # estimate relative distance to other agents
            if speed_repr == 'rel_vel_new_new':
                # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
                pos_self = torch.diagonal(pos, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(pos)
                pos_diff = torch.abs(pos.detach().clone() - pos_self)
            else:
                pos_diff = pos
            # (B, M, (Q), T, 2) -> (B, M, (Q), T, 1)
            rel_d = torch.norm(pos_diff, dim=-1).unsqueeze(-1)
            
            # rel_d_lw also consider the lw of both agents (half of the mean of each lw)
            # (B, M, (Q), T, 2) -> (B, M, (Q), T, 1)
            lw_avg_half = (torch.mean(lw, dim=-1) / 2).unsqueeze(-1)
            # (B, M, (Q), T, 1) -> (B, M, T, 1)
            ego_lw_avg_half = lw_avg_half[...,torch.arange(M), torch.arange(M), :, :]
            # (B, M, (Q), T, 1)
            lw_avg_half_sum = lw_avg_half + ego_lw_avg_half.unsqueeze(2).expand_as(lw_avg_half)
            # (B, M, (Q), T, 1)
            rel_d_lw = rel_d - lw_avg_half_sum
            # normalize rel_d and rel_d_lw
            rel_d = torch.clip(rel_d, min=0, max=d_th)
            rel_d = (d_th - rel_d) / d_th
            rel_d_lw = torch.clip(rel_d_lw, min=0, max=d_th)
            rel_d_lw = (d_th - rel_d_lw) / d_th

            B, M, M, T = speed.shape
            # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
            ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
            # (B, M, M, T, 2) -> (B, T, 2, M) -> (B, M, T, 2) -> (B, M, 1, T, 2) -> (B, M, M, T, 2)
            ego_lw = torch.diagonal(lw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand(B, M, M, T, 2).clone()
            ego_vx[torch.isnan(ego_vx)] = 0.0
            
            if speed_repr == 'rel_vel_new_new':
                # (B, M, M, T, 1) -> (B, T, 1, M) -> (B, M, T, 1) -> (B, M, 1, T, 1) -> (B, M, M, T, 1)
                yaw_self = torch.diagonal(yaw, dim1=1, dim2=2).permute(0, 3, 1, 2).unsqueeze(2).expand_as(yaw)
                hvec_self = torch.cat([torch.cos(yaw_self), torch.sin(yaw_self)], dim=-1)
                vx = ego_vx * hvec_self[...,0] - speed * hvec[...,0]
                vy = ego_vx * hvec_self[...,1] - speed * hvec[...,1]
            else:
                vx = ego_vx - speed * hvec[...,0]
                vy = 0 - speed * hvec[...,1]

            x_dist = pos_diff[...,0] - (ego_lw[...,0]/2) - (lw[...,0]/2)
            y_dist = pos_diff[...,1] - (ego_lw[...,1]/2) - (lw[...,1]/2)
            x_t_to_col = x_dist / vx
            y_t_to_col = y_dist / vy
            # if collision has not happened and moving in opposite direction, set t_to_col to t_to_col_th
            x_t_to_col[(x_dist>0) & (x_t_to_col<0)] = t_to_col_th
            y_t_to_col[(y_dist>0) & (y_t_to_col<0)] = t_to_col_th
            # if collision already happened, set t_to_col to 0
            x_t_to_col[x_dist<0] = 0
            y_t_to_col[y_dist<0] = 0
            # both directions need to be met for collision to happen
            rel_t_to_col = torch.max(torch.cat([x_t_to_col.unsqueeze(-1), y_t_to_col.unsqueeze(-1)], dim=-1), dim=-1)[0]
            rel_t_to_col = torch.clip(rel_t_to_col, min=0, max=t_to_col_th)
            # normalize rel_t_to_col
            rel_t_to_col = (t_to_col_th - rel_t_to_col.unsqueeze(-1)) / t_to_col_th

        # normalize everything
        #  note: don't normalize hvec since already unit vector
        add_coeffs = torch.tensor(norm_info[0]).to(pos.device)
        div_coeffs = torch.tensor(norm_info[1]).to(pos.device)

        if len(pos.shape) == 4:
            add_coeffs_expand = add_coeffs[None, None, None, :]
            div_coeffs_expand = div_coeffs[None, None, None, :]
        else:
            add_coeffs_expand = add_coeffs[None, None, None, None, :]
            div_coeffs_expand = div_coeffs[None, None, None, None, :]

        pos_original = pos.detach().clone()
        if scale:
            pos = (pos + add_coeffs_expand[...,:2]) / div_coeffs_expand[...,:2]
            speed = (speed.unsqueeze(-1) + add_coeffs[2]) / div_coeffs[2]
            lw = (lw + add_coeffs_expand[...,3:]) / div_coeffs_expand[...,3:]
        else:
            speed = speed.unsqueeze(-1)
        
        if speed_repr in ['rel_vel', 'rel_vel_new', 'rel_vel_per_step', 'rel_vel_new_new']:
            speed = speed.squeeze(-1)
            B, M, M, T = speed.shape
            if speed_repr == 'rel_vel_new_new':
                # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M) -> (B, M, 1, 1) -> (B, M, M, T)
                ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1)[...,0].unsqueeze(-1).unsqueeze(-1).expand(B, M, M, T).clone()
                ego_vx[torch.isnan(ego_vx)] = 0.0
                vx = speed * hvec[...,0] - ego_vx
                vy = speed * hvec[...,1]
            else:
                # (B, M, M, T) -> (B, T, M) -> (B, M, T) -> (B, M, 1, T) -> (B, M, M, T)
                ego_vx = torch.diagonal(speed, dim1=1, dim2=2).permute(0, 2, 1).unsqueeze(-2).expand(B, M, M, T).clone()
                ego_vx[torch.isnan(ego_vx)] = 0.0
                vx = speed * hvec[...,0] - ego_vx
                vy = speed * hvec[...,1]
            vvec = torch.cat([vx.unsqueeze(-1), vy.unsqueeze(-1)], dim=-1) # (B, M, M, T, 2)

            # also need to zero out the symmetric entries as we apply anothe matrix transformation
            if speed_repr in ['rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']:
                # (B, M1, M2, T) -> (B, M2, M1, T)
                avail_perm = avail.permute(0, 2, 1, 3)
                avail = avail * avail_perm

                hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, pos_original, avail.unsqueeze(-1)], dim=-1)
            elif speed_repr in ['rel_vel_new', 'rel_vel_new_new']:
                hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, pos_original, avail.unsqueeze(-1)], dim=-1)
            else:
                hist_in = torch.cat([pos, hvec, vvec, lw, rel_d, rel_d_lw, rel_t_to_col, avail.unsqueeze(-1)], dim=-1)
        elif speed_repr == 'abs_speed':
            # combine to get full input
            hist_in = torch.cat([pos, hvec, speed, lw, avail.unsqueeze(-1)], dim=-1)
        else:
            raise ValueError('Unknown speed representation: {}'.format(speed_repr))
        # zero out values we don't have data for
        hist_in[~avail] = 0.0
        if torch.isnan(hist_in).any():
            hist_in = torch.where(torch.isnan(hist_in), torch.zeros_like(hist_in), hist_in)

            # log nan values
            print('torch.where(torch.isnan(hist_in))', torch.where(torch.isnan(hist_in)))
            # with open('nan_log.txt', 'a') as f:
            #     f.write('torch.where(torch.isnan(hist_in)): '+str(torch.where(torch.isnan(hist_in))))
            #     f.write('torch.where(torch.isnan(pos)): '+str(torch.where(torch.isnan(pos))))
            #     f.write('torch.where(torch.isnan(rel_d)): '+str(torch.where(torch.isnan(rel_d))))
            #     f.write('torch.where(torch.isnan(lw_avg_half_sum)): '+str(torch.where(torch.isnan(lw_avg_half_sum))))
                    

        return hist_in

    def get_aux_info(self, data_batch, include_class_free_cond=False):
        # if torch.isnan(data_batch["history_positions"]).any():
        #     print('0 torch.where(torch.isnan(data_batch["history_positions"]))', torch.where(torch.isnan(data_batch["history_positions"])))
        #     raise ValueError('NaN in data_batch["history_positions"]')

        # current ego state (B, 4)
        # Note: [x, y, vel, yaw] (unicycle), always need this for rolling out actions
        if self._dynamics_type is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        else:
            curr_states = None
        

        if self.single_cond_feat:
            cond_feat_in_list = []
            non_cond_feat_in_list = []

        # 1.agents_hist (B, M, T, K_vehicle)
        agents_hist = self.prepare_scene_agent_hist(data_batch["history_positions"], data_batch["history_yaws"], data_batch["history_speeds"], data_batch["extent"], data_batch["history_availabilities"], self.agent_hist_norm_info)
        if include_class_free_cond:
            non_cond_avail = torch.zeros_like(data_batch["history_speeds"]).to(torch.bool) # BxT
            agents_hist_non_cond = self.prepare_scene_agent_hist(data_batch["history_positions"], data_batch["history_yaws"], data_batch["history_speeds"], data_batch["extent"], non_cond_avail, self.agent_hist_norm_info)

        if self.agent_hist_embed_method == 'mlp':
            # encode agents_hist with MLP (only support vectorized version)
            B, M, _, _ = agents_hist.shape
            # (B, M, T, 8) -> (B*M, K_d)
            agent_hist_feat = self.agent_hist_encoder.scene_forward(agents_hist)
            # (B*M, K_d) -> (B, M, K_d) -> (B, M, 1, K_d)
            agent_hist_feat = agent_hist_feat.reshape((B, M, -1)).unsqueeze(-2)
            if self.single_cond_feat:
                cond_feat_in_list.append(agent_hist_feat)
            if include_class_free_cond:
                # (B, M, T, 8) -> (B*M, K_d)
                agent_hist_feat_non_cond = self.agent_hist_encoder.scene_forward(agents_hist_non_cond)
                # (B*M, K_d) -> (B, M, K_d) -> (B, M, 1, K_d)
                agent_hist_feat_non_cond = agent_hist_feat_non_cond.reshape((B, M, -1)).unsqueeze(-2)
                if self.single_cond_feat:
                    non_cond_feat_in_list.append(agent_hist_feat_non_cond)
        elif self.agent_hist_embed_method in ['transformer', 'concat']:
            # do nothing and directly pass the vector forms (only support vectorized version)
            # (B, M, T, K_vehicle)
            agent_hist_feat = agents_hist
            if include_class_free_cond:
                # (B, M, T, K_vehicle)
                agent_hist_feat_non_cond = agents_hist_non_cond
        else:
            raise NotImplementedError

        # 2.neighbor trajectory encoding (B, M, Q, T, K_vehicle)
        if self.neigh_hist_embed_method == 'mlp':
            neighbor_hist = self.prepare_scene_agent_hist(data_batch["all_other_agents_history_positions"], data_batch["all_other_agents_history_yaws"], data_batch["all_other_agents_history_speeds"], data_batch["all_other_agents_extents"], data_batch["all_other_agents_history_availabilities"], self.neighbor_hist_norm_info)

            # (B, M, Q, T, K_vehicle) -> (B*M, K_d)
            neighbor_hist_feat = self.neighbor_hist_encoder.scene_forward(neighbor_hist, data_batch["all_other_agents_history_availabilities"])
            # (B*M, K_d) -> (B*M, K_d) -> (B, M, 1, K_d)
            neighbor_hist_feat = neighbor_hist_feat.reshape((B, M, -1)).unsqueeze(-2)
            if self.single_cond_feat:
                cond_feat_in_list.append(neighbor_hist_feat)
            if include_class_free_cond:
                # make all agents zero availability
                non_cond_neighbor_avail = torch.zeros_like(data_batch["all_other_agents_history_speeds"]).to(torch.bool) 
                neighbor_hist_non_cond = self.prepare_scene_agent_hist(data_batch["all_other_agents_history_positions"], data_batch["all_other_agents_history_yaws"], data_batch["all_other_agents_history_speeds"], data_batch["all_other_agents_extents"], non_cond_neighbor_avail, self.neighbor_hist_norm_info)

                neighbor_hist_feat_non_cond = self.neighbor_hist_encoder(neighbor_hist_non_cond, non_cond_neighbor_avail)
                # (B*M, K_d) -> (B, M, K_d) -> (B, M, 1, K_d)
                neighbor_hist_feat_non_cond = neighbor_hist_feat_non_cond.reshape((B, M, -1)).unsqueeze(-2)
                if self.single_cond_feat:
                    non_cond_feat_in_list.append(neighbor_hist_feat_non_cond)
        elif self.neigh_hist_embed_method in ['interaction_edge', 'interaction_edge_and_input']:
            # (B, M, M, T_hist, K_vehicle), (B, M, M, T_hist, K_vehicle)
            neighbor_hist_feat, neighbor_hist_feat_non_cond = self.get_neighbor_history_relative_states(data_batch, include_class_free_cond)

        elif self.neigh_hist_embed_method is None:
            neighbor_hist_feat, neighbor_hist_feat_non_cond = None, None
        else:
            raise NotImplementedError(f'{self.neigh_hist_embed_method}')


        # 3.map encoding
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            # encode map with CNN (only support rasterized version)
            # rasterized map (B, M, C, H, W)
            image_batch = data_batch["image"]
            B, M, C, H, W = image_batch.shape
            # (B, M, C, H, W) -> (B*M, C, H, W)
            image_batch = image_batch.reshape(B*M, C, H, W)
            # (B*M, C, H, W) -> (B*M, K_d)
            map_global_feat, map_grid_feat = self.map_encoder(image_batch)

            if map_global_feat is not None:
                # (B*M, K_d) -> (B, M, 1, K_d)
                map_feat = map_global_feat.reshape(B, M, -1).unsqueeze(-2)
            else:
                map_feat = None

            if self.single_cond_feat:
                cond_feat_in_list.append(map_feat)

            if self.use_map_feat_grid and self.map_encoder is not None:
                raster_from_agent = data_batch["raster_from_agent"]
                map_grid_feat = map_grid_feat.reshape(B, M, *map_grid_feat.shape[1:])

            if include_class_free_cond:
                image_non_cond = torch.ones_like(image_batch) * self.cond_fill_value
                # (B*M, C, H, W) -> (B*M, K_d)
                map_global_feat_non_cond, map_grid_feat_non_cond = self.map_encoder(image_non_cond)
                # (B*M, K_d) -> (B, M, 1, K_d)
                map_feat_non_cond = map_global_feat_non_cond.reshape(B, M, -1).unsqueeze(-2)
                if self.single_cond_feat:
                    non_cond_feat_in_list.append(map_feat_non_cond)

        elif self.map_embed_method == 'transformer':
            # do nothing and directly pass the vector forms (only support vector version)
            # vectorized map (B, M, S_seg, S_p, K_map)
            map_feat = data_batch["extras"]["closest_lane_point"]
            # print('map_feat.shape', map_feat.shape)
            if include_class_free_cond:
                map_feat_non_cond = torch.ones_like(map_feat) * self.cond_fill_value
        else:
            raise NotImplementedError


        if self.single_cond_feat:
            cond_feat_in = torch.cat(cond_feat_in_list, dim=-1)
            B, M, _, _ = cond_feat_in.shape
            # (B, M, 1, K_d*3) -> (B, M, K_d)
            cond_feat = self.process_cond_mlp(cond_feat_in.reshape(B*M, -1)).reshape(B, M, -1)
            if include_class_free_cond:
                non_cond_feat_in = torch.cat(non_cond_feat_in_list, dim=-1)
                non_cond_feat = self.process_cond_mlp(non_cond_feat_in.reshape(B*M, -1)).reshape(B, M, -1)
        else:
            cond_feat = None


        aux_info = {
            'curr_states': curr_states,
            'agent_hist_feat': agent_hist_feat,
            'map_feat': map_feat,
            'cond_feat': cond_feat,
            'neighbor_hist_feat': neighbor_hist_feat,
        }

        # do not have non_cond_curr_states since it is not used in the conditioning
        if include_class_free_cond:
            aux_info['agent_hist_feat_non_cond'] = agent_hist_feat_non_cond
            aux_info['map_feat_non_cond'] = map_feat_non_cond
            aux_info['non_cond_feat'] = non_cond_feat
            aux_info['neighbor_hist_feat_non_cond'] = neighbor_hist_feat_non_cond
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                aux_info['map_grid_feat'] = map_grid_feat
                if include_class_free_cond:
                    aux_info['map_grid_feat_non_cond'] = map_grid_feat_non_cond
                aux_info['raster_from_agent'] = raster_from_agent

        return aux_info
    
    def get_neighbor_relative_states(self, relative_positions, relative_speeds, relative_yaws, data_batch_agent_from_world, data_batch_world_from_agent, data_batch_yaw, data_batch_extent):
        BN, M, _, _ = relative_positions.shape
        B = data_batch_agent_from_world.shape[0]
        N = int(BN // B)

        # [M, M]
        nb_idx = torch.arange(M).unsqueeze(0).repeat(M, 1)

        all_other_agents_relative_positions_list = []
        all_other_agents_relative_yaws_list = []
        all_other_agents_relative_speeds_list = []
        all_other_agents_extent_list = []

        # get relative states
        for k in range(BN):
            i = int(k // N)
            agent_from_world = data_batch_agent_from_world[i]
            world_from_agent = data_batch_world_from_agent[i]

            all_other_agents_relative_positions_list_sub = []
            all_other_agents_relative_yaws_list_sub = []
            all_other_agents_relative_speeds_list_sub = []
            all_other_agents_extent_list_sub = []

            for j in range(M):
                chosen_neigh_inds = nb_idx[j][nb_idx[j]>=0].tolist()

                # (Q. 3. 3)
                center_from_world = agent_from_world[j]
                world_from_neigh = world_from_agent[chosen_neigh_inds]
                center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh

                fut_neigh_pos_b_sub = relative_positions[k][chosen_neigh_inds]
                fut_neigh_yaw_b_sub = relative_yaws[k][chosen_neigh_inds]

                all_other_agents_relative_positions_list_sub.append(GeoUtils.transform_points_tensor(fut_neigh_pos_b_sub,center_from_neigh))
                all_other_agents_relative_yaws_list_sub.append(fut_neigh_yaw_b_sub+data_batch_yaw[i][chosen_neigh_inds][:,None,None]-data_batch_yaw[i][j])
                all_other_agents_relative_speeds_list_sub.append(relative_speeds[k][chosen_neigh_inds])
                all_other_agents_extent_list_sub.append(data_batch_extent[i][chosen_neigh_inds])

            all_other_agents_relative_positions_list.append(pad_sequence(all_other_agents_relative_positions_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_relative_yaws_list.append(pad_sequence(all_other_agents_relative_yaws_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_relative_speeds_list.append(pad_sequence(all_other_agents_relative_speeds_list_sub, batch_first=True, padding_value=np.nan))
            all_other_agents_extent_list.append(pad_sequence(all_other_agents_extent_list_sub, batch_first=True, padding_value=0))

        max_second_dim = max(a.size(1) for a in all_other_agents_relative_positions_list)

        all_other_agents_relative_positions = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_positions_list], dim=0)
        all_other_agents_relative_yaws = angle_wrap(torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_yaws_list], dim=0))
        all_other_agents_relative_speeds = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_relative_speeds_list], dim=0)
        all_other_agents_extents = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_extent_list], dim=0)

        return all_other_agents_relative_positions, all_other_agents_relative_yaws, all_other_agents_relative_speeds, all_other_agents_extents

    def get_neighbor_future_relative_states(self, x: torch.tensor, aux_info: dict, data_batch: dict) -> dict:
        '''
        -param x: (B*N, M, T, transition_dim)
        -return: aux_info with 
            'neighbor_fut_feat': (B*N, M, M, T_fut, K_neigh)
        '''
        
        T_fut = data_batch['target_positions'].shape[-2]

        if self.use_last_hist_step:
            # (B*N, M, M, T_hist, K_neigh) -> (B*N, M, M, K_neigh) -> (B*N, M, M, 1, K_neigh) -> (B*N, M, M, T_fut, K_neigh) (x,y,cos,sin,speed,l,w,avail) or (x,y,cos,sin,vx,vy,l,w,avail)
            neighbor_fut = aux_info['neighbor_hist_feat'][...,-1,:].unsqueeze(-2).repeat(1,1,1,T_fut,1)
            # Set relative velocity to 0 as we assume they are relatively stationary
            # if self.interaction_edge_speed_repr == 'rel_vel':
            #     neighbor_fut[...,4:6] = 0.0
        elif self.use_noisy_fut_edge:
            # TBD: support other transition_dim
            assert x.shape[-1] == 6, f'currently this mode only support x.shape[-1] being 6, but got {x.shape[-1]}'
            # descale to agent coords (x,y,speed,yaw,accel,yawvel)
            x_descaled = self.descale_traj(x)
            future_positions = x_descaled[...,:2]
            future_speeds = x_descaled[...,2]
            future_yaws = x_descaled[...,3:4]

            all_other_agents_future_positions, all_other_agents_future_yaws, all_other_agents_future_speeds, all_other_agents_extents = self.get_neighbor_relative_states(future_positions, future_speeds, future_yaws, data_batch['agent_from_world'], data_batch['world_from_agent'], data_batch['yaw'], data_batch["extent"])

            # future availabilities depend on the current time step availability
            # (B*N, M, M, T_hist, k_vehicle) -> (B*N, M, M) -> (B*N, M, M, 1) -> (B*N, M, M, T_fut)
            all_other_agents_future_availabilities = aux_info['neighbor_hist_feat'][...,-1,-1].bool().unsqueeze(-1).repeat(1,1,1,T_fut)

            neighbor_fut = self.prepare_scene_agent_hist(all_other_agents_future_positions, all_other_agents_future_yaws, all_other_agents_future_speeds, all_other_agents_extents, all_other_agents_future_availabilities, self.neighbor_fut_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)
        elif self.use_const_speed_edge or x is None:
            # (B*N, M, M, T_hist, K_neigh) -> (B*N, M, M, K_neigh) -> (B*N, M, M, 1, K_neigh) -> (B*N, M, M, T_fut, K_neigh) (x,y,cos,sin,speed,l,w,avail) or (x,y,cos,sin,vx,vy,l,w,avail)
            neighbor_fut = aux_info['neighbor_hist_feat'][...,-1,:].unsqueeze(-2).repeat(1,1,1,T_fut,1)
            if self.interaction_edge_speed_repr in ['rel_vel', 'rel_vel_new', 'rel_vel_per_step', 'rel_vel_new_new']:
                dt = 0.1
                # Get a time weight with shape (T_fut) from 0.1 to 0.1*T_fut
                time = torch.arange(1, T_fut+1, dtype=torch.float32, device=neighbor_fut.device) * dt
                # (T_fut) -> (1, 1, 1, T_fut)
                time = time.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                neighbor_fut[...,:2] = neighbor_fut[...,:2] + neighbor_fut[...,4:6] * time
            else:
                raise NotImplementedError

        
        aux_info['neighbor_fut_feat'] = neighbor_fut
        aux_info['neighbor_feat'] = torch.cat([aux_info['neighbor_hist_feat'], aux_info['neighbor_fut_feat']], dim=-2)

        if 'agent_hist_feat_non_cond' in aux_info:
            if self.use_last_hist_step:
                # TBD: if cond_fill_value is not zero, this needs to be changed
                neighbor_fut_non_cond = torch.zeros_like(neighbor_fut)
            elif self.use_noisy_fut_edge:
                all_other_agents_future_availabilities_non_cond = torch.zeros_like(all_other_agents_future_availabilities, dtype=torch.bool, device=all_other_agents_future_availabilities.device)
                neighbor_fut_non_cond = self.prepare_scene_agent_hist(all_other_agents_future_positions, all_other_agents_future_yaws, all_other_agents_future_speeds, all_other_agents_extents, all_other_agents_future_availabilities_non_cond, self.neighbor_fut_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)
            elif self.use_const_speed_edge:
                pass
            
            aux_info['neighbor_fut_feat_non_cond'] = neighbor_fut_non_cond
            aux_info['neighbor_feat_non_cond'] = torch.cat([aux_info['neighbor_hist_feat_non_cond'], aux_info['neighbor_fut_feat_non_cond']], dim=-2)

        return aux_info
        
    def get_neighbor_history_relative_states(self, data_batch: dict, include_class_free_cond: bool) -> dict:
        '''
        get the neighbor history relative states (only need once per data_batch). We do this because fields like all_other_agents_history_positions in data_batch may not include all the agents controlled and may include agents not controlled.

        - output neighbor_hist: (B, M, M, T_hist, K_vehicle)
        - output neighbor_hist_non_cond: (B, M, M, T_hist, K_vehicle)
        '''
        M = data_batch['history_positions'].shape[1]

        all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents = self.get_neighbor_relative_states(data_batch['history_positions'], data_batch['history_speeds'], data_batch['history_yaws'], data_batch['agent_from_world'], data_batch['world_from_agent'], data_batch['yaw'], data_batch["extent"])

        # (B, M, T_hist) -> (B, 1, M, T_hist) -> (B, M, M, T_hist)
        all_other_agents_history_availabilities = data_batch["history_availabilities"].unsqueeze(1).repeat(1,M,1,1)

        neighbor_hist = self.prepare_scene_agent_hist(all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents, all_other_agents_history_availabilities, self.neighbor_fut_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)

        # # (B, M, M, T_hist, K_vehicle) -> (B, M, M, T_hist, K_neigh)
        # neighbor_hist_feat = self.neighbor_hist_encoder(neighbor_hist)

        if include_class_free_cond:
            all_other_agents_history_availabilities_non_cond = torch.zeros_like(all_other_agents_history_availabilities, dtype=torch.bool, device=all_other_agents_history_availabilities.device)
            neighbor_hist_non_cond = self.prepare_scene_agent_hist(all_other_agents_history_positions, all_other_agents_history_yaws, all_other_agents_history_speeds, all_other_agents_extents, all_other_agents_history_availabilities_non_cond, self.neighbor_hist_norm_info, scale=self.normalize_rel_states, speed_repr=self.interaction_edge_speed_repr)
            
            # # (B, M, M, T_fut, K_vehicle) -> (B, M, M, T_fut, K_neigh)
            # neighbor_hist_feat_non_cond = self.neighbor_hist_encoder(neighbor_hist_non_cond)
        else:
            # neighbor_hist_feat_non_cond = None
            neighbor_hist_non_cond = None

        return neighbor_hist, neighbor_hist_non_cond

    def query_neigh_feats(self, neighbor_fut):
        '''
        -params neighbor_fut : (B*N/B, M, M, T_fut, K_neigh)
        -output neighbor_fut_feat : (B*N/B, M, T_fut, K_d)
        '''
        # (B*N, M, M, T_fut, K_neigh)
        BN, M, _, T, _ = neighbor_fut.shape

        # (B*N, M, M, T_fut, K_neigh) -> (B*N, M, T_fut, M, K_neigh)
        neighbor_fut = neighbor_fut.permute(0, 1, 3, 2, 4)

        # --- Sort neighbors according to distances ---
        
        # Assign large values to invalid neighbors
        invalid_mask = neighbor_fut[..., -1] == 0
        expanded_mask = invalid_mask.unsqueeze(-1).expand(-1, -1, -1, -1, 2)
        neighbor_fut[..., :2].masked_fill_(expanded_mask, torch.inf)
        
        # sort neighbor indices
        norms = torch.norm(neighbor_fut[..., :2], dim=-1)
        _, sorted_indices = torch.sort(norms, dim=-1) 
        # assign 0 to neighbors far away
        neighbor_fut[norms > self.social_attn_radius] = 0
        neighbor_fut = torch.gather(neighbor_fut, 3, sorted_indices.unsqueeze(-1).expand_as(neighbor_fut))
        # # remove self, (B*N, M, T_fut, M, K_neigh) -> (B*N, M, T_fut, M-1, K_neigh)
        # neighbor_fut = neighbor_fut[...,1:,:]

        # --- Encoder neighbors using MLP ---
        # (B*N, M, T_fut, M, K_neigh) -> (B*N*M*T_fut, M, K_neigh)
        neighbor_fut = neighbor_fut.view(BN*M*T, *neighbor_fut.shape[-2:])
        # (B*N*M*T_fut, M, K_neigh) -> (B*N*M*T_fut, K_d)
        neighbor_fut_feat = self.neighbor_fut_encoder(neighbor_fut)
        # (B*N*M*T_fut, K_d) -> (B*N, M, T_fut, K_d)
        neighbor_fut_feat = neighbor_fut_feat.view(BN, M, T, -1)
        return neighbor_fut_feat
    
    def query_map_feats(self, x, map_grid_feat, raster_from_agent):
        '''
        - x : (B, M, T, D)
        - map_grid_feat : (B, M, C, H, W)
        - raster_from_agent: (B, M, 3, 3)

        - output feats_out : (B, M, T, C)
        '''

        B, M, T, _ = x.shape
        x = x.reshape((B*M, *x.shape[2:]))
        map_grid_feat = map_grid_feat.reshape((B*M, *map_grid_feat.shape[2:]))
        raster_from_agent = raster_from_agent.reshape((B*M, *raster_from_agent.shape[2:]))

        Hfeat, Wfeat = map_grid_feat.shape[-2:]

        # unscale to agent coords
        pos_traj = self.descale_traj(x.detach())[...,:2]
        # convert to raster frame
        raster_pos_traj = GeoUtils.transform_points_tensor(pos_traj, raster_from_agent)

        # scale to the feature map size
        _, H, W = self.input_image_shape
        xscale = Wfeat / W
        yscale = Hfeat / H
        raster_pos_traj[...,0] = raster_pos_traj[...,0] * xscale
        raster_pos_traj[...,1] = raster_pos_traj[...,1] * yscale

        # interpolate into feature grid
        feats_out = query_feature_grid(
                            raster_pos_traj,
                            map_grid_feat
                            )
        feats_out = feats_out.reshape((B, M, T, -1))
        return feats_out

    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        '''
        Extract state and(or) action from the data_batch from data_batch
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            data_batch: dict
        Output:
            x: (batch_size, [num_agents], num_steps, len(chosen_inds)).
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        # NOTE: for predicted agent, history and future with always be fully available
        traj_state = torch.cat(
                (data_batch["target_positions"][..., :self.horizon, :], data_batch["target_yaws"][..., :self.horizon, :]), dim=-1)
        traj_state_and_action = convert_state_to_state_and_action(traj_state, data_batch["curr_speed"], self.dt)

        return traj_state_and_action[..., chosen_inds]
    
    def convert_action_to_state_and_action(self, x_out, curr_states, scaled_input=True, descaled_output=False):
        '''
        Apply dynamics on input action trajectory to get state+action trajectory
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            x_out: (B, (M), T, 2). scaled action trajectory
            curr_states: (B, (M), 4). current state
        Output:
            x_out: (B, (M), T, 6). scaled state+action trajectory
        '''

        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel',
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        return x_out_all

    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        scale the trajectory from original scale to standard normal distribution
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            - target_traj_orig: (B, (M), T, D)
        Output:
            - target_traj: (B, (M), T, D)
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        # TODO make these a buffer so they're put on the device automatically
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        scale back the trajectory from standard normal distribution to original scale
        Note:
            Support both agent-centric and scene-centric (extra dimension for the inputs).
        Input:
            - target_traj_orig: (B, (M), T, D)
        Output:
            - target_traj: (B, (M), T, D)
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add

        return target_traj

    
    def forward(self, data_batch: Dict[str, torch.Tensor], plan=None, num_samp=1,
                    return_diffusion=False,
                    return_guidance_losses=False,
                    class_free_guide_w=0.0,
                    apply_guidance=True,
                    guide_clean=False,
                    global_t=0, stationary_mask=None, mode='testing') -> Dict[str, torch.Tensor]:
        # update stationary_mask at the beginnig of each rollout
        if global_t == 0:
            self.stationary_mask = stationary_mask
        self.global_t = global_t
        use_class_free_guide = class_free_guide_w != 0.0
        aux_info = self.get_aux_info(data_batch, use_class_free_guide)
        
        cond_samp_out = self.conditional_sample(data_batch, 
                                                horizon=None,
                                                aux_info=aux_info,
                                                return_diffusion=return_diffusion,
                                                return_guidance_losses=return_guidance_losses,
                                                num_samp=num_samp,
                                                verbose=False, class_free_guide_w=class_free_guide_w,
                                                apply_guidance=apply_guidance,
                                                guide_clean=guide_clean, mode=mode)
        traj_init = cond_samp_out['pred_traj']
        attn_weights = cond_samp_out['attn_weights']
        diff_init = guide_losses = None
        if return_diffusion:
            diff_init = cond_samp_out['diffusion']
        if return_guidance_losses:
            guide_losses = cond_samp_out['guide_losses']

        traj = self.descale_traj(traj_init)
        if diff_init is not None:
            diff_steps = self.descale_traj(diff_init)
        else:
            diff_steps = None
                
        # CHANGE: keep the length same as the input observation        
        # traj = traj[..., :target_traj.size()[1], :]

        if self.diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:
            traj_pred = traj[..., [0, 1, 3]]
        elif self.diffuser_input_mode == 'action':
            traj = self.convert_action_to_state_and_action(traj_init, aux_info['curr_states'], scaled_input=False)
            traj_pred = traj[..., [0, 1, 3]]
        elif self.diffuser_input_mode == 'state':
            pass
        else:
            raise

        pred_positions = traj_pred[..., :2]
        pred_yaws = traj_pred[..., 2:3]
        
        out_dict = {
            # "trajectories": traj,
            # include trajectories in prediction so they can be leveraged by wrapper
            "predictions": {"positions": pred_positions, "yaws": pred_yaws, "trajectories": traj, "attn_weights": attn_weights},
        }
        if diff_steps is not None:
            out_dict["predictions"]["diffusion_steps"] = diff_steps # (1, B*N, M, T, transition_dim)
        if guide_losses is not None:
            out_dict["predictions"]["guide_losses"] = guide_losses
        if self.dyn is not None:
            out_dict["curr_states"] = aux_info['curr_states']
        return out_dict

    def compute_losses(self, data_batch):
        aux_info = self.get_aux_info(data_batch)
        target_traj = self.get_state_and_action_from_data_batch(data_batch)
        
        if self.use_reconstructed_state and self.diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']:
            target_traj = self.convert_action_to_state_and_action(target_traj[..., [4, 5]], aux_info['curr_states'], scaled_input=False)        


        x = self.scale_traj(target_traj)
                
        diffusion_loss, info = self.loss(x, data_batch, aux_info=aux_info)
        losses = OrderedDict(
            diffusion_loss=diffusion_loss,
        )
        if 'collision_loss' in info:
            losses['collision_loss'] = info['collision_loss']
        if 'offroad_loss' in info:
            losses['offroad_loss'] = info['offroad_loss']
        if 'history_reconstruction_loss' in info:
            losses['history_reconstruction_loss'] = info['history_reconstruction_loss']
        
        return losses

    def get_loss_weights(self, action_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # CHANGE: comment out since we only consider action for now
        # ## set loss coefficients for dimensions of observation
        # if weights_dict is None: weights_dict = {}
        # for ind, w in weights_dict.to_list():
        #     dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        ## manually set a0 weight
        loss_weights[0, -self.action_dim:] = action_weight

        if self.action_loss_only:
            loss_weights = loss_weights[:, -2:]
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise, force_noise=False):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon or force_noise:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x_start):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return x_start
        else:
            # TODO don't need device stuff if buffer
            return (
                extract(self.sqrt_recip_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                extract(self.sqrt_alphas_over_one_minus_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_start
            )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, aux_info={}, class_free_guide_w=0.0, data_batch=None, x_clean_model_out=None):

        t_inp = t
            
        # model_prediction = self.model(x, cond, t_inp, aux_info)
        x_model_in = x
        if self.map_embed_method in ['cnn', 'cnn_local_patch']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                # get features from map and append to the trajectory
                map_feat_traj = self.query_map_feats(x.detach().clone(),
                                                    aux_info['map_grid_feat'],
                                                    aux_info['raster_from_agent'])
                x_model_in = torch.cat([x_model_in, map_feat_traj], dim=-1)

        if self.neigh_hist_embed_method in ['interaction_edge', 'interaction_edge_and_input']:
            # fill in noise at the first denoising step
            if self.use_noisy_fut_edge:
                x_edge = x.detach().clone()
            else:
                if x_clean_model_out is None:
                    x_edge = None
                    # x_edge = x_model_in.clone().detach()
                else:
                    x_edge = self.convert_action_to_state_and_action(x_clean_model_out.detach(), aux_info['curr_states'])
            aux_info = self.get_neighbor_future_relative_states(x_edge, aux_info, data_batch)

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                # (B*N, M, M, T_fut, K_neigh) -> (B*N, M, T_fut, K_d)
                neighbor_fut_feat = self.query_neigh_feats(aux_info['neighbor_fut_feat'])
                # (B*N, M, T_fut, D) + (B*N, M, T_fut, K_d) -> (B*N, M, T_fut, D+K_d)
                x_model_in = torch.cat([x_model_in, neighbor_fut_feat], dim=-1)

        # visualzie attention weigths for the last denoising step of each rollout step
        if t_inp[0] == 0:
            # aux_info['attn_name'] = str(self.global_t)+'_'+str(t_inp[0].detach().cpu().item())
            aux_info['attn_name'] = ""
        else:
            aux_info['attn_name'] = ""

        model_prediction, info = self.model(x_model_in, aux_info, t_inp)
        if self.agent_hist_embed_method == 'concat':
            model_prediction = model_prediction[..., -self.horizon:,:]

        if self.diffuser_input_mode == 'state_and_action':
            x_tmp = x[..., 4:].detach()
        else:
            x_tmp = x.detach()

        if class_free_guide_w != 0.0:
            # now run non-cond once
            # aux_info_non_cond = {k : v for k, v in aux_info.items() if k not in ['map_feat', 'map_feat_non_cond']}
            # aux_info_non_cond['map_feat'] = aux_info['map_feat_non_cond']
            # model_non_cond_prediction = self.model(x, cond, t_inp, aux_info_non_cond)
            x_model_non_cond_in = x
            if self.use_map_feat_grid and self.map_encoder is not None:
                # get features from map and append to the trajectory
                map_feat_traj = self.query_map_feats(x_model_non_cond_in.detach().clone(),
                                                    aux_info['map_grid_feat_non_cond'],
                                                    aux_info['raster_from_agent'])
                x_model_non_cond_in = torch.cat([x_model_non_cond_in, map_feat_traj], dim=-1)

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                # (B*N, M, T_fut, D) + (B*N, M, T_fut, K_d) -> (B*N, M, T_fut, D+K_d)
                x_model_non_cond_in = torch.cat([x_model_non_cond_in, torch.zeros_like(neighbor_fut_feat)], dim=-1)
            model_non_cond_prediction, _ = self.model(x_model_non_cond_in, aux_info, t_inp, use_cond=False)
            if self.agent_hist_embed_method == 'concat':
                model_non_cond_prediction = model_non_cond_prediction[..., -self.horizon:,:]


            # and combine to get actual model prediction (in noise space as in original paper)
            model_pred_noise = self.predict_noise_from_start(x_tmp, t=t, x_start=model_prediction)
            model_non_cond_pred_noise = self.predict_noise_from_start(x_tmp, t=t, x_start=model_non_cond_prediction)

            # print(torch.sum(model_pred_noise - model_non_cond_pred_noise))

            class_free_guide_noise = (1 + class_free_guide_w)*model_pred_noise - class_free_guide_w*model_non_cond_pred_noise

            model_prediction = self.predict_start_from_noise(x_tmp, t=t, noise=class_free_guide_noise, force_noise=True)


        x_recon = self.predict_start_from_noise(x_tmp, t=t, noise=model_prediction)

        # if self.clip_denoised:
        #     x_recon.clamp_(-1., 1.)
        # else:
        #     assert RuntimeError()
        
        if self.disable_control_on_stationary:
            if self.diffuser_input_mode == 'state_and_action':
                inds = [4, 5]
            else:
                inds = self.default_chosen_inds
            x_recon_stationary = x_recon[self.stationary_mask]
            x_recon_stationary = self.descale_traj(x_recon_stationary, inds)
            x_recon_stationary[...] = 0
            x_recon_stationary = self.scale_traj(x_recon_stationary, inds)
            x_recon[self.stationary_mask] = x_recon_stationary

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_tmp, t=t)
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x_tmp, t), model_prediction, info

    def state_action_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))

        x_all = self.convert_action_to_state_and_action(x_guidance, expand_states, scaled_input=transform_params['scaled_input'], descaled_output=transform_params['scaled_output'])
        return x_all

    def state_action_no_dyn_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))

        x_all = self.convert_action_to_state_and_action(x_guidance[..., -2:], expand_states, scaled_input=transform_params['scaled_input'], descaled_output=transform_params['scaled_output'])
        return x_all
        
    def state_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        x_state = self.descale_traj(x_guidance, [0, 1, 3])
        curr_speed = data_batch['curr_speed']
        expand_speed = curr_speed.unsqueeze(1).expand((bsize, num_samp)).reshape((bsize*num_samp))
        x_all = convert_state_to_state_and_action(x_state, expand_speed, self.dt)
        return x_all

    @torch.no_grad()
    def p_sample(self, x, t, data_batch, aux_info={}, num_samp=1, class_free_guide_w=0.0, apply_guidance=True, guide_clean=False, eval_final_guide_loss=False, x_clean_model_out=None, data_batch_for_guidance={}):
        b, *_, device = *x.shape, x.device
        with_func = torch.no_grad
        if self.current_perturbation_guidance.current_guidance is not None and apply_guidance and guide_clean == "video_diff":
            # will need to take grad wrt noisy
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad
        with with_func():
            # get prior mean and variance for next step
            model_mean, _, model_log_variance, q_posterior_in, x_clean_model_out, info = self.p_mean_variance(x=x, t=t, aux_info=aux_info, class_free_guide_w=class_free_guide_w, data_batch=data_batch, x_clean_model_out=x_clean_model_out)
        sigma = (0.5 * model_log_variance).exp()
        
        # no noise when t == 0
        #       i.e. use the mean of the distribution predicted at the final step rather than sampling.
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        if self.current_perturbation_guidance.current_guidance is not None and apply_guidance and guide_clean:
            # want to guide the predicted clean traj from model, not the noisy one
            x_initial = q_posterior_in[0]
            return_grad_of = x
        else:
            x_initial = model_mean.clone().detach()
            return_grad_of = x_initial
            x_initial.requires_grad_()

        guide_losses = dict()
        x_guidance = None
        # consider intermediate and final guidance (for ablation) separately
        if apply_guidance and self.guidance_optimization_params is not None:
            if t[0] == 0:
                apply_guidance = self.apply_guidance_output
                if apply_guidance:
                    opt_params = self.final_step_opt_params
            else:
                apply_guidance = self.apply_guidance_intermediate
                if apply_guidance:
                    assert self.guidance_optimization_params['grad_steps'] > 0
                    perturb_th = self.guidance_optimization_params['perturb_th']
                    apply_guidance_output = self.apply_guidance_output
                    lr = self.guidance_optimization_params['lr']
                    
                    if perturb_th is not None:
                        # gradually decrease clip bounds from 1 to perturb_th
                        sig_scale = (torch.sigmoid(10 * t[0] / self.n_timesteps) - 1/2) * 2 
                        perturb_th = sig_scale * (4-perturb_th) + perturb_th
                        # print(t[0].item(), 'perturb_th', perturb_th)
                        if not apply_guidance_output:
                            perturb_th = perturb_th * nonzero_mask
                    else:
                        if t[0] == 0 and not apply_guidance_output:
                            perturb_th = nonzero_mask * sigma
                        else:
                            perturb_th = sigma

                    if lr is None:
                        lr = sigma
                    opt_params = deepcopy(self.guidance_optimization_params)
                    opt_params['lr'] = lr
                    opt_params['perturb_th'] = perturb_th
            
            if apply_guidance:
                if guide_clean == "video_diff":
                    x_guidance, guide_losses = self.current_perturbation_guidance.perturb_video_diffusion(x_initial, data_batch_for_guidance, opt_params, num_samp=num_samp, return_grad_of=return_grad_of)
                    # re-compute next step distribution with guided clean & noisy trajectories
                    x_guidance, _, _ = self.q_posterior(x_start=x_guidance, x_t=q_posterior_in[1], t=q_posterior_in[2])
                else:
                    x_guidance, guide_losses = self.current_perturbation_guidance.perturb(x_initial, data_batch_for_guidance, opt_params, num_samp=num_samp, return_grad_of=return_grad_of)
        
        # for filtration only
        if x_guidance is None:
            if self.current_perturbation_guidance.current_guidance is not None and eval_final_guide_loss:
                _, guide_losses = self.current_perturbation_guidance.compute_guidance_loss(x_initial, data_batch_for_guidance, num_samp=num_samp)
            x_guidance = x_initial

        # add noise
        noise = torch.randn_like(x_guidance)
        noise = nonzero_mask * sigma * noise
        x_out = x_guidance + noise

        # convert action to state+action
        if self.diffuser_input_mode == 'state_and_action':
            x_out = self.convert_action_to_state_and_action(x_out, aux_info['curr_states'])

        return x_out, guide_losses, x_clean_model_out, info
        
    @torch.no_grad()
    def p_sample_loop(self, shape, data_batch, num_samp, 
                    aux_info={}, 
                    verbose=True, 
                    return_diffusion=False,
                    return_guidance_losses=False,
                    class_free_guide_w=0.0,
                    apply_guidance=True,
                    guide_clean=False,
                    mode='testing'):
        '''
        shape: (5), batch_size, num_samp, num_agents, horizon, self.transition_dim
        '''
        # merge B and M to be compatible with guidance loss developed for agent-centric models
        data_batch_for_guidance = {}
        if apply_guidance:
            data_batch_for_guidance = extract_data_batch_for_guidance(data_batch, mode=mode)

        device = self.betas.device
        batch_size = shape[0]
        if self.current_perturbation_guidance.current_guidance is not None and not apply_guidance:
            print('DIFFUSER: Note, not using guidance during sampling, only evaluating guidance loss at very end...')

        # sample from base distribution
        x = torch.randn(shape, device=device) # (B, N, M, T(+T_hist), transition_dim)
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2) # (B*N, M, T(+T_hist), transition_dim)
        x_clean_model_out = None

        # (B, M, C) -> (B*N, M, C)
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)
        if return_diffusion: diffusion = [x] #(1, B*N, M, T, transition_dim)
        progress = Progress(self.n_timesteps) if verbose else Silent()

        steps = [i for i in reversed(range(0, self.n_timesteps, self.stride))]
        attn_weights = []
        for i in steps:
            # (B*N)
            timesteps = torch.full((batch_size*num_samp,), i, device=device, dtype=torch.long)
            x, guide_losses, x_clean_model_out, info = self.p_sample(x, timesteps, data_batch, aux_info=aux_info, num_samp=num_samp, class_free_guide_w=class_free_guide_w,
            apply_guidance=apply_guidance, guide_clean=guide_clean, eval_final_guide_loss=(i == steps[-1]), x_clean_model_out=x_clean_model_out, data_batch_for_guidance=data_batch_for_guidance)
            # apply hard constraints (overwrite waypoints at certain timesteps)
            if self.current_constraints is not None: # and i != steps[-1]: # TODO don't do it for last step?
                # TODO why isn't this working very well? And why is y upside down?
                # apply constraints expects traj in shape (B, N, T, D) and metric space
                x = self.descale_traj(x.reshape((shape[0], shape[1], shape[2], -1)))
                x = apply_constraints(x, data_batch['scene_index'], self.current_constraints)
                x = self.scale_traj(x.reshape((shape[0]*shape[1], shape[2], -1)))


            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
            if i == 0:
                attn_weights = info['attn_weights']

        progress.close()

        if any(guide_losses):
            print('===== GUIDANCE LOSSES ======')
            for k,v in guide_losses.items():
                if torch.isnan(v).any() or len(v) == 0:
                    v_mean = np.nan
                else:
                    v_mean = np.nanmean(v.cpu())
                print('%s: %.012f' % (k, v_mean))
        # (B*N, M, T(+T_hist), output_dim) -> (B, N, M, T(+T_hist), output_dim)
        x = TensorUtils.reshape_dimensions(x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))

        out_dict = {'pred_traj' : x, 'attn_weights': attn_weights}
        if return_guidance_losses:
            out_dict['guide_losses'] = guide_losses
        if return_diffusion:
            diffusion = [TensorUtils.reshape_dimensions(cur_diff, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))
                         for cur_diff in diffusion]
            out_dict['diffusion'] = torch.stack(diffusion, dim=3)

        return out_dict


    @torch.no_grad()
    def conditional_sample(self, data_batch, horizon=None, num_samp=1, class_free_guide_w=0.0, **kwargs):
        batch_size, num_agents = data_batch['history_positions'].size()[:2]
        horizon = horizon or self.horizon
        shape = (batch_size, num_samp, num_agents, horizon, self.transition_dim)

        return self.p_sample_loop(shape, data_batch, num_samp, class_free_guide_w=class_free_guide_w, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    
    def p_losses(self, x_start_init, t, data_batch, aux_info={}):
        noise_init = torch.randn_like(x_start_init)
        x_start = x_start_init
        noise = noise_init
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        t_inp = t

        # TODO should apply hard constraints here to x_noisy

        if self.diffuser_input_mode == 'state_and_action':
            x_action_noisy = x_noisy[..., [4, 5]]
            x_noisy = self.convert_action_to_state_and_action(x_action_noisy, aux_info['curr_states'])
        
        x_noisy_in = x_noisy
        if self.map_embed_method in ['cnn', 'cnn_and_input']:
            if self.use_map_feat_grid and self.map_encoder is not None:
                # get features from map and append to the trajectory
                map_feat_traj = self.query_map_feats(x_noisy.detach().clone(),
                                                    aux_info['map_grid_feat'],
                                                    aux_info['raster_from_agent'])
                x_noisy_in = torch.cat([x_noisy_in, map_feat_traj], dim=-1)

        if self.neigh_hist_embed_method in ['interaction_edge', 'interaction_edge_and_input']:
            if self.use_noisy_fut_edge:
                x_edge = x_noisy.detach().clone()
            else:
                x_edge = x_start_init.detach().clone()
            aux_info = self.get_neighbor_future_relative_states(x_edge, aux_info, data_batch)

            if self.neigh_hist_embed_method == 'interaction_edge_and_input':
                # (B*N, M, M, T_fut, K_neigh) -> (B*N, M, T_fut, K_d)
                neighbor_fut_feat = self.query_neigh_feats(aux_info['neighbor_fut_feat'])
                # (B*N, M, T_fut, D) + (B*N, M, T_fut, K_d) -> (B*N, M, T_fut, D+K_d)
                x_noisy_in = torch.cat([x_noisy_in, neighbor_fut_feat], dim=-1)

        noise, _ = self.model(x_noisy_in, aux_info, t_inp)
        if self.agent_hist_embed_method == 'concat':
            noise_hist = noise[..., :-self.horizon, :]
            noise_fut = noise[..., -self.horizon:, :]
        else:
            noise_fut = noise
        # TODO should apply hard constraints here to noise

        if self.diffuser_input_mode == 'state_and_action':
            # Note: if predict_eps, we convert noise into x_start for loss estimation since we need to apply forward dynamics
            x_recon_action = self.predict_start_from_noise(x_action_noisy, t=t, noise=noise_fut)
            x_recon = self.convert_action_to_state_and_action(x_recon_action, aux_info['curr_states'])
        else:
            x_recon = self.predict_start_from_noise(x_noisy_in, t=t, noise=noise_fut)
        


        if self.diffuser_input_mode in ['action', 'state_and_action', 'state_and_action_no_dyn'] and self.action_loss_only:
            x_recon_selected, x_start_selected = x_recon[..., -2:], x_start[..., -2:]
        else:
            x_recon_selected, x_start_selected = x_recon, x_start
        
        if self.use_target_availabilities:
            x_recon_selected = x_recon_selected * data_batch['target_availabilities'][..., :self.horizon].unsqueeze(-1)
            x_start_selected = x_start_selected * data_batch['target_availabilities'][..., :self.horizon].unsqueeze(-1)

        # currently other losses can only used if agent_centric
        if self.coordinate == 'agent_centric':
            # merge B and M to be compatible with guidance loss developed for agent-centric models
            data_batch_for_guidance = extract_data_batch_for_guidance(data_batch, mode='training')
            loss, info = self.loss_fn(x_recon_selected, x_start_selected, data_batch_for_guidance)
        else:
            loss, info = self.loss_fn(x_recon_selected, x_start_selected, {})

        if self.agent_hist_embed_method == 'concat':
            # TBD: currently only support predicting x0
            assert not self.predict_epsilon
            history_reconstruction_loss = self.estimate_history_reconstruction_loss(noise_hist, data_batch)
            info['history_reconstruction_loss'] = history_reconstruction_loss

        return loss, info

    def loss(self, x, data_batch, aux_info={}):
        batch_size = len(x)

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            
        return self.p_losses(x, t, data_batch, aux_info=aux_info)

    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return torch.tensor(list(reversed(out))).flatten().cpu().numpy()

    def estimate_history_reconstruction_loss(self, noise_hist, data_batch):
        '''
        noise_hist: [B, M, T_hist, 2]
        TBD: get history diffuser norm info to apply
        '''
        x_recon_action_hist = noise_hist

        # Get initial history state
        hist_start_state = torch.zeros(x_recon_action_hist.shape[0], x_recon_action_hist.shape[1], 4).to(x_recon_action_hist.device)  # [x, y, vel, yaw]
        hist_start_state[..., :2] = data_batch["history_positions"][..., 0, :]
        hist_start_state[..., 2] = data_batch["history_speeds"][..., 0]
        hist_start_state[..., 3] = data_batch["history_yaws"][..., 0, 0]

        # Get predicted state and action
        x_recon_hist = self.convert_action_to_state_and_action(x_recon_action_hist[...,1:,:], hist_start_state)

        # Get GT state and action
        traj_state_history = torch.cat(
            (data_batch["history_positions"][..., 1:, :], data_batch["history_yaws"][..., 1:, :]), dim=-1)
        vel_init = hist_start_state[..., 2]

        x_start_hist = convert_state_to_state_and_action(traj_state_history, vel_init, self.dt, data_type='torch')
        x_start_hist = self.scale_traj(x_start_hist)

        # Select relevant fields
        if self.diffuser_input_mode in ['action', 'state_and_action', 'state_and_action_no_dyn'] and self.action_loss_only:
            x_recon_hist_selected, x_start_hist_selected = x_recon_hist[..., -2:], x_start_hist[..., -2:]
        else:
            x_recon_hist_selected, x_start_hist_selected = x_recon_hist, x_start_hist

        # Apply availabilities
        if self.use_target_availabilities:
            x_recon_hist_selected = x_recon_hist_selected * data_batch['history_availabilities'][..., 1:].unsqueeze(-1)
            x_start_hist_selected = x_start_hist_selected * data_batch['history_availabilities'][..., 1:].unsqueeze(-1)
            # invalidate those that are not available at the first timestep in history
            x_recon_hist_selected[data_batch['history_availabilities'][..., 0]==False] = 0
            x_start_hist_selected[data_batch['history_availabilities'][..., 0]==False] = 0
        
        # Estimate loss
        history_loss = torch.mean(torch.nn.functional.mse_loss(x_recon_hist_selected, x_start_hist_selected, reduction='none'))

        return history_loss