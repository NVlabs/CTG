from typing import Dict, List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import tqdm

import tbsim.models.base_models as base_models
import tbsim.models.vaes as vaes
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.loss_utils import (
    trajectory_loss,
    log_normal
)
from tbsim.models.diffuser_helpers import AgentHistoryEncoder, NeighborHistoryEncoder, MapEncoder

from tbsim.utils.guidance_loss import DiffuserGuidance, verify_guidance_config_list

class ConditionEncoder(nn.Module):
    """Condition Encoder (x -> c) for CVAE"""
    def __init__(
            self,
            input_image_shape,
            hist_num_frames,
            map_feature_dim=256,
            hist_feature_dim=128,
            cond_feature_dim=256,
            map_encoder_model_arch="resnet18",
            # if using non-rasterized histories, need these
            agent_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
            neighbor_hist_norm_info=([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]),
    ) -> None:
        super(ConditionEncoder, self).__init__()
        
        cond_in_feat_size = 0
        cond_out_feat_size = cond_feature_dim

        # history encoding
        self.agent_hist_encoder = self.neighbor_hist_encoder = None

        # ego history is ALWAYS used as conditioning
        self.agent_hist_encoder = AgentHistoryEncoder(hist_num_frames,
                                                        out_dim=hist_feature_dim,
                                                        use_norm=True,
                                                        norm_info=agent_hist_norm_info)
        cond_in_feat_size += hist_feature_dim

        self.neighbor_hist_encoder = NeighborHistoryEncoder(hist_num_frames,
                                                            out_dim=hist_feature_dim,
                                                            use_norm=True,
                                                            norm_info=neighbor_hist_norm_info)
        cond_in_feat_size += hist_feature_dim

        # map encoding
        self.input_image_shape = input_image_shape
        self.map_encoder = MapEncoder(
            model_arch=map_encoder_model_arch,
            input_image_shape=input_image_shape,
            global_feature_dim=map_feature_dim,
        )

        cond_in_feat_size += map_feature_dim

        # MLP to combine conditioning from all sources
        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)
        self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                cond_out_feat_size,
                                                combine_layer_dims,
                                                normalization=True)

    def forward(self, data_batch):
        N = data_batch["history_positions"].size(0)
        device = data_batch["history_positions"].device

        cond_feat_in = torch.empty((N,0)).to(device)
        #
        # rasterized map
        #
        image_batch = data_batch["image"]
        map_global_feat, _ = self.map_encoder(image_batch)
        cond_feat_in = torch.cat([cond_feat_in, map_global_feat], dim=-1)
        #
        # ego history
        #
        agent_hist_feat = self.agent_hist_encoder(data_batch["history_positions"],
                                                    data_batch["history_yaws"],
                                                    data_batch["history_speeds"],
                                                    data_batch["extent"],
                                                    data_batch["history_availabilities"])
        cond_feat_in = torch.cat([cond_feat_in, agent_hist_feat], dim=-1)
        #
        # neighbor history
        #
        neighbor_hist_feat = self.neighbor_hist_encoder(data_batch["all_other_agents_history_positions"],
                                                        data_batch["all_other_agents_history_yaws"],
                                                        data_batch["all_other_agents_history_speeds"],
                                                        data_batch["all_other_agents_extents"],
                                                        data_batch["all_other_agents_history_availabilities"])
        cond_feat_in = torch.cat([cond_feat_in, neighbor_hist_feat], dim=-1)  
        #
        # Process all features together
        #
        cond_feat = self.process_cond_mlp(cond_feat_in)

        return cond_feat

class STRIVEVaeModel(nn.Module):
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(STRIVEVaeModel, self).__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        prior = vaes.FixedGaussianPrior(latent_dim=algo_config.vae.latent_dim)

        # map_encoder = base_models.RasterizedMapEncoder(
        #     model_arch=algo_config.model_architecture,
        #     input_image_shape=modality_shapes["image"],
        #     feature_dim=algo_config.map_feature_dim,
        #     use_spatial_softmax=algo_config.spatial_softmax.enabled,
        #     spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        # )

        # c_encoder = base_models.ConditionEncoder(
        #     map_encoder=map_encoder,
        #     trajectory_shape=trajectory_shape,
        #     condition_dim=algo_config.vae.condition_dim,
        #     mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
        #     goal_encoder=None
        # )

        c_encoder = ConditionEncoder(
            modality_shapes["image"],
            algo_config.history_num_frames+1, # takes in past history and current
            map_feature_dim=algo_config.map_feature_dim,
            hist_feature_dim=algo_config.history_feature_dim,
            cond_feature_dim=algo_config.vae.condition_dim,
            agent_hist_norm_info=algo_config.agent_hist_norm_info,
            neighbor_hist_norm_info=algo_config.neighbor_hist_norm_info,
        )

        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim + algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time
        )

        q_encoder = base_models.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=prior.posterior_param_shapes,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size,
            normalization=True, # layernorm
        )

        decoder = base_models.ConditionDecoder(traj_decoder)

        self.vae = vaes.CVAE(
            q_net=q_encoder,
            c_net=c_encoder,
            decoder=decoder,
            prior=prior
        )

        self.dyn = traj_decoder.dyn
        self.algo_config = algo_config

        # for guided sampling
        self.current_guidance = None

    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        inputs = OrderedDict(trajectories=trajectories)
        condition_inputs = batch_inputs #OrderedDict(image=batch_inputs["image"], goal=None)

        decoder_kwargs = dict()
        if self.dyn is not None:
            decoder_kwargs["current_states"] = batch_utils().get_current_states(batch_inputs, self.dyn.type())

        outs = self.vae.forward(inputs=inputs, condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        outs.update(self._traj_to_preds(outs["x_recons"]["trajectories"]))
        if self.dyn is not None:
            outs["controls"] = outs["x_recons"]["controls"]
        return outs

    def sample(self, batch_inputs: dict, n: int,
                guide_as_filter_only=False):
        condition_inputs = batch_inputs #OrderedDict(image=batch_inputs["image"], goal=None)

        decoder_kwargs = dict()
        if self.dyn is not None:
            curr_states = batch_utils().get_current_states(batch_inputs, self.dyn.type())
            decoder_kwargs["current_states"] = TensorUtils.repeat_by_expand_at(curr_states, repeats=n, dim=0).detach()

        guide_losses = None
        if self.current_guidance is not None:
            # run latent optimization using configured guidance
            outs, guide_losses = self.guidance_optim(condition_inputs, n, decoder_kwargs, batch_inputs,
                                                        num_iter=0 if guide_as_filter_only else 200)
        else:
            outs = self.vae.sample(condition_inputs=condition_inputs, n=n, decoder_kwargs=decoder_kwargs)

        outs = self._traj_to_preds(outs["trajectories"])
        if guide_losses is not None:
            outs["guide_losses"] = guide_losses
        return outs

    def predict(self, batch_inputs: dict):
        condition_inputs = batch_inputs #OrderedDict(image=batch_inputs["image"], goal=None)

        decoder_kwargs = dict()
        if self.dyn is not None:
            decoder_kwargs["current_states"] = batch_utils().get_current_states(batch_inputs, self.dyn.type())

        outs = self.vae.predict(condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        return self._traj_to_preds(outs["trajectories"])

    def compute_losses(self, pred_batch, data_batch):
        kl_loss = self.vae.compute_kl_loss(pred_batch)
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        losses = OrderedDict(prediction_loss=pred_loss, kl_loss=kl_loss)
        if self.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses

    #------------------------------------------ guidance utils ------------------------------------------#

    def set_guidance(self, guidance_config_list, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config_list is not None:
            if len(guidance_config_list) > 0 and verify_guidance_config_list(guidance_config_list):
                print('Instantiating test-time guidance with configs:')
                print(guidance_config_list)
                self.current_guidance = DiffuserGuidance(guidance_config_list, example_batch)

    def update_guidance(self, **kwargs):
        if self.current_guidance is not None:
            self.current_guidance.update(**kwargs)
    
    def clear_guidance(self):
        self.current_guidance = None

    # ---------------------------- latent "guidance" optimization helpers ------------------------------ 

    def guidance_optim(self, condition_inputs, num_samp, decoder_kwargs, data_batch,
                        lr=0.02,
                        num_iter=100,
                        ):
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        if num_iter == 0:
            print('STRIVE: not using full latent optim, only computing loss for filtering!')

        with torch.enable_grad():
            # get initial latents to start optim from
            init_z, cond_feat = self.vae.sample_z(condition_inputs, num_samp) # B x N x D
            prior_mean = torch.zeros_like(init_z)
            prior_var = torch.ones_like(init_z)

            # ["controls", "trajectories", "terminal_state"]
            # init_out = self.vae.decode_z(init_z, cond_feat, decoder_kwargs) 
            # print(init_out.keys())
            # print(init_out["trajectories"].size()) # B x N x T x 3 (x,y,yaw)
            # print(init_out["controls"].size()) # B x N x T x 2 (acc,yawvel)
            # print(init_out["states"].size()) # B x N x T x 4 (x,y,vel,yaw)
            
            # optimize latents
            z = init_z.clone().detach().requires_grad_(True)
            c = cond_feat.clone().detach()
            if decoder_kwargs is not None:
                decoder_kwargs["current_states"] = decoder_kwargs["current_states"].clone().detach()
            optim_z = optim.Adam([z], lr=lr)

            # run optim
            pbar_optim = tqdm.tqdm(range(num_iter))
            for oidx in pbar_optim:
                optim_z.zero_grad()

                cur_out = self.vae.decode_z(z, c, decoder_kwargs) 
                # Need (x,y,vel,yaw,acc,yawvel) for guidance loss torch.cat(states, controls)
                x = torch.cat([cur_out['states'], cur_out['controls']], dim=-1)
                loss, guide_losses = self.current_guidance.compute_guidance_loss(x, data_batch)
                # NOTE: prior loss assumes standard normal for now
                prior_loss = torch.mean(-log_normal(z, prior_mean, prior_var))
                loss = loss + prior_loss
                loss.backward()
                optim_z.step()

            final_out = self.vae.decode_z(z, c, decoder_kwargs) 
            final_x = torch.cat([final_out['states'], final_out['controls']], dim=-1)
            final_loss, guide_losses = self.current_guidance.compute_guidance_loss(final_x, data_batch)

            if guide_losses is not None:
                print('===== GUIDANCE LOSSES ======')
                for k,v in guide_losses.items():
                    print('%s: %.012f' % (k, np.nanmean(v.cpu())))

            return final_out, guide_losses
