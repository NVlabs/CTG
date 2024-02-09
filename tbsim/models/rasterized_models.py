from typing import Dict, List
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import random

import tbsim.models.base_models as base_models
from tbsim.models.Transformer import SimpleTransformer
import tbsim.models.vaes as vaes
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.loss_utils import (
    trajectory_loss,
    MultiModal_trajectory_loss,
    goal_reaching_loss,
    collision_loss,
    collision_loss_masked,
    log_normal_mixture,
    NLL_GMM_loss,
    compute_pred_loss
)


class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
    ) -> None:

        super().__init__()
        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=map_feature_dim,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs,
            output_activation=nn.ReLU
        )
        self.traj_decoder = trajectory_decoder
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    def forward(self, data_batch: Dict[str, torch.Tensor], with_guidance: bool = False) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        dec_output = self.traj_decoder.forward(inputs=map_feat, current_states=curr_states, with_guidance=with_guidance, data_batch=data_batch)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]
            out_dict["curr_states"] = curr_states
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        # pred_edges = batch_utils().get_edges_from_batch(
        #     data_batch=data_batch,
        #     ego_predictions=pred_batch["predictions"]
        # )
        #
        # coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(
            prediction_loss=pred_loss,
            goal_loss=goal_loss,
            # collision_loss=coll_loss
        )
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses
    
    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.traj_decoder.set_guidance_optimization_params(guidance_optimization_params)

    def set_guidance(self, guidance_config_list, example_batch=None):
        self.traj_decoder.set_guidance(guidance_config_list, example_batch)
    
    def clear_guidance(self):
        self.traj_decoder.clear_guidance()


class RasterizedGCModel(RasterizedPlanningModel):
    def __init__(
            self,
            model_arch: str,
            input_image_shape: int,
            map_feature_dim: int,
            goal_feature_dim: int,
            weights_scaling: List[float],
            trajectory_decoder: nn.Module,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
    ) -> None:
        super(RasterizedGCModel, self).__init__(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            map_feature_dim=map_feature_dim,
            weights_scaling=weights_scaling,
            trajectory_decoder=trajectory_decoder,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs,
        )

        self.goal_encoder = base_models.MLP(
            input_dim=trajectory_decoder.state_dim,
            output_dim=goal_feature_dim,
            output_activation=nn.ReLU
        )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"])
        goal_state = torch.gather(
            target_traj,  # [B, T, 3]
            dim=1,
            index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])
        ).squeeze(1)  # -> [B, 3]
        goal_feat = self.goal_encoder(goal_state) # -> [B, D]
        input_feat = torch.cat((map_feat, goal_feat), dim=-1)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        preds = self.traj_decoder.forward(inputs=input_feat, current_states=curr_states)

        traj = preds["trajectories"]
        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = preds["controls"]
        return out_dict


class RasterizedGANModel(nn.Module):
    """
    GAN-based latent variable model (e.g., social GAN)
    """

    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super().__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)

        self.gen_map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs
        )

        self.disc_map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs
        )

        self.traj_encoder = base_models.RNNTrajectoryEncoder(
            trajectory_dim=3,
            rnn_hidden_size=algo_config.traj_encoder.rnn_hidden_size,
            feature_dim=algo_config.traj_encoder.feature_dim,
            mlp_layer_dims=algo_config.traj_encoder.mlp_layer_dims
        )
        self.traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + algo_config.gan.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time
        )
        self.gan_disc = base_models.MLP(
            input_dim=algo_config.map_feature_dim + algo_config.traj_encoder.feature_dim,
            output_dim=1,
            layer_dims=algo_config.gan.disc_layer_dims
        )

        self.generator_mods = nn.ModuleList(modules=[self.gen_map_encoder, self.traj_decoder])
        self.discriminator_mods = nn.ModuleList(modules=[self.disc_map_encoder, self.traj_encoder, self.gan_disc])

        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        self.algo_config = algo_config

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None

        gan_noise = torch.randn(image_batch.shape[0], self.algo_config.gan.latent_dim).to(image_batch.device)
        input_feats = torch.cat((map_feat, gan_noise), dim=-1)
        dec_output = self.traj_decoder.forward(inputs=input_feats, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]

        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_traj_feats = self.traj_encoder(traj)
        pred_score = self.gan_disc(torch.cat((map_feat, pred_traj_feats), dim=-1))
        out_dict["gen_score"] = torch.sigmoid(pred_score).squeeze(-1)

        pred_traj_feats = self.traj_encoder(traj.detach())
        real_traj_feats = self.traj_encoder(target_traj)
        disc_pred_score = self.gan_disc(torch.cat((map_feat, pred_traj_feats), dim=-1))
        disc_real_score = self.gan_disc(torch.cat((map_feat, real_traj_feats), dim=-1))

        out_dict["disc_pred_score"] = torch.sigmoid(disc_pred_score).squeeze(-1)
        out_dict["disc_real_score"] = torch.sigmoid(disc_real_score).squeeze(-1)

        return out_dict

    def forward_generator(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.gen_map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None

        gan_noise = torch.randn(image_batch.shape[0], self.algo_config.gan.latent_dim).to(image_batch.device)
        input_feats = torch.cat((map_feat, gan_noise), dim=-1)
        dec_output = self.traj_decoder.forward(inputs=input_feats, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]

        pred_traj_feats = self.traj_encoder(traj)
        pred_score = self.gan_disc(torch.cat((map_feat, pred_traj_feats), dim=-1))
        out_dict["gen_score"] = torch.sigmoid(pred_score).squeeze(-1)
        return out_dict

    def forward_discriminator(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.gen_map_encoder(image_batch)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None


        gan_noise = torch.randn(image_batch.shape[0], self.algo_config.gan.latent_dim).to(image_batch.device)
        input_feats = torch.cat((map_feat, gan_noise), dim=-1)
        dec_output = self.traj_decoder.forward(inputs=input_feats, current_states=curr_states)
        traj = dec_output["trajectories"]

        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_traj_feats = self.traj_encoder(traj.detach())
        real_traj_feats = self.traj_encoder(target_traj)
        disc_pred_score = self.gan_disc(torch.cat((map_feat, pred_traj_feats), dim=-1))
        disc_real_score = self.gan_disc(torch.cat((map_feat, real_traj_feats), dim=-1))

        out_dict = dict()
        out_dict["disc_pred_score"] = torch.sigmoid(disc_pred_score).squeeze(-1)
        out_dict["disc_real_score"] = torch.sigmoid(disc_real_score).squeeze(-1)
        return out_dict

    def sample(self, data_batch, n):
        image_batch = data_batch["image"]
        map_feat = self.gen_map_encoder(image_batch)
        map_feat = TensorUtils.repeat_by_expand_at(map_feat, repeats=n, dim=0)

        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
            curr_states = TensorUtils.repeat_by_expand_at(curr_states, repeats=n, dim=0)
        else:
            curr_states = None

        gan_noise = torch.randn(map_feat.shape[0], self.algo_config.gan.latent_dim).to(image_batch.device)
        input_feats = torch.cat((map_feat, gan_noise), dim=-1)
        dec_output = self.traj_decoder.forward(inputs=input_feats, current_states=curr_states)
        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

        return TensorUtils.reshape_dimensions(out_dict, begin_axis=0, end_axis=1, target_dims=(image_batch.shape[0], n))

    def get_adv_loss_function(self):
        if self.algo_config.gan.loss_type == "gan":
            adv_loss_func = nn.BCELoss()
        elif self.algo_config.gan.loss_type == "lsgan":
            adv_loss_func = nn.MSELoss()
        else:
            raise Exception("GAN loss {} is not supported".format(self.algo_config.gan.loss_type))
        return adv_loss_func

    def compute_losses_generator(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        device = target_traj.device
        valid = torch.ones(target_traj.shape[0]).to(device)
        gen_loss = self.get_adv_loss_function()(pred_batch["gen_score"], valid)
        losses = OrderedDict(
            prediction_loss=pred_loss,
            gan_gen_loss=gen_loss,
        )
        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses

    def compute_losses_discriminator(self, pred_batch, data_batch):
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)

        device = target_traj.device
        valid = torch.ones(target_traj.shape[0]).to(device)
        fake = torch.zeros(target_traj.shape[0]).to(device)
        adv_loss_func = self.get_adv_loss_function()
        real_loss = adv_loss_func(pred_batch["disc_real_score"], valid)
        fake_loss = adv_loss_func(pred_batch["disc_pred_score"], fake)
        disc_loss = (real_loss + fake_loss) / 2

        losses = OrderedDict(
            gan_disc_loss=disc_loss
        )
        return losses


class RasterizedVAEModel(nn.Module):
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(RasterizedVAEModel, self).__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        prior = vaes.FixedGaussianPrior(latent_dim=algo_config.vae.latent_dim)

        goal_dim = 0 if not algo_config.goal_conditional else algo_config.goal_feature_dim

        map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim + goal_dim + algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time
        )

        if algo_config.goal_conditional:
            goal_encoder = base_models.MLP(
                input_dim=traj_decoder.state_dim,
                output_dim=algo_config.goal_feature_dim,
                output_activation=nn.ReLU
            )
        else:
            goal_encoder = None

        c_encoder = base_models.ConditionEncoder(
            map_encoder=map_encoder,
            trajectory_shape=trajectory_shape,
            condition_dim=algo_config.vae.condition_dim,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            goal_encoder=goal_encoder
        )

        q_encoder = base_models.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim + goal_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=prior.posterior_param_shapes,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
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

    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def _get_goal_states(self, data_batch) -> torch.Tensor:
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=-1)
        goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"])  # [B]
        goal_state = torch.gather(
            target_traj,  # [B, T, 3]
            dim=1,
            index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])  # [B, 1, 3]
        ).squeeze(1)  # -> [B, 3]
        return goal_state

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        inputs = OrderedDict(trajectories=trajectories)
        goal = self._get_goal_states(batch_inputs) if self.algo_config.goal_conditional else None
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=goal)

        decoder_kwargs = dict()
        if self.dyn is not None:
            decoder_kwargs["current_states"] = batch_utils().get_current_states(batch_inputs, self.dyn.type())

        outs = self.vae.forward(inputs=inputs, condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        outs.update(self._traj_to_preds(outs["x_recons"]["trajectories"]))
        if self.dyn is not None:
            outs["controls"] = outs["x_recons"]["controls"]
        return outs

    def sample(self, batch_inputs: dict, n: int, with_guidance: bool = False):
        goal = self._get_goal_states(batch_inputs) if self.algo_config.goal_conditional else None
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=goal)

        decoder_kwargs = dict()
        if self.dyn is not None:
            curr_states = batch_utils().get_current_states(batch_inputs, self.dyn.type())
            decoder_kwargs["current_states"] = TensorUtils.repeat_by_expand_at(curr_states, repeats=n, dim=0)

        outs = self.vae.sample(condition_inputs=condition_inputs, n=n, decoder_kwargs=decoder_kwargs, with_guidance=with_guidance, batch_inputs=batch_inputs)
        return self._traj_to_preds(outs["trajectories"])

    def predict(self, batch_inputs: dict):
        goal = self._get_goal_states(batch_inputs) if self.algo_config.goal_conditional else None
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=goal)

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

    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.vae.set_guidance_optimization_params(guidance_optimization_params)
    
    def set_guidance(self, guidance_config_list, example_batch=None):
        self.vae.set_guidance(guidance_config_list, example_batch)
    
    def clear_guidance(self):
        self.vae.clear_guidance()

class RasterizedDiscreteVAEModel(nn.Module):
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(RasterizedDiscreteVAEModel, self).__init__()
        trajectory_shape = (algo_config.future_num_frames, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

        goal_dim = 0 if not algo_config.goal_conditional else algo_config.goal_feature_dim
        if algo_config.agent_future_cond.enabled:
            agent_traj_encoder = base_models.AgentTrajEncoder(trajectory_shape=trajectory_shape,
                                                              feature_dim=algo_config.agent_future_cond.feature_dim,
                                                              use_transformer=algo_config.agent_future_cond.transformer)
            agent_future_dim = algo_config.agent_future_cond.feature_dim
        else:
            agent_traj_encoder=None
            agent_future_dim = 0
        map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )
        if algo_config.vae.recon_loss_type=="MSE":
            algo_config.vae.decoder.Gaussian_var=False
        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim+goal_dim+agent_future_dim+algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time,
            Gaussian_var=algo_config.vae.decoder.Gaussian_var
        )
        self.recon_loss_type = algo_config.vae.recon_loss_type

        if algo_config.goal_conditional:
            goal_encoder = base_models.MLP(
                input_dim=traj_decoder.state_dim,
                output_dim=algo_config.goal_feature_dim,
                output_activation=nn.ReLU
            )
        else:
            goal_encoder = None
        
        self.agent_future_cond = algo_config.agent_future_cond.enabled
        
        c_encoder = base_models.ConditionEncoder(
            map_encoder=map_encoder,
            trajectory_shape=trajectory_shape,
            condition_dim=algo_config.vae.condition_dim,
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            goal_encoder=goal_encoder,
            agent_traj_encoder = agent_traj_encoder
        )
        q_encoder = base_models.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim+goal_dim+agent_future_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=OrderedDict(logq=(algo_config.vae.latent_dim,)),
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )
        p_encoder = base_models.SplitMLP(
            input_dim=algo_config.vae.condition_dim+goal_dim+agent_future_dim,
            layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            output_shapes=OrderedDict(logp=(algo_config.vae.latent_dim,))
        )
        decoder = base_models.ConditionDecoder(traj_decoder)
        if "logpi_clamp" in algo_config.vae:
            logpi_clamp = algo_config.vae.logpi_clamp
        else:
            logpi_clamp=None
        self.vae = vaes.DiscreteCVAE(
            q_net=q_encoder,
            p_net=p_encoder,
            c_net=c_encoder,
            decoder=decoder,
            K=algo_config.vae.latent_dim,
            logpi_clamp=logpi_clamp
        )

        self.dyn = traj_decoder.dyn
        self.algo_config = algo_config

    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def _get_goal_states(self, data_batch) -> torch.Tensor:
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=-1)
        goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"])  # [B]
        goal_state = torch.gather(
            target_traj,  # [B, T, 3]
            dim=1,
            index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])  # [B, 1, 3]
        ).squeeze(1)  # -> [B, 3]
        return goal_state

    def forward(self, batch_inputs: dict):
        trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
        inputs = OrderedDict(trajectories=trajectories)
        goal = self._get_goal_states(batch_inputs) if self.algo_config.goal_conditional else None
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=goal)
        if self.agent_future_cond:
            agent_traj = torch.cat((batch_inputs["all_other_agents_future_positions"],batch_inputs["all_other_agents_future_yaws"]),-1)
            condition_inputs["agent_traj"] = agent_traj

        decoder_kwargs = dict()
        if self.dyn is not None:
            current_states = batch_utils().get_current_states(batch_inputs, self.dyn.type())
            decoder_kwargs["current_states"] = current_states.tile(self.vae.K,1)

        outs = self.vae.forward(inputs=inputs, condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        outs.update(self._traj_to_preds(outs["x_recons"]["trajectories"]))
        if self.dyn is not None:
            outs["controls"] = outs["x_recons"]["controls"]
        return outs

    def sample(self, batch_inputs: dict, n: int):
        goal = self._get_goal_states(batch_inputs) if self.algo_config.goal_conditional else None
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=goal)
        if self.agent_future_cond:
            agent_traj = torch.cat((batch_inputs["all_other_agents_future_positions"],batch_inputs["all_other_agents_future_yaws"]),-1)
            condition_inputs["agent_traj"] = agent_traj

        decoder_kwargs = dict()
        if self.dyn is not None:
            curr_states = batch_utils().get_current_states(batch_inputs, self.dyn.type())
            decoder_kwargs["current_states"] = TensorUtils.repeat_by_expand_at(curr_states, repeats=n, dim=0)

        outs = self.vae.sample(condition_inputs=condition_inputs, n=n, decoder_kwargs=decoder_kwargs)

        return self._traj_to_preds(outs["trajectories"])

    def predict(self, batch_inputs: dict):
        goal = self._get_goal_states(batch_inputs) if self.algo_config.goal_conditional else None
        condition_inputs = OrderedDict(image=batch_inputs["image"], goal=goal)
        if self.agent_future_cond:
            agent_traj = torch.cat((batch_inputs["all_other_agents_future_positions"],batch_inputs["all_other_agents_future_yaws"]),-1)
            condition_inputs["agent_traj"] = agent_traj

        decoder_kwargs = dict()
        if self.dyn is not None:
            decoder_kwargs["current_states"] = batch_utils().get_current_states(batch_inputs, self.dyn.type())

        outs = self.vae.predict(condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        return self._traj_to_preds(outs["trajectories"])

    def compute_losses(self, pred_batch, data_batch):
        kl_loss = self.vae.compute_kl_loss(pred_batch)
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        z1 = torch.argmax(pred_batch["z"],dim=-1)
        prob = torch.gather(pred_batch["q"],1,z1)
        if self.recon_loss_type=="NLL":
            assert "logvar" in pred_batch["x_recons"]
            bs, M, T, D = pred_batch["trajectories"].shape
            var = (torch.exp(pred_batch["x_recons"]["logvar"])+torch.ones_like(pred_batch["x_recons"]["logvar"])*1e-4).reshape(bs,M,-1)
            var = torch.gather(var,1,z1.unsqueeze(-1).repeat(1,1,var.size(-1)))
            avails = data_batch["target_availabilities"].unsqueeze(-1).repeat(1, 1, target_traj.shape[-1]).reshape(bs, -1)
            pred_loss = NLL_GMM_loss(
                x=target_traj.reshape(bs,-1),
                m=pred_batch["trajectories"].reshape(bs,M,-1),
                v=var,
                pi=prob,
                avails=avails
            )
            pred_loss = pred_loss.mean()
        elif self.recon_loss_type=="NLL_torch":

            bs, num_modes, _, _ = pred_batch["trajectories"].shape
            # Use torch distribution family to calculate likelihood
            means = pred_batch["trajectories"].reshape(bs, num_modes, -1)
            scales = torch.exp(pred_batch["x_recons"]["logvar"]).reshape(bs, num_modes, -1)
            scales = torch.gather(scales,1,z1.unsqueeze(-1).repeat(1,1,scales.size(-1)))
            mode_probs = prob
            # Calculate scale
            # post-process the scale accordingly
            scales = scales + self.algo_config.min_std

            # mixture components - make sure that `batch_shape` for the distribution is equal
            # to (batch_size, num_modes) since MixtureSameFamily expects this shape
            component_distribution = distributions.Normal(loc=means, scale=scales)
            component_distribution = distributions.Independent(component_distribution, 1)

            # unnormalized logits to categorical distribution for mixing the modes
            mixture_distribution = distributions.Categorical(probs=mode_probs)

            dist = distributions.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )
            log_prob = dist.log_prob(target_traj.reshape(bs, -1))
            pred_loss = -log_prob.mean()
            # TODO: support detach mode and masking

        elif self.recon_loss_type=="MSE":

            pred_loss = MultiModal_trajectory_loss(
                predictions=pred_batch["trajectories"],
                targets=target_traj,
                availabilities=data_batch["target_availabilities"],
                prob=prob,
                weights_scaling=self.weights_scaling,
            )
        else:
            raise NotImplementedError("{} is not implemented".format(self.recon_loss_type))
        losses = OrderedDict(prediction_loss=pred_loss, kl_loss=kl_loss)
        if self.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)

        return losses


class RasterizedECModel(nn.Module):
    """Raster-based model for planning with ego conditioning.
    """

    def __init__(self,algo_config, modality_shapes, weights_scaling):

        super().__init__()

        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )
        trajectory_shape = (algo_config.future_num_frames, 3)
        goal_dim = 0 if not algo_config.goal_conditional else algo_config.goal_feature_dim
        self.traj_decoder = base_models.MLPECTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + goal_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            EC_RNN_dim = algo_config.EC.RNN_hidden_size,
            EC_feature_dim = algo_config.EC.feature_dim,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time,
        )
        self.GC = algo_config.goal_conditional
        if self.GC:
            self.goal_encoder = base_models.MLP(
                input_dim=self.traj_decoder.state_dim,
                output_dim=algo_config.goal_feature_dim,
                output_activation=nn.ReLU
            )


        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        map_feat = self.map_encoder(image_batch)
        target_traj = torch.cat((data_batch["target_positions"],data_batch["target_yaws"]),-1)
        if self.GC:
            goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"])
            goal_state = torch.gather(
                target_traj,  # [B, T, 3]
                dim=1,
                index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])
            ).squeeze(1)  # -> [B, 3]
            goal_feat = self.goal_encoder(goal_state) # -> [B, D]
            input_feat = torch.cat((map_feat, goal_feat), dim=-1)
        else:
            input_feat = map_feat
        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None

        cond_traj = torch.cat((data_batch["all_other_agents_future_positions"],data_batch["all_other_agents_future_yaws"]),-1)
        dec_output = self.traj_decoder.forward(inputs=input_feat, current_states=curr_states, cond_traj=cond_traj)

        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
            "cond_traj":cond_traj,
            "cond_availability": data_batch["all_other_agents_future_availability"],
            "EC_trajectories":dec_output["EC_trajectories"]
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]
            out_dict["curr_states"] = curr_states
        return out_dict

    def EC_predict(self,obs,cond_traj,goal_state=None):
        image_batch = obs["image"]
        map_feat = self.map_encoder(image_batch)
        if self.GC:
            if goal_state is None:
                goal_inds = batch_utils().get_last_available_index(obs["target_availabilities"])
                target_traj = torch.cat((obs["target_positions"],obs["target_yaws"]),-1)
                goal_state = torch.gather(
                    target_traj,  # [B, T, 3]
                    dim=1,
                    index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])
                ).squeeze(1)  # -> [B, 3]
            else:
                if goal_state.ndim==3:
                    goal_state = goal_state[...,-1,:]
            goal_feat = self.goal_encoder(goal_state) # -> [B, D]

            input_feat = torch.cat((map_feat, goal_feat), dim=-1)
        else:
            input_feat = map_feat
        if self.traj_decoder.dyn is not None:
            curr_states = batch_utils().get_current_states(obs, dyn_type=self.traj_decoder.dyn.type())
        else:
            curr_states = None
        dec_output = self.traj_decoder.forward(inputs=input_feat, current_states=curr_states, cond_traj=cond_traj)

        traj = dec_output["trajectories"]

        pred_positions = traj[:, :, :2]
        pred_yaws = traj[:, :, 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
            "cond_traj":cond_traj,
            "cond_availability": torch.ones(cond_traj.shape[:3]).to(cond_traj.device),
            "EC_trajectories":dec_output["EC_trajectories"]
        }
        if self.traj_decoder.dyn is not None:
            out_dict["controls"] = dec_output["controls"]
            out_dict["curr_states"] = curr_states
        return out_dict

    def compute_losses(self, pred_batch, data_batch):

        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=data_batch["target_availabilities"],
            weights_scaling=self.weights_scaling
        )
        EC_edges,type_mask = batch_utils().gen_EC_edges(
            pred_batch["EC_trajectories"],
            pred_batch["cond_traj"],
            data_batch["extent"][...,:2],
            data_batch["all_other_agents_future_extents"][...,:2].max(dim=2)[0],
            data_batch["all_other_agents_types"]
        )
        EC_coll_loss = collision_loss_masked(EC_edges,type_mask)

        A = pred_batch["EC_trajectories"].shape[1]
        deviation_loss = trajectory_loss(
            predictions=pred_batch["EC_trajectories"],
            targets=target_traj.unsqueeze(1).repeat(1,A,1,1),
            availabilities=data_batch["all_other_agents_future_availability"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        pred_edges = batch_utils().get_edges_from_batch(
            data_batch=data_batch,
            ego_predictions=pred_batch["predictions"]
        )

        coll_loss = collision_loss(pred_edges=pred_edges)
        losses = OrderedDict(
            prediction_loss=pred_loss,
            goal_loss=goal_loss,
            collision_loss=coll_loss,
            EC_collision_loss = EC_coll_loss,
            deviation_loss = deviation_loss
        )

        if self.traj_decoder.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["controls"][..., 1] ** 2)
        return losses

class RasterizedTreeVAEModel(nn.Module):
    """A rasterized model that generates trajectory tree for prediction
    """
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(RasterizedTreeVAEModel, self).__init__()
        trajectory_shape = (algo_config.num_frames_per_stage, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)


        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )
        self.stage = algo_config.stage
        self.num_frames_per_stage=algo_config.num_frames_per_stage
        if algo_config.vae.recon_loss_type=="MSE":
            algo_config.vae.decoder.Gaussian_var=False


        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim + algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.num_frames_per_stage,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time,
            Gaussian_var=algo_config.vae.decoder.Gaussian_var
        )
        self.recon_loss_type = algo_config.vae.recon_loss_type
        
        if algo_config.goal_conditional:
            goal_encoder = base_models.MLP(
                input_dim=traj_decoder.state_dim,
                output_dim=algo_config.goal_feature_dim,
                output_activation=nn.ReLU
            )
        else:
            goal_encoder = None
        self.EC = algo_config.ego_conditioning
        if self.EC:
            c_encoder = base_models.ECEncoder(
                map_encoder=self.map_encoder.output_shape()[0],
                trajectory_shape=trajectory_shape,  # [T, D]
                EC_dim=algo_config.EC_feat_dim,
                condition_dim=algo_config.vae.condition_dim, 
                mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
                goal_encoder=goal_encoder,
                rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size)
        else:
            c_encoder = base_models.ConditionEncoder(
                map_encoder=self.map_encoder.output_shape()[0],
                trajectory_shape=trajectory_shape,
                condition_dim=algo_config.vae.condition_dim,
                mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
                goal_encoder=goal_encoder,
            )
        q_encoder = base_models.PosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=OrderedDict(logq=(algo_config.vae.latent_dim,)),
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )
        p_encoder = base_models.SplitMLP(
            input_dim=algo_config.vae.condition_dim,
            layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            output_shapes=OrderedDict(logp=(algo_config.vae.latent_dim,))
        )
        
        decoder = base_models.ConditionDecoder(traj_decoder)
        if "logpi_clamp" in algo_config.vae:
            logpi_clamp = algo_config.vae.logpi_clamp
        else:
            logpi_clamp=None
        if algo_config.ego_conditioning:
            vae_model = vaes.ECDiscreteCVAE
        else:
            vae_model = vaes.DiscreteCVAE
            
        self.vae = vae_model(
            q_net=q_encoder,
            p_net=p_encoder,
            c_net=c_encoder,
            decoder=decoder,
            K=algo_config.vae.latent_dim,
            logpi_clamp=logpi_clamp
        )
        self.dyn = traj_decoder.dyn
        if self.dyn is None:
            rnn_input_dim = trajectory_shape[1]
        else:
            rnn_input_dim = self.dyn.udim
        self.FeatureRoller = base_models.RNNFeatureRoller(rnn_input_dim,algo_config.map_feature_dim)
        self.algo_config = algo_config

    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def _get_goal_states(self, data_batch) -> torch.Tensor:
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=-1)
        goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"])  # [B]
        goal_state = torch.gather(
            target_traj,  # [B, T, 3]
            dim=1,
            index=goal_inds[:, None, None].expand(-1, 1, target_traj.shape[-1])  # [B, 1, 3]
        ).squeeze(1)  # -> [B, 3]
        return goal_state

    def forward(self, batch_inputs: dict,sample=False,**kwargs):
        if not sample:
            trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
            assert batch_inputs["target_positions"].shape[-2]>=self.stage*self.num_frames_per_stage
        H = self.num_frames_per_stage
        if self.algo_config.goal_conditional:
            if "goal" in kwargs and kwargs["goal"] is not None:
                goal = kwargs["goal"]
            else:
                goal = self._get_goal_states(batch_inputs)
        else:
            goal = None
        
        map_feat=self.map_encoder(batch_inputs["image"])
        curr_map_feat = map_feat
        preds = list()
        decoder_kwargs = dict()
        bs = batch_inputs["image"].shape[0]
        if self.EC:
            if "cond_traj" not in kwargs:
                cond_traj_total = torch.cat((batch_inputs["all_other_agents_future_positions"],batch_inputs["all_other_agents_future_yaws"]),-1)
            else:
                cond_traj_total = kwargs["cond_traj"]
            cond_traj_total = cond_traj_total[...,:self.stage*self.num_frames_per_stage,:]
            Na = cond_traj_total.shape[1]
            curr_map_feat = TensorUtils.unsqueeze_expand_at(curr_map_feat,Na+1,-2)
            cond_traj = list()
            for t in range(self.stage):
                cond_traj.append(TensorUtils.slice_tensor(cond_traj_total,2,t*H,(t+1)*H))
        for t in range(self.stage):
            
                
            batch_shape = [bs]+[self.vae.K]*t
            
            if self.dyn is not None:
                if t==0:
                    current_states = batch_utils().get_current_states(batch_inputs, self.dyn.type())
                    decoder_kwargs["current_states"] = current_states
                else:
                    current_states = outs["x_recons"]["terminal_state"]
                    if self.EC:
                        decoder_kwargs["current_states"] = TensorUtils.join_dimensions(current_states,0,t+1)
                    else:
                        decoder_kwargs["current_states"] = current_states.reshape(-1,current_states.shape[-1])
         
            if t==0:
                map_feat_t = curr_map_feat
                if goal is not None:
                    if self.EC:
                        goal_t = goal.unsqueeze(-2).repeat(1,Na+1,1)
                    else:
                        goal_t = goal.reshape(-1,goal.shape[-1])
            else:
                if goal is not None:
                    goal = TensorUtils.expand_at_single(goal.unsqueeze(-2),self.vae.K,-2)

                curr_map_feat = TensorUtils.unsqueeze_expand_at(curr_map_feat,self.vae.K,1).contiguous()
                curr_map_feat = TensorUtils.join_dimensions(curr_map_feat,0,2).contiguous()
                input_seq = outs["controls"]
                
                if self.EC:
                    input_seq_tran = input_seq.reshape(-1,*input_seq.shape[-2:])
                    
                    curr_map_feat = self.FeatureRoller(curr_map_feat.reshape(-1,*curr_map_feat.shape[-1:]),input_seq_tran)
                    curr_map_feat = TensorUtils.reshape_dimensions(curr_map_feat,0,1,(-1,Na+1))
                    current_bs = math.prod(batch_shape)
                    goal_tiled = TensorUtils.unsqueeze_expand_at(goal,Na+1,-2)
                    map_feat_t = curr_map_feat
                    goal_t = goal_tiled.reshape(current_bs,Na+1,-1)
                    
                else:
                    curr_map_feat = self.FeatureRoller(curr_map_feat.reshape(-1,*curr_map_feat.shape[-1:]),input_seq.reshape(-1,*input_seq.shape[-2:]))
                    map_feat_t = curr_map_feat.reshape(-1,curr_map_feat.shape[-1])
                    goal_t = goal.reshape(-1,goal.shape[-1])
            condition_inputs = OrderedDict(map_feature=map_feat_t, goal=goal_t)
            if not sample:
                GT_traj = trajectories[...,t*H:(t+1)*H,:]
                inputs = OrderedDict(trajectories=GT_traj.tile(self.vae.K**t,1,1))
            else:
                inputs=None
            if self.EC:
                cond_traj_tiled = cond_traj[t].repeat_interleave(int(math.prod(batch_shape[1:])),0)
                outs = self.vae.forward(inputs=inputs, condition_inputs=condition_inputs, cond_traj=cond_traj_tiled, decoder_kwargs=decoder_kwargs)
                outs = TensorUtils.recursive_dict_list_tuple_apply(outs,{torch.Tensor:lambda x:x.transpose(1,2)})
            else:
                outs = self.vae.forward(inputs=inputs, condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)
        
            outs = TensorUtils.reshape_dimensions(outs,0,1,batch_shape)

            if self.dyn is not None:
                outs["controls"] = outs["x_recons"]["controls"]
            
            preds.append(outs)
        x_recons = self._batching_from_stages(preds)
        pred_batch = OrderedDict(x_recons=x_recons)
        pred_batch.update(self._traj_to_preds(x_recons["trajectories"]))
        

        p = preds[-1]["p"]
        for t in range(self.stage-1):
            desired_shape = [self.vae.K]*(t+1)+[1]*(self.stage-t-1)
            p = p*TensorUtils.reshape_dimensions(preds[t]["p"],1,t+2,desired_shape)
        p = TensorUtils.join_dimensions(p,1,self.stage+1)
        if self.EC:
            pred_batch["p"] = p.transpose(1,2)
        else:
            pred_batch["p"] = p
        if not sample:
            q = preds[-1]["q"]
            for t in range(self.stage-1):
                desired_shape = [self.vae.K]*(t+1)+[1]*(self.stage-t-1)
                q = q*TensorUtils.reshape_dimensions(preds[t]["q"],1,t+2,desired_shape)
            q = TensorUtils.join_dimensions(q,1,self.stage+1)
            if self.EC:
                pred_batch["q"] = q.transpose(1,2)
            else:
                pred_batch["q"] = q
        pred = TensorUtils.join_dimensions(TensorUtils.slice_tensor(pred_batch,1,0,1),0,2)
        if self.EC:
            EC_pred = TensorUtils.slice_tensor(pred_batch,1,1,Na+1)
            EC_pred["cond_traj"] = cond_traj_total
            pred["EC_pred"] = EC_pred

        return pred

    def sample(self, batch_inputs: dict, n: int):
        
        pred_batch = self(batch_inputs,sample=True)
        dis_p = distributions.Categorical(probs=pred_batch["p"])  # [n_sample, batch] -> [batch, n_sample]
        z = dis_p.sample((n,)).permute(1, 0)
        traj = pred_batch["trajectories"]
        idx = z[...,None,None].repeat(1,1,*traj.shape[-2:])
        traj_selected = torch.gather(traj,1,idx)
        prob = torch.gather(pred_batch["p"],1,z)

        outs = self._traj_to_preds(traj_selected)
        outs["p"] = prob
        return outs

    def _batching_from_stages(self,preds):
        bs = preds[0]["x_recons"]["trajectories"].shape[0]
        outs = {k:list() for k in preds[0]["x_recons"] if k!="terminal_state"} #TODO: make it less hacky
        
        Na = preds[0]["p"].shape[-1]-1
        for t in range(self.stage):
            desired_shape = [bs]+[self.vae.K]*(t+1)+[1]*(self.stage-t-1)
            if self.EC:
                repeats = [1]*(t+2)+[self.vae.K]*(self.stage-t-1)+[1]*3
            else:
                repeats = [1]*(t+2)+[self.vae.K]*(self.stage-t-1)+[1]*2
            for k,v in preds[t]["x_recons"].items():
                if k!="terminal_state":
                    batched_v = TensorUtils.reshape_dimensions_single(v,0,t+2,desired_shape)
                    batched_v = batched_v.repeat(repeats)
                    outs[k].append(batched_v)

        for k,v in outs.items():
            v_cat = torch.cat(v,-2)
            outs[k]=v_cat.reshape(bs,-1,*v_cat.shape[self.stage+1:])
            if self.EC:
                outs[k] = outs[k].transpose(1,2)
        
        return outs


    def predict(self, batch_inputs: dict):
        H = self.num_frames_per_stage
        if self.algo_config.goal_conditional:
            goal = self._get_goal_states(batch_inputs)
        else:
            goal = None
        map_feat=self.map_encoder(batch_inputs["image"])
        curr_map_feat = map_feat
        preds = list()
        decoder_kwargs = dict()
        traj = list()
        for t in range(self.stage):
            if self.dyn is not None:
                if t==0:
                    current_states = batch_utils().get_current_states(batch_inputs, self.dyn.type())
                else:
                    current_states = outs["terminal_state"]
                decoder_kwargs["current_states"] = current_states
            if t>0:
                input_seq = outs["controls"] 
                curr_map_feat = self.FeatureRoller(curr_map_feat.contiguous(),input_seq.reshape(-1,*input_seq.shape[-2:]))

            condition_inputs = OrderedDict(map_feature=curr_map_feat.reshape(-1,curr_map_feat.shape[-1]), goal=goal.reshape(-1,goal.shape[-1]))
            outs = self.vae.predict(condition_inputs=condition_inputs, decoder_kwargs=decoder_kwargs)   
                 
            traj.append(outs["trajectories"])

        traj = torch.cat(traj,dim=-2)
        outs = OrderedDict(trajectories=traj)
        outs.update(self._traj_to_preds(outs["trajectories"]))
        return outs

    def compute_losses(self, pred_batch, data_batch):
        
        kl_loss = self.vae.compute_kl_loss(pred_batch)
        target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        total_horizon = self.stage*self.num_frames_per_stage
        target_traj = target_traj[...,:total_horizon,:]
        avails = data_batch["target_availabilities"][...,:total_horizon]
        prob = pred_batch["q"]
        pred_loss = compute_pred_loss(self.recon_loss_type,pred_batch,target_traj,avails,prob,self.weights_scaling)
        
        losses = OrderedDict(prediction_loss=pred_loss, kl_loss=kl_loss)
        if "EC_pred" in pred_batch:
            EC_pred = pred_batch["EC_pred"]
            EC_edges,type_mask = batch_utils().gen_EC_edges(
                EC_pred["trajectories"],
                EC_pred["cond_traj"],
                data_batch["extent"][...,:2],
                data_batch["all_other_agents_future_extents"][...,:2].max(dim=2)[0],
                data_batch["all_other_agents_types"]
            )
            EC_coll_loss = collision_loss_masked(EC_edges,type_mask)

            Na = EC_pred["trajectories"].shape[1]
            EC_batch = TensorUtils.join_dimensions(EC_pred,0,2)
            target_traj_tiled = TensorUtils.join_dimensions(TensorUtils.unsqueeze_expand_at(target_traj,Na,1),0,2)
            avails_tiled = TensorUtils.join_dimensions(TensorUtils.unsqueeze_expand_at(avails,Na,1),0,2)

            EC_prob = EC_batch["q"]
            deviation_loss = compute_pred_loss(self.recon_loss_type,EC_batch,target_traj_tiled,avails_tiled,EC_prob,self.weights_scaling)

            EC_losses = OrderedDict(EC_coll_loss=EC_coll_loss,deviation_loss=deviation_loss)
            losses.update(EC_losses)

        if self.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["x_recons"]["controls"][..., 1] ** 2)

        return losses


class RasterizedSceneTreeModel(nn.Module):
    def __init__(self, algo_config, modality_shapes, weights_scaling):
        super(RasterizedSceneTreeModel, self).__init__()
        trajectory_shape = (algo_config.num_frames_per_stage, 3)
        self.weights_scaling = nn.Parameter(torch.Tensor(weights_scaling), requires_grad=False)
        self.shuffle = algo_config.shuffle
        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],
            feature_dim=algo_config.map_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )
        self.stage = algo_config.stage
        self.num_frames_per_stage=algo_config.num_frames_per_stage
        if algo_config.vae.recon_loss_type=="MSE":
            algo_config.vae.decoder.Gaussian_var=False

        traj_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=algo_config.vae.condition_dim + algo_config.vae.latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=algo_config.num_frames_per_stage,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            network_kwargs=algo_config.decoder,
            step_time=algo_config.step_time,
            Gaussian_var=algo_config.vae.decoder.Gaussian_var
        )
        self.recon_loss_type = algo_config.vae.recon_loss_type
        
        if algo_config.goal_conditional:
            goal_encoder = base_models.MLP(
                input_dim=traj_decoder.state_dim,
                output_dim=algo_config.goal_feature_dim,
                output_activation=nn.ReLU
            )
        else:
            goal_encoder = None
        c_encoder = base_models.ECEncoder(
            map_encoder=self.map_encoder.output_shape()[0]+self.stage, # adding the stage info, TODO: make it less hacky
            trajectory_shape=trajectory_shape,  # [T, D]
            EC_dim=algo_config.EC_feat_dim,
            condition_dim=algo_config.vae.condition_dim, 
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            goal_encoder=goal_encoder,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size)

        q_encoder = base_models.ScenePosteriorEncoder(
            condition_dim=algo_config.vae.condition_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=OrderedDict(logq=(algo_config.vae.latent_dim,)),
            mlp_layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            rnn_hidden_size=algo_config.vae.encoder.rnn_hidden_size
        )
        p_encoder = base_models.SplitMLP(
            input_dim=algo_config.vae.condition_dim,
            layer_dims=algo_config.vae.encoder.mlp_layer_dims,
            output_shapes=OrderedDict(logp=(algo_config.vae.latent_dim,))
        )
        
        decoder = base_models.ConditionDecoder(traj_decoder)
        if "logpi_clamp" in algo_config.vae:
            logpi_clamp = algo_config.vae.logpi_clamp
        else:
            logpi_clamp=None

        transformer = SimpleTransformer(src_dim=algo_config.vae.condition_dim)    
        self.vae = vaes.SceneDiscreteCVAE(
            q_net=q_encoder,
            p_net=p_encoder,
            c_net=c_encoder,
            decoder=decoder,
            transformer = transformer,
            K=algo_config.vae.latent_dim,
            logpi_clamp=logpi_clamp
        )
        self.dyn = traj_decoder.dyn
        if self.dyn is None:
            rnn_input_dim = trajectory_shape[1]
        else:
            rnn_input_dim = self.dyn.udim
        self.FeatureRoller = base_models.RNNFeatureRoller(rnn_input_dim,algo_config.map_feature_dim)
        self.algo_config = algo_config
        if "perturb" in algo_config and algo_config.perturb.enabled:
            self.N_pert = algo_config.perturb.N_pert
            theta = algo_config.perturb.OU.theta
            sigma = algo_config.perturb.OU.sigma
            scale = torch.tensor(algo_config.perturb.OU.scale)
            self.pert = OrnsteinUhlenbeckPerturbation(theta*torch.ones(3),sigma*scale)
        else:
            self.N_pert = 0
            self.pert = None
    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }
    def _get_goal_states(self, data_batch) -> torch.Tensor:
        if data_batch["target_positions"].ndim==3:
            target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=-1)
            goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"][...,:self.stage*self.num_frames_per_stage])  # [B]
            goal_state = torch.gather(
                target_traj,  # [B, T, 3]
                dim=1,
                index=goal_inds[..., None, None].expand(-1, 1, target_traj.shape[-1])  # [B, 1, 3]
            ).squeeze(1)  # -> [B, 3]
            agent_target_traj = torch.cat((data_batch["all_other_agents_future_positions"], data_batch["all_other_agents_future_yaws"]), dim=-1)
            goal_inds = batch_utils().get_last_available_index(data_batch["all_other_agents_future_availability"][...,:self.stage*self.num_frames_per_stage])
            agent_goal_state = torch.gather(
                agent_target_traj,  # [B, A, T, 3]
                dim=-2,
                index=goal_inds[:,:, None, None].expand(-1,-1, 1, agent_target_traj.shape[-1])  # [B, A, 1, 3]
            ).squeeze(-2)
            goal = torch.cat((goal_state,agent_goal_state),1)
        elif data_batch["target_positions"].ndim==4:
            target_traj = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=-1)
            goal_inds = batch_utils().get_last_available_index(data_batch["target_availabilities"][...,:self.stage*self.num_frames_per_stage])  # [B, A]
            goal = torch.gather(
                target_traj,  # [B, A, T, 3]
                dim=-2,
                index=goal_inds[:,:, None, None].expand(-1,-1, 1, target_traj.shape[-1])  # [B, A, 1, 3]
            ).squeeze(-2)
        return goal


    def forward(self, batch_inputs: dict,predict=False,**kwargs):
        if batch_inputs["target_positions"].ndim==3:
            assert batch_inputs["target_positions"].shape[-2]>=self.stage*self.num_frames_per_stage
            bs,Na, = batch_inputs["all_other_agents_future_positions"].shape[:2]
            Na = Na+1
            agent_avail = batch_inputs["agent_raster_availability"]
            device = agent_avail.device
            
            ego_traj = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
            agent_traj = torch.cat((batch_inputs["all_other_agents_future_positions"], batch_inputs["all_other_agents_future_yaws"]), dim=-1)
            trajectories = torch.cat((ego_traj.unsqueeze(1),agent_traj),1)
            agent_avail = torch.cat((torch.ones([bs,1]).to(device),agent_avail),1)
            
            curr_yaw = torch.cat((batch_inputs["history_yaws"][:,0].unsqueeze(1),batch_inputs["all_other_agents_history_yaws"][:,:,0]),1)
            agent_from_world = torch.cat((batch_inputs["agent_from_world"].unsqueeze(1),batch_inputs["other_agents_agent_from_world"]),1).float()
            ego_to_agent = agent_from_world@(batch_inputs["world_from_agent"].float().unsqueeze(1))
            agent_to_ego,_ = torch.linalg.inv_ex(ego_to_agent)
            agent_to_ego = agent_to_ego.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            num_agent = [agent_idx[i].shape[0] for i in range(bs)]

        elif batch_inputs["target_positions"].ndim==4:

            bs,Na, = batch_inputs["target_positions"].shape[:2]
            agent_avail = batch_inputs["agent_type"]>0
            device = agent_avail.device
            trajectories = torch.cat((batch_inputs["target_positions"], batch_inputs["target_yaws"]), dim=-1)
            curr_yaw = batch_inputs["history_yaws"][:,:,-1]

            agent_from_world = batch_inputs["agents_from_center"]@(batch_inputs["agent_from_world"].unsqueeze(1))*agent_avail[...,None,None]
            ego_to_agent = batch_inputs["agents_from_center"]*agent_avail[...,None,None]
            agent_to_ego = batch_inputs["center_from_agents"]*agent_avail[...,None,None]
            num_agent = batch_inputs["num_agents"].tolist()
        
        agent_idx = [torch.where(agent_avail[i])[0] for i in range(bs)]
        
        H = self.num_frames_per_stage
        if self.algo_config.goal_conditional:
            if "goal" in kwargs and kwargs["goal"] is not None:
                goal = kwargs["goal"]
            else:
                goal = self._get_goal_states(batch_inputs)
                goal_pos = GeoUtils.batch_nd_transform_points(goal[...,:2],ego_to_agent)
                goal_yaw = goal[...,2:]-curr_yaw
                goal = torch.cat((goal_pos,goal_yaw),-1)
        else:
            goal = None
        if batch_inputs["image"].ndim==4:
            images = torch.cat((batch_inputs["image"].unsqueeze(1),batch_inputs["other_agents_image"]),dim=1)
        else:
            images = batch_inputs["image"]
        map_feat=self.map_encoder(TensorUtils.join_dimensions(images,0,2)).reshape(bs,Na,-1)
        curr_map_feat = map_feat
        preds = list()
        decoder_kwargs = dict()
        
        if "cond_traj" not in kwargs:
            if self.shuffle:
                cond_idx = [random.choice(agent_idx[i]) if num_agent[i]>1 else -1 for i in range(bs)]
            else:
                cond_idx = [0 if num_agent[i]>1 else -1 for i in range(bs)]
            cond_traj_local = [trajectories[i,cond_idx[i]] if cond_idx[i]>=0 else torch.zeros_like(trajectories[i,cond_idx[i]]) for i in range(bs)]
            cond_traj_local = torch.stack(cond_traj_local,0)[...,:self.stage*self.num_frames_per_stage,:]
            if self.N_pert>0:
                
                cond_traj_tiled = cond_traj_local.repeat_interleave(self.N_pert,0)
                cond_traj_local_pert = self.pert.perturb(dict(target_positions=cond_traj_tiled[...,:2],target_yaws=cond_traj_tiled[...,2:]))
                cond_traj_local_pert = torch.cat((cond_traj_local_pert["target_positions"],cond_traj_local_pert["target_yaws"]),-1).reshape(bs,self.N_pert,*cond_traj_local.shape[1:])
                flag = (torch.tensor(cond_idx)>=0).to(cond_traj_tiled.device)
                cond_traj_local_pert = cond_traj_local_pert*flag[:,None,None,None]
                cond_traj_local = torch.cat((cond_traj_local.unsqueeze(1),cond_traj_local_pert),1)
            else:
                cond_traj_local = cond_traj_local.unsqueeze(1)
            agent_avail[torch.arange(bs).to(device),cond_idx] = 0
        else:
            cond_traj_local = kwargs["cond_traj"]
            if "cond_idx" in kwargs:
                cond_idx = kwargs["cond_idx"]
                agent_avail[torch.arange(bs).to(device),cond_idx] = 0
            else:
                cond_idx = None
        if cond_traj_local is not None:
            M = cond_traj_local.shape[1]
            cond_traj_local = TensorUtils.join_dimensions(cond_traj_local,0,2)
            
            agent_avail_tiled = agent_avail.repeat_interleave(M,0)
            curr_map_feat = curr_map_feat.repeat_interleave(M,0)
            if goal is not None:
                goal = goal.repeat_interleave(M,0)
            cond_pos = GeoUtils.batch_nd_transform_points(cond_traj_local[...,:2].unsqueeze(1),ego_to_agent.repeat_interleave(M,0).unsqueeze(2))
            cond_yaw = cond_traj_local[...,2:].unsqueeze(1)-curr_yaw.repeat_interleave(M,0).unsqueeze(2)
            cond_traj = torch.cat((cond_pos,cond_yaw),-1)
            cond_traj_total = cond_traj*agent_avail_tiled[...,None,None]
            cond_traj_total = cond_traj_total[...,:self.stage*self.num_frames_per_stage,:]
            cond_traj = list()
            for t in range(self.stage):
                cond_traj.append(TensorUtils.slice_tensor(cond_traj_total,2,t*H,(t+1)*H))
        else:
            agent_avail_tiled = agent_avail
            cond_traj = None
            M = 1

        for t in range(self.stage):      
            batch_shape = [bs,M]+[self.vae.K]*t
                 
            if t==0:
                
                
                if self.dyn is not None:
                    current_states = batch_utils().get_current_states_all_agents(batch_inputs, self.algo_config.step_time,self.dyn.type())
                    local_pos = GeoUtils.batch_nd_transform_points(current_states[...,:2],ego_to_agent)
                    local_yaw = current_states[...,3:]-curr_yaw
                    current_states_local = torch.cat((local_pos,current_states[...,2:3],local_yaw),-1)
                    current_states_local = current_states_local.repeat_interleave(M,0)
                    decoder_kwargs["current_states"] = current_states_local
                    
                else:
                    current_states = batch_utils().get_current_states_all_agents(batch_inputs, self.algo_config.step_time,None)
                pos = current_states[...,:2].repeat_interleave(M,0)
                map_feat_t = curr_map_feat
            else:
                pos = TensorUtils.join_dimensions(outs["x_recons"]["trajectories"][...,-1,:2],0,t+2)
                if self.dyn is not None:
                    current_states_local = outs["x_recons"]["terminal_state"]
                    decoder_kwargs["current_states"] = TensorUtils.join_dimensions(current_states_local,0,t+2)
                if goal is not None:
                    goal = TensorUtils.repeat_by_expand_at(goal,self.vae.K,0)

                curr_map_feat = TensorUtils.unsqueeze_expand_at(curr_map_feat,self.vae.K,-2).contiguous()
                curr_map_feat = TensorUtils.join_dimensions(curr_map_feat,0,2).contiguous()
                if self.dyn is not None:
                    input_seq = outs["controls"]
                else:
                    input_seq = outs["x_recons"]["trajectories"]
                
                input_seq_tran = input_seq.reshape(-1,*input_seq.shape[-2:])
                    
                curr_map_feat = self.FeatureRoller(curr_map_feat.reshape(-1,*curr_map_feat.shape[-1:]),input_seq_tran)
                curr_map_feat = TensorUtils.reshape_dimensions(curr_map_feat,0,1,(-1,Na))
                map_feat_t = curr_map_feat
            stage_enc = torch.ones_like(map_feat_t[...,0],dtype=torch.int64)*t
            stage_enc = TensorUtils.to_one_hot(stage_enc,num_class=self.stage)
            
            condition_inputs = OrderedDict(map_feature=torch.cat((map_feat_t,stage_enc),-1), goal=goal)
            if not predict:
                GT_traj = trajectories[...,t*H:(t+1)*H,:]
                inputs = OrderedDict(trajectories=GT_traj.tile(self.vae.K**t*M,1,1,1))
            else:
                inputs=None
            cond_traj_tiled = cond_traj[t].repeat_interleave(int(math.prod(batch_shape[2:])),0) if cond_traj is not None else None
            outs = self.vae.forward(inputs=inputs, 
                                    condition_inputs=condition_inputs, 
                                    mask = agent_avail_tiled.repeat_interleave(self.vae.K**t,0),
                                    pos = pos,
                                    cond_traj=cond_traj_tiled, 
                                    decoder_kwargs=decoder_kwargs)
            outs["x_recons"] = TensorUtils.recursive_dict_list_tuple_apply(outs["x_recons"],{torch.Tensor:lambda x:x.transpose(1,2)})
            rep = int(outs["x_recons"]["trajectories"].shape[0]/agent_to_ego.shape[0])
            ego_frame_pos = GeoUtils.batch_nd_transform_points(outs["x_recons"]["trajectories"][...,:2],agent_to_ego.repeat_interleave(rep,0)[:,None,:,None,:])
            ego_frame_yaw = outs["x_recons"]["trajectories"][...,2:]+curr_yaw.repeat_interleave(rep,0)[:,None,:,None,:]
            outs["x_recons"]["trajectories"] = torch.cat((ego_frame_pos,ego_frame_yaw),-1)
            outs = TensorUtils.reshape_dimensions(outs,0,1,batch_shape)

            if self.dyn is not None:
                outs["controls"] = outs["x_recons"]["controls"]
            preds.append(outs)
        x_recons = self._batching_from_stages(preds)
        pred_batch = OrderedDict(x_recons=x_recons)
        pred_batch.update(self._traj_to_preds(x_recons["trajectories"])) 

        p = preds[-1]["p"]
        for t in range(self.stage-1):
            desired_shape = [self.vae.K]*(t+1)+[1]*(self.stage-t-1)
            p = p*TensorUtils.reshape_dimensions(preds[t]["p"],2,t+3,desired_shape)
        p = TensorUtils.join_dimensions(p,2,self.stage+2)

        pred_batch["p"] = p
        if not predict:
            q = preds[-1]["q"]
            for t in range(self.stage-1):
                desired_shape = [self.vae.K]*(t+1)+[1]*(self.stage-t-1)
                q = q*TensorUtils.reshape_dimensions(preds[t]["q"],2,t+3,desired_shape)
            q = TensorUtils.join_dimensions(q,2,self.stage+2)

            pred_batch["q"] = q
        pred_batch["cond_traj"] = cond_traj_local
        pred_batch["agent_avail"] = agent_avail
        pred_batch["target_trajectory"] = trajectories
        if cond_idx is not None:
            pred_batch["cond_idx"] = torch.tensor(cond_idx).to(device)

        return pred_batch

    def _batching_from_stages(self,preds):
        
        bs,M = preds[0]["x_recons"]["trajectories"].shape[:2]
        Na = preds[0]["x_recons"]["trajectories"].shape[-3]
        outs = {k:list() for k in preds[0]["x_recons"] if k!="terminal_state"} #TODO: make it less hacky
        
        for t in range(self.stage):
            desired_shape = [bs,M]+[self.vae.K]*(t+1)+[1]*(self.stage-t-1)+[Na]
            repeats = [1]*(t+3)+[self.vae.K]*(self.stage-t-1)+[1]*3
            for k,v in preds[t]["x_recons"].items():
                if k!="terminal_state":       
                    batched_v = TensorUtils.reshape_dimensions_single(v,0,t+4,desired_shape)
                    batched_v = batched_v.repeat(repeats)
                    outs[k].append(batched_v)

        for k,v in outs.items():
            v_cat = torch.cat(v,-2)
            v_cat = TensorUtils.join_dimensions(v_cat,2,2+self.stage)
            outs[k] = v_cat
        return outs

    def sample(self, batch_inputs: dict, n: int):
        pred_batch = self(batch_inputs,predict=True)
        dis_p = distributions.Categorical(probs=pred_batch["p"])  # [n_sample, batch] -> [batch, n_sample]
        z = dis_p.sample((n,)).permute(1, 2, 0)
        traj = pred_batch["x_recons"]["trajectories"]
        Na = traj.shape[-3]
        idx = z[:,:,:,None,None,None].repeat(1,1,1,*traj.shape[-3:])
        traj_selected = torch.gather(traj,2,idx)
        prob = torch.gather(pred_batch["p"],2,z)

        outs = self._traj_to_preds(traj_selected)
        outs["p"] = prob

        return outs

    def predict(self, batch_inputs: dict):
        pred_batch = self(batch_inputs,predict=True,cond_traj=None)
        z = torch.argmax(pred_batch["p"],-1)
        traj = pred_batch["x_recons"]["trajectories"]
        bs,M, = traj.shape[:2]
        Na = traj.shape[-3]
        
        idx = z.reshape(bs,M,1,1,1,1).repeat(1,1,1,*traj.shape[-3:])
        traj_selected = torch.gather(traj,2,idx)
        
        traj_selected = traj_selected[:,0].squeeze(1)

        outs = self._traj_to_preds(traj_selected)
        return outs
    def compute_losses(self, pred_batch, data_batch):
        if data_batch["target_positions"].ndim==3:
            availability = torch.cat((data_batch["target_availabilities"].unsqueeze(1),data_batch["all_other_agents_future_availability"]),1)
            agents_extent = data_batch["all_other_agents_future_extents"][...,:2].max(dim=2)[0]
            extent = torch.cat((data_batch["extent"][:,:2].unsqueeze(1),agents_extent),1)
            raw_type = torch.cat((data_batch["type"].unsqueeze(1),data_batch["all_other_agents_types"]),1)
        elif data_batch["target_positions"].ndim==4:
            availability = data_batch["target_availabilities"]
            extent = data_batch["extent"][...,:2]
            raw_type = data_batch["type"]
        bs,M,numMode,Na,T = pred_batch["x_recons"]["trajectories"].shape[:5]
        target_traj = pred_batch["target_trajectory"]
        target_traj = target_traj.repeat_interleave(M,0)
        target_traj = target_traj[...,:T,:]
        
        availability = availability[...,:T]*pred_batch["agent_avail"].unsqueeze(-1)
        traj_pred = TensorUtils.join_dimensions(pred_batch["x_recons"]["trajectories"],0,2)
        traj_pred_tiled = TensorUtils.join_dimensions(traj_pred,0,2)
        prob = TensorUtils.join_dimensions(pred_batch["q"],0,2)
        pred_loss, goal_loss = MultiModal_trajectory_loss(
            predictions=traj_pred, 
            targets=target_traj, 
            availabilities=availability.repeat_interleave(M,0), 
            prob=prob, 
            weights_scaling=self.weights_scaling,
            calc_goal_reach=True,
        )
        
        cond_extent = extent[torch.arange(bs),pred_batch["cond_idx"]]
        
        
        EC_edges,type_mask = batch_utils().gen_EC_edges(
            traj_pred_tiled,
            pred_batch["cond_traj"].unsqueeze(1).repeat_interleave(numMode,0).repeat_interleave(Na,1),
            cond_extent.repeat_interleave(M*numMode,0),
            extent.repeat_interleave(M*numMode,0),
            raw_type.repeat_interleave(M*numMode,0),
            pred_batch["agent_avail"].repeat_interleave(M*numMode,0)
        )
        EC_edges = TensorUtils.reshape_dimensions(EC_edges,0,1,(bs,M,numMode))
        type_mask = TensorUtils.reshape_dimensions(type_mask,0,1,(bs,M,numMode))

        EC_coll_loss = collision_loss_masked(EC_edges,type_mask,weight=pred_batch["q"].unsqueeze(-1))/numMode/M
        if not isinstance(EC_coll_loss,torch.Tensor):
            EC_coll_loss = torch.tensor(EC_coll_loss).to(target_traj.device)
        # compute collision loss
        
        pred_edges = batch_utils().generate_edges(
            (raw_type*pred_batch["agent_avail"]).repeat_interleave(M*numMode,0),
            extent.repeat_interleave(M*numMode,0),
            traj_pred_tiled[...,:2],
            traj_pred_tiled[...,2:]
        )
        
        coll_loss = collision_loss(pred_edges=pred_edges)
        if not isinstance(coll_loss,torch.Tensor):
            coll_loss = torch.tensor(coll_loss).to(target_traj.device)
        losses = OrderedDict(
            prediction_loss=pred_loss,
            goal_loss=goal_loss,
            collision_loss=coll_loss,
            EC_collision_loss = EC_coll_loss,
        )
        if self.dyn is not None:
            losses["yaw_reg_loss"] = torch.mean(pred_batch["x_recons"]["controls"][..., 1] ** 2 *pred_batch["agent_avail"][:,None,None,:,None])/numMode
        return losses