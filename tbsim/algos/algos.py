from collections import OrderedDict
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import tbsim.utils.torch_utils as TorchUtils

from tbsim.models.rasterized_models import (
    RasterizedPlanningModel,
    RasterizedVAEModel,
    RasterizedGCModel,
    RasterizedGANModel,
    RasterizedDiscreteVAEModel,
    RasterizedECModel,
    RasterizedTreeVAEModel,
    RasterizedSceneTreeModel,
)
from tbsim.models.base_models import (
    MLPTrajectoryDecoder,
    RasterizedMapUNet,
)
from tbsim.models.transformer_model import TransformerModel
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
from tbsim.policies.common import Plan, Action
import tbsim.algos.algo_utils as AlgoUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.models.diffuser import DiffuserModel
from tbsim.models.diffuser_helpers import EMA
from tbsim.models.strive import STRIVEVaeModel
from tbsim.models.scenediffuser import SceneDiffuserModel
from tbsim.utils.guidance_loss import choose_action_from_guidance, choose_action_from_gt
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates,  add_scene_dim_to_agent_data, get_stationary_mask

class BehaviorCloning(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(BehaviorCloning, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict, with_guidance=False):
        return self.nets["policy"](obs_dict, with_guidance)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        # targets_all = batch_utils().batch_to_target_all_agents(data_batch)
        # raw_type = torch.cat(
        #     (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        #     dim=1,
        # ).type(torch.int64)
        #
        # pred_edges = batch_utils().generate_edges(
        #     raw_type,
        #     targets_all["extents"],
        #     pos_pred=targets_all["target_positions"],
        #     yaw_pred=targets_all["target_yaws"],
        # )
        #
        # coll_rates = TensorUtils.to_numpy(
        #     Metrics.batch_pairwise_collision_rate(pred_edges)
        # )
        # for c in coll_rates:
        #     metrics["coll_" + c] = float(coll_rates[c])

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict, with_guidance=False, **kwargs):
        preds = self(obs_dict, with_guidance)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        # for scene_editor visualization
        info = dict(
            action_samples=Action(
                positions=preds["positions"][:, None, ...],
                yaws=preds["yaws"][:, None, ...]
            ).to_dict(),
        )
        return action, info
    
    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.nets["policy"].set_guidance_optimization_params(guidance_optimization_params)

    def set_guidance(self, guidance_config, example_batch=None):
        self.nets["policy"].set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        cur_policy.clear_guidance()


class BehaviorCloningGC(BehaviorCloning):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        pl.LightningModule.__init__(self)
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + algo_config.goal_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        self.nets["policy"] = RasterizedGCModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            goal_feature_dim=algo_config.goal_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    def get_action(self, obs_dict, **kwargs):
        obs_dict = dict(obs_dict)
        if "plan" in kwargs:
            plan = kwargs["plan"]
            assert isinstance(plan, Plan)
            obs_dict["target_positions"] = plan.positions
            obs_dict["target_yaws"] = plan.yaws
            obs_dict["target_availabilities"] = plan.availabilities
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}


class SpatialPlanner(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(SpatialPlanner, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        self.nets["policy"] = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=4,  # (pixel, x_residual, y_residual, yaw)
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        keys = {"posErr": "val/metrics_goal_pos_err"}
        if self.algo_config.loss_weights.pixel_bce_loss > 0:
            keys["valBCELoss"] = "val/losses_pixel_bce_loss"
        if self.algo_config.loss_weights.pixel_ce_loss > 0:
            keys["valCELoss"] = "val/losses_pixel_ce_loss"
        return keys

    def forward(self, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        pred_map = self.nets["policy"](obs_dict["image"])
        return self.forward_prediction(
            pred_map,
            obs_dict,
            mask_drivable=mask_drivable,
            num_samples=num_samples,
            clearance=clearance
        )

    @staticmethod
    def forward_prediction(pred_map, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        assert pred_map.shape[1] == 4  # [location_logits, residual_x, residual_y, yaw]

        pred_map[:, 1:3] = torch.sigmoid(pred_map[:, 1:3])
        location_map = pred_map[:, 0]

        # get normalized probability map
        location_prob_map = torch.softmax(location_map.flatten(1), dim=1).reshape(location_map.shape)

        if mask_drivable:
            # At test time: optionally mask out undrivable regions
            if "drivable_map" not in obs_dict:
                drivable_map = batch_utils().get_drivable_region_map(obs_dict["image"])
            else:
                drivable_map = obs_dict["drivable_map"]
            for i, m in enumerate(drivable_map):
                if m.sum() == 0:  # if nowhere is drivable, set it to all True's to avoid decoding problems
                    drivable_map[i] = True

            location_prob_map = location_prob_map * drivable_map.float()

        # decode map as predictions
        pixel_pred, res_pred, yaw_pred, pred_prob = AlgoUtils.decode_spatial_prediction(
            prob_map=location_prob_map,
            residual_yaw_map=pred_map[:, 1:],
            num_samples=num_samples,
            clearance = clearance,
        )

        # transform prediction to agent coordinate
        pos_pred = transform_points_tensor(
            (pixel_pred + res_pred),
            obs_dict["agent_from_raster"].float()
        )

        return dict(
            predictions=dict(
                positions=pos_pred,
                yaws=yaw_pred
            ),
            log_likelihood=torch.log(pred_prob),
            spatial_prediction=pred_map,
            location_map=location_map,
            location_prob_map=location_prob_map
        )

    @staticmethod
    def compute_metrics(pred_batch, data_batch):
        metrics = dict()
        goal_sup = data_batch["goal"]
        goal_pred = TensorUtils.squeeze(pred_batch["predictions"], dim=1)

        pos_norm_err = torch.norm(
            goal_pred["positions"] - goal_sup["goal_position"], dim=-1
        )
        metrics["goal_pos_err"] = torch.mean(pos_norm_err)

        metrics["goal_yaw_err"] = torch.mean(
            torch.abs(goal_pred["yaws"] - goal_sup["goal_yaw"])
        )

        pixel_pred = torch.argmax(
            torch.flatten(pred_batch["location_map"], start_dim=1), dim=1
        )  # [B]
        metrics["goal_selection_err"] = torch.mean(
            (goal_sup["goal_position_pixel_flat"].long() != pixel_pred).float()
        )
        metrics["goal_cls_err"] = torch.mean((torch.exp(pred_batch["log_likelihood"]) < 0.5).float())
        metrics = TensorUtils.to_numpy(metrics)
        for k, v in metrics.items():
            metrics[k] = float(v)
        return metrics

    @staticmethod
    def compute_losses(pred_batch, data_batch):
        losses = dict()
        pred_map = pred_batch["spatial_prediction"]
        b, c, h, w = pred_map.shape

        goal_sup = data_batch["goal"]
        # compute pixel classification loss
        location_prediction = pred_map[:, 0]
        losses["pixel_bce_loss"] = torch.binary_cross_entropy_with_logits(
            input=location_prediction,  # [B, H, W]
            target=goal_sup["goal_spatial_map"],  # [B, H, W]
        ).mean()

        losses["pixel_ce_loss"] = torch.nn.CrossEntropyLoss()(
            input=location_prediction.flatten(start_dim=1),  # [B, H * W]
            target=goal_sup["goal_position_pixel_flat"].long(),  # [B]
        )

        # compute residual and yaw loss
        gather_inds = TensorUtils.unsqueeze_expand_at(
            goal_sup["goal_position_pixel_flat"].long(), size=c, dim=1
        )[..., None]  # -> [B, C, 1]

        local_pred = torch.gather(
            input=torch.flatten(pred_map, 2),  # [B, C, H * W]
            dim=2,
            index=gather_inds  # [B, C, 1]
        ).squeeze(-1)  # -> [B, C]
        residual_pred = local_pred[:, 1:3]
        yaw_pred = local_pred[:, 3:4]
        losses["pixel_res_loss"] = torch.nn.MSELoss()(residual_pred, goal_sup["goal_position_residual"])
        losses["pixel_yaw_loss"] = torch.nn.MSELoss()(yaw_pred, goal_sup["goal_yaw"])

        return losses

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.forward(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = self.compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        with torch.no_grad():
            metrics = self.compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = TensorUtils.detach(self.compute_losses(pout, batch))
        metrics = self.compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, mask_drivable=False, sample=False, num_plan_samples=1, clearance=None, **kwargs):
        num_samples = num_plan_samples if sample else None
        preds = self.forward(obs_dict, mask_drivable=mask_drivable, num_samples=num_samples,clearance=clearance)  # [B, num_sample, ...]
        b, n = preds["predictions"]["positions"].shape[:2]
        plan_dict = dict(
            predictions=TensorUtils.unsqueeze(preds["predictions"], dim=1),  # [B, 1, num_sample...]
            availabilities=torch.ones(b, 1, n).to(self.device),  # [B, 1, num_sample]
        )
        # pad plans to the same size as the future trajectories
        n_steps_to_pad = self.algo_config.future_num_frames - 1
        plan_dict = TensorUtils.pad_sequence(plan_dict, padding=(n_steps_to_pad, 0), batched=True, pad_values=0.)
        plan_samples = Plan(
            positions=plan_dict["predictions"]["positions"].permute(0, 2, 1, 3),  # [B, num_sample, T, 2]
            yaws=plan_dict["predictions"]["yaws"].permute(0, 2, 1, 3),  # [B, num_sample, T, 1]
            availabilities=plan_dict["availabilities"].permute(0, 2, 1)  # [B, num_sample, T]
        )

        # take the first sample as the plan
        plan = TensorUtils.map_tensor(plan_samples.to_dict(), lambda x: x[:, 0])
        plan = Plan.from_dict(plan)

        return plan, dict(location_map=preds["location_map"], plan_samples=plan_samples, log_likelihood=preds["log_likelihood"])


class VAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(VAETrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss", "minADE": "val/metrics_ego_avg_ADE"}

    def forward(self, obs_dict):
        return self.nets["policy"].predict(obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on posterior params
        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, recon_preds, avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, recon_preds, avail
        ).mean()

        # compute ADE & FDE based on trajectory samples
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan=None, with_guidance=False, **kwargs):
        obs_dict = dict(obs_dict)
        if plan is not None and self.algo_config.goal_conditional:
            assert isinstance(plan, Plan)
            obs_dict["target_positions"] = plan.positions
            obs_dict["target_yaws"] = plan.yaws
            obs_dict["target_availabilities"] = plan.availabilities
        else:
            assert not self.algo_config.goal_conditional

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples, with_guidance=with_guidance)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                ).to_dict()
            )
        else:
            # otherwise, use prior mean to generate the sample
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info
    
    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.nets["policy"].set_guidance_optimization_params(guidance_optimization_params)

    def set_guidance(self, guidance_config, example_batch=None):
        self.nets["policy"].set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        cur_policy.clear_guidance()


class DiscreteVAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(DiscreteVAETrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedDiscreteVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss", "minADE": "val/metrics_ego_avg_ADE"}

    def forward(self, obs_dict):
        return self.nets["policy"].predict(obs_dict)["predictions"]

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=min(self.algo_config.vae.num_eval_samples,self.algo_config.vae.latent_dim))
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=min(self.algo_config.vae.num_eval_samples,self.algo_config.vae.latent_dim))

        metrics = self._compute_metrics(pout, samples, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])
        z1 = TensorUtils.to_numpy(torch.argmax(pred_batch["z"],dim=-1))
        prob = np.take_along_axis(TensorUtils.to_numpy(pred_batch["q"]),z1,1)
        # compute ADE & FDE based on posterior params
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])

        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, sample_preds[:,0], avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, sample_preds[:,0], avail
        ).mean()

        # compute ADE & FDE based on trajectory samples

        fake_prob = np.ones(sample_preds.shape[:2])/sample_preds.shape[1]

        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        
        metrics["mode_max"] = prob.max(1).mean()-1/prob.shape[1]

        return metrics

    def get_metrics(self, data_batch, traj_batch=None,horizon=None):
        
        pout = self.nets["policy"](data_batch)
        bs, M = pout["x_recons"]["trajectories"].shape[:2]
        if horizon is None:
            horizon = pout["x_recons"]["trajectories"].shape[-2]
        horizon = min([horizon,pout["x_recons"]["trajectories"].shape[-2]])
        if "logvar" in pout["x_recons"]:
            horizon = min([horizon,pout["x_recons"]["logvar"].shape[-2]])
        if traj_batch is not None:
            horizon = min([traj_batch["target_positions"].shape[-2],horizon])
            GT_traj = traj_batch["target_positions"][:,:horizon].reshape(bs,-1)
        else:
            GT_traj = data_batch["target_positions"][:,:horizon].reshape(bs,-1)
        if "logvar" in pout["x_recons"]:
            var = torch.exp(pout["x_recons"]["logvar"][:,:,:horizon,:2]).reshape(bs,M,-1).clamp(min=1e-4)
        else:
            var = None
        pred_traj = pout["x_recons"]["trajectories"][:,:,:horizon,:2].reshape(bs,M,-1)
        
        self.algo_config.eval.mode="mean"
        with torch.no_grad():
            try:
                loglikelihood = Metrics.GMM_loglikelihood(GT_traj, pred_traj, var, pout["p"],mode=self.algo_config.eval.mode)
            except:
                horizon1 = min([GT_traj.shape[-1],pred_traj.shape[-1]])
                if var is not None:
                    horizon1 = min([horizon1,var.shape[-1]])
                loglikelihood = Metrics.GMM_loglikelihood(GT_traj[...,:horizon1], pred_traj[...,:horizon1], var[...,:horizon1], pout["p"],mode=self.algo_config.eval.mode)    
        return OrderedDict(loglikelihood=loglikelihood.detach())

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                ).to_dict()
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info

class BehaviorCloningEC(BehaviorCloning):
    def __init__(self, algo_config, modality_shapes):
        super(BehaviorCloningEC, self).__init__(algo_config, modality_shapes)

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedECModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]


    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)
        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        targets_all = batch_utils().batch_to_target_all_agents(data_batch)
        raw_type = torch.cat(
            (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
            dim=1,
        ).type(torch.int64)
        pred_edges = batch_utils().generate_edges(
            raw_type,
            targets_all["extents"],
            pos_pred=targets_all["target_positions"],
            yaw_pred=targets_all["target_yaws"],
        )

        coll_rates = TensorUtils.to_numpy(
            Metrics.batch_pairwise_collision_rate(pred_edges))

        EC_edges,type_mask = batch_utils().gen_EC_edges(
            pred_batch["EC_trajectories"],
            pred_batch["cond_traj"],
            data_batch["extent"][...,:2],
            data_batch["all_other_agents_future_extents"][...,:2].max(dim=2)[0],
            data_batch["all_other_agents_types"]
        )
        EC_coll_rate = TensorUtils.to_numpy(Metrics.batch_pairwise_collision_rate_masked(EC_edges,type_mask))
        sample_preds = TensorUtils.to_numpy(pred_batch["EC_trajectories"][...,:2])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        EC_ade = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()

        EC_fde = metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["EC_ADE"] = EC_ade
        metrics["EC_FDE"] = EC_fde
        for c in coll_rates:
            metrics["coll_" + c] = float(coll_rates[c])
        for c in EC_coll_rate:
            metrics["EC_coll_" + c] = float(EC_coll_rate[c])

        return metrics

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan=None, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}
    def get_EC_pred(self,obs,cond_traj,goal_state=None):
        return self.nets["policy"].EC_predict(obs,cond_traj,goal_state)


class GANTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(GANTrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = RasterizedGANModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"egoADE": "val/metrics_ego_ADE"}

    def forward(self, obs_dict):
        return self.nets.forward(obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch, sample_batch = None):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on posterior params
        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, recon_preds, avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, recon_preds, avail
        ).mean()

        # print(metrics["ego_ADE"])

        # compute ADE & FDE based on trajectory samples
        if sample_batch is None:
            return metrics

        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def training_step(self, batch, batch_idx, optimizer_idx):
        # pout = self.nets(batch)
        # losses = self.nets.compute_losses(pout, batch)
        batch = batch_utils().parse_batch(batch)

        if optimizer_idx == 0:
            pout = self.nets.forward_generator(batch)
            losses = self.nets.compute_losses_generator(pout, batch)
            total_loss = 0.0
            for lk, l in losses.items():
                loss = l * self.algo_config.loss_weights[lk]
                self.log("train/losses_" + lk, loss)
                total_loss += loss
            metrics = self._compute_metrics(pout, batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m)
            # print("gen", gen_loss.item())
            return total_loss
        if optimizer_idx == 1:
            pout = self.nets.forward_discriminator(batch)
            losses = self.nets.compute_losses_discriminator(pout, batch)
            total_loss = losses["gan_disc_loss"] * self.algo_config.loss_weights["gan_disc_loss"]
            # print("disc", total_loss.item())
            self.log("train/losses_disc_loss", total_loss)
            return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = TensorUtils.detach(self.nets.forward_generator(batch))
        losses = self.nets.compute_losses_generator(pout, batch)
        with torch.no_grad():
            samples = self.nets.sample(batch, n=self.algo_config.gan.num_eval_samples)
        metrics = self._compute_metrics(pout, batch, samples)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        gen_optim = optim.Adam(
            params=self.nets.generator_mods.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

        optim_params = self.algo_config.optim_params["disc"]
        disc_optim = optim.Adam(
            params=self.nets.discriminator_mods.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

        return [gen_optim, disc_optim], []

    def get_action(self, obs_dict, num_action_samples=1, **kwargs):
        obs_dict = dict(obs_dict)

        preds = self.nets.sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
        action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
        info = dict(
            action_samples=Action(
                positions=preds["positions"],
                yaws=preds["yaws"]
            ).to_dict()
        )

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info


class TransformerTrafficModel(pl.LightningModule):
    def __init__(self, algo_config):
        """
        Creates networks and places them into @self.nets.
        """
        super(TransformerTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerModel(algo_config)
        device = TorchUtils.get_torch_device(algo_config.try_to_use_cuda)
        self.nets["policy"].to(device)
        self.rasterizer = None

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch, batch_idx)
        losses = self.nets["policy"].compute_losses(pout, batch)
        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)

        total_loss = 0.0
        for v in losses.values():
            total_loss += v

        metrics = self._compute_metrics(pout["predictions"], batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m, prog_bar=False)
        tqdm_dict = {"g_loss": total_loss}
        output = OrderedDict(
            {"loss": total_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch, batch_idx)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout["predictions"], batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}

    def _compute_metrics(self, predictions, batch):
        metrics = {}
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(batch["target_positions"])
        avail = TensorUtils.to_numpy(batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)
        return metrics


class TransformerGANTrafficModel(pl.LightningModule):
    def __init__(self, algo_config):
        """
        Creates networks and places them into @self.nets.
        """
        super(TransformerGANTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = TransformerModel(algo_config)
        device = TorchUtils.get_torch_device(algo_config.try_to_use_cuda)
        self.nets["policy"].to(device)
        self.rasterizer = None

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, pout, batch):
        predictions = pout["predictions"]
        metrics = {}
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(batch["target_positions"])
        avail = TensorUtils.to_numpy(batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)
        metrics["positive_likelihood"] = TensorUtils.to_numpy(
            pout["scene_predictions"]["likelihood"]
        ).mean()
        metrics["negative_likelihood"] = TensorUtils.to_numpy(
            pout["scene_predictions"]["likelihood_new"]
        ).mean()
        return metrics

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

            optimizer_idx (int): index of the optimizer, 0 for discriminator and 1 for generator

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch, batch_idx)

        # adversarial loss is binary cross-entropy
        if optimizer_idx == 0:
            real_label = torch.ones_like(pout["scene_predictions"]["likelihood"])
            fake_label = torch.zeros_like(pout["scene_predictions"]["likelihood_new"])
            d_loss_real = self.adversarial_loss(
                pout["scene_predictions"]["likelihood"], real_label
            )
            d_loss_fake = self.adversarial_loss(
                pout["scene_predictions"]["likelihood_new"], fake_label
            )
            d_loss = d_loss_real + d_loss_fake
            # if batch_idx % 200 == 0:
            #     print("positive:", pout["likelihood"][0:5])
            #     print("negative:", pout["likelihood_new"][0:5])
            return d_loss
        if optimizer_idx == 1:

            losses = self.nets["policy"].compute_losses(pout, batch)

            for lk, l in losses.items():
                self.log("train/losses_" + lk, l)

            g_loss = 0.0
            for v in losses.values():
                g_loss += v

            g_loss += (
                torch.mean(1.0 - pout["scene_predictions"]["likelihood_new"])
                * self.algo_config.GAN_weight
            )
            metrics = self._compute_metrics(pout, batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m, prog_bar=False)
            return g_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params
        optim_generator = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params["policy"],
            net=self.nets["policy"].Transformermodel,
        )

        optim_params_discriminator = self.algo_config.optim_params_discriminator
        optim_discriminator = TorchUtils.optimizer_from_optim_params(
            net_optim_params=optim_params_discriminator,
            net=self.nets["policy"].Discriminator,
        )
        return [optim_discriminator, optim_generator], []

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}

class TreeVAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(TreeVAETrafficModel, self).__init__()
        # assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedTreeVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def get_EC_pred(self,obs_dict,cond_traj,goal=None):
        if goal is None:
            return self.nets["policy"](obs_dict,cond_traj=cond_traj)
        else:
            return self.nets["policy"](obs_dict,cond_traj=cond_traj,goal=goal)


    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}
        total_horizon = self.nets["policy"].stage*self.nets["policy"].num_frames_per_stage
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        gt = gt[...,:total_horizon,:]
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])
        avail = avail[...,:total_horizon]
        
        # compute ADE & FDE based on posterior params

        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        prob = TensorUtils.to_numpy(sample_batch["p"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, sample_preds[:,0], avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, sample_preds[:,0], avail
        ).mean()

        # compute ADE & FDE based on trajectory samples
        
        fake_prob = prob/prob.sum(-1,keepdims=True)
        
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        
        metrics["mode_max"] = prob.max(1).mean()-1/prob.shape[1]

        return metrics

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                )
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info



class SceneTreeTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(SceneTreeTrafficModel, self).__init__()
        # assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedSceneTreeModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def predict(self,obs,**kwargs):
        return TensorUtils.detach(self.nets["policy"](obs,predict=True,**kwargs))



    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}
        
        total_horizon = self.nets["policy"].stage*self.nets["policy"].num_frames_per_stage
        if data_batch["target_positions"].ndim==3:
            gt = torch.cat((data_batch["target_positions"].unsqueeze(1),data_batch["all_other_agents_future_positions"]),1)
            avail = torch.cat((data_batch["target_availabilities"].unsqueeze(1),data_batch["all_other_agents_future_availability"]),1)*pred_batch["agent_avail"].unsqueeze(-1)
        elif data_batch["target_positions"].ndim==4:
            gt = data_batch["target_positions"]
            avail = data_batch["target_availabilities"]
        gt = TensorUtils.to_numpy(gt)
        gt = gt[...,:total_horizon,:]
        
        avail = TensorUtils.to_numpy(avail)
        avail = avail[...,:total_horizon]
        
        # compute ADE & FDE based on posterior params
        
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        z = TensorUtils.to_numpy(torch.argmax(pred_batch["p"],-1))
        bs,M= preds.shape[:2]
        idx = np.tile(z[:,0].reshape(bs,1,1,1,1),(1,1,*preds.shape[-3:]))
        pred_selected = np.take_along_axis(preds[:,0],idx,1)

        prob = TensorUtils.to_numpy(sample_batch["p"])
        pred_prob = TensorUtils.to_numpy(pred_batch["p"])
        
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, 
            gt.reshape(-1,total_horizon,2), pred_selected.reshape(-1,total_horizon,2), avail.reshape(-1,total_horizon)
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, 
            gt.reshape(-1,total_horizon,2), pred_selected.reshape(-1,total_horizon,2), avail.reshape(-1,total_horizon)
        ).mean()
        
        # compute ADE & FDE based on trajectory samples
        
        fake_prob = prob/prob.sum(-1,keepdims=True)
        n = prob.shape[-1]
        Na = preds.shape[-3]
        gt_tiled = np.tile(gt,(M,1,1,1)).reshape(-1,total_horizon,2)
        fake_prob = np.tile(fake_prob.reshape(-1,n),(Na,1))
        sample_pred_tiled = TensorUtils.join_dimensions(sample_preds.swapaxes(2,3),0,3) 
        avail_tiled = np.tile(avail,(M,1,1)).reshape(-1,avail.shape[-1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "max").mean()
        
        metrics["mode_max"] = pred_prob.max(-1).mean()-1/pred_prob.shape[-1]
        return metrics

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                )
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info


class DiffuserTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name, do_log=True, guidance_config=None, constraint_config=None):
        """
        Creates networks and places them into @self.nets.
        """
        super(DiffuserTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate
        # used only when data_centric == 'scene' and coordinate == 'agent'
        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist
        # to help control stationary agent's behavior
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.stationary_mask = None

        # "Observations" are inputs to diffuser that are not outputs
        # "Actions" are inputs and outputs
        # "transition" dim = observation + action this is the input at each step of denoising
        # "output" is final output of the entired denoising process.

        # TBD: extract these and modify the later logics
        if algo_config.diffuser_input_mode == 'state':
            observation_dim = 0
            action_dim = 3 # x, y, yaw
            output_dim = 3 # x, y, yaw
        elif algo_config.diffuser_input_mode == 'action':
            observation_dim = 0
            action_dim = 2 # acc, yawvel
            output_dim = 2 # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4 # x, y, vel, yaw
            action_dim = 2 # acc, yawvel
            output_dim = 2 # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action_no_dyn':
            observation_dim = 4 # x, y, vel, yaw
            action_dim = 2 # acc, yawvel
            output_dim = 6 # x, y, vel, yaw, acc, yawvel
        else:
            raise
        
        print('registered_name', registered_name)
        if 'nusc' in registered_name:
            diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
            neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']
        elif 'l5' in registered_name:
            diffuser_norm_info = algo_config.lyft_norm_info['diffuser']
            agent_hist_norm_info = algo_config.lyft_norm_info['agent_hist']
            neighbor_hist_norm_info = algo_config.lyft_norm_info['neighbor_hist']
        elif 'nuplan' in registered_name:
            diffuser_norm_info = algo_config.nuplan_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nuplan_norm_info['agent_hist']
            neighbor_hist_norm_info = algo_config.nuplan_norm_info['neighbor_hist']
        else:
            raise


        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0 # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map
        self.use_rasterized_hist = algo_config.rasterized_history

        if self.use_cond:
            if self.cond_drop_map_p > 0:
                print('DIFFUSER: Dropping map input conditioning with p = %f during training...' % (self.cond_drop_map_p))
            if self.cond_drop_neighbor_p > 0:
                print('DIFFUSER: Dropping neighbor traj input conditioning with p = %f during training...' % (self.cond_drop_neighbor_p))


        self.nets["policy"] = DiffuserModel(
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,

            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames+1, # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,

            cond_feature_dim=algo_config.cond_feat_dim,
            curr_state_feature_dim=algo_config.curr_state_feat_dim,

            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,

            observation_dim=observation_dim, 
            action_dim=action_dim,

            output_dim=output_dim,

            n_timesteps=algo_config.n_diffusion_steps,
            
            loss_type=algo_config.loss_type, 
            clip_denoised=algo_config.clip_denoised,

            predict_epsilon=algo_config.predict_epsilon,
            action_weight=algo_config.action_weight, 
            loss_discount=algo_config.loss_discount, 
            loss_weights=algo_config.diffusor_loss_weights,
            
            dim_mults=algo_config.dim_mults,

            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,

            base_dim=algo_config.base_dim,
            diffuser_building_block=algo_config.diffuser_building_block,

            action_loss_only = algo_config.action_loss_only,
            
            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_reconstructed_state=algo_config.use_reconstructed_state,

            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,

            diffuser_norm_info=diffuser_norm_info,
            agent_hist_norm_info=agent_hist_norm_info,
            neighbor_hist_norm_info=neighbor_hist_norm_info,

            disable_control_on_stationary=self.disable_control_on_stationary,
        )

        # set up initial guidance and constraints
        if guidance_config is not None:
            self.set_guidance(guidance_config)
        if constraint_config is not None:
            self.set_constraints(constraint_config)

        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.cur_train_step = 0

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}

    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):

        if self.disable_control_on_stationary and global_t == 0:
            self.stationary_mask = get_stationary_mask(obs_dict, self.disable_control_on_stationary, self.moving_speed_th)
            B = self.stationary_mask.shape[0]
            # (B, M) -> (B*N, M)
            stationary_mask_expand =  self.stationary_mask.unsqueeze(1).expand(B, num_samp).reshape(B*num_samp)
        else:
            stationary_mask_expand = None
    
        cur_policy = self.nets["policy"]
        # this function is only called at validation time, so use ema
        if self.use_ema:
            cur_policy = self.ema_policy
        return cur_policy(obs_dict, plan, num_samp, return_diffusion=True,
                                   return_guidance_losses=True, class_free_guide_w=class_free_guide_w,
                                   apply_guidance=(not guide_as_filter_only),
                                   guide_clean=guide_clean, global_t=global_t, stationary_mask=stationary_mask_expand)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # NOTE: old single-mode
        # ade = Metrics.single_mode_metrics(
        #     Metrics.batch_average_displacement_error, gt, preds, avail
        # )
        # fde = Metrics.single_mode_metrics(
        #     Metrics.batch_final_displacement_error, gt, preds, avail
        # )

        # metrics["ego_ADE"] = np.mean(ade)
        # metrics["ego_FDE"] = np.mean(fde)

        # ======================

        # compute ADE & FDE based on trajectory samples
        sample_preds = preds
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number (relative to the CURRENT epoch) - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        if self.use_ema and self.cur_train_step % self.ema_update_every == 0:
            self.step_ema(self.cur_train_step)
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            pass
        elif self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, merge_BM=True, max_neighbor_dist=self.scene_agent_max_neighbor_dist)
        else:
            raise NotImplementedError

        # drop out conditioning if desired
        if self.use_cond:
            if self.use_rasterized_map:
                num_sem_layers = batch['maps'].size(1) # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                if self.cond_drop_map_p > 0:
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_map_p
                    # only fill the last num_sem_layers as these correspond to semantic map
                    batch["image"][drop_mask, -num_sem_layers:] = self.cond_fill_val

            if self.use_rasterized_hist:
                # drop layers of map corresponding to histories
                # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                num_sem_layers = batch['maps'].size(1) if batch['maps'] is not None else None
                if self.cond_drop_neighbor_p > 0:
                    # sample different mask so sometimes both get dropped, sometimes only one
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_neighbor_p
                    if num_sem_layers is None:
                        batch["image"][drop_mask] = self.cond_fill_val
                    else:
                        # only fill the layers before semantic map corresponding to trajectories (neighbors and ego)
                        batch["image"][drop_mask, :-num_sem_layers] = self.cond_fill_val
            else:
                if self.cond_drop_neighbor_p > 0:
                    # drop actual neighbor trajectories instead
                    # set availability to False, will be zeroed out in model
                    B = batch["all_other_agents_history_availabilities"].size(0)
                    drop_mask = torch.rand((B)) < self.cond_drop_neighbor_p
                    batch["all_other_agents_history_availabilities"][drop_mask] = 0

        # diffuser only take the data to estimate loss
        losses = self.nets["policy"].compute_losses(batch)

        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        # metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        # for mk, m in metrics.items():
        #     self.log("train/metrics_" + mk, m)

        return {
            "loss": total_loss,
            "all_losses": losses,
            # "all_metrics": metrics
        }
    
    def validation_step(self, batch, batch_idx):
        cur_policy = self.nets["policy"]
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            pass
        elif self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, merge_BM=True, max_neighbor_dist=self.scene_agent_max_neighbor_dist)
        else:
            raise NotImplementedError
        
        losses = TensorUtils.detach(cur_policy.compute_losses(batch))
        
        pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False)
        metrics = self._compute_metrics(pout, batch)
        return_dict =  {"losses": losses, "metrics": metrics}

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
            ema_losses = TensorUtils.detach(cur_policy.compute_losses(batch))
            pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False)
            ema_metrics = self._compute_metrics(pout, batch)
            return_dict["ema_losses"] = ema_losses
            return_dict["ema_metrics"] = ema_metrics

        return return_dict


    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)
        
        if self.use_ema:
            for k in outputs[0]["ema_losses"]:
                m = torch.stack([o["ema_losses"][k] for o in outputs]).mean()
                self.log("val/ema_losses_" + k, m)
            for k in outputs[0]["ema_metrics"]:
                m = np.stack([o["ema_metrics"][k] for o in outputs]).mean()
                self.log("val/ema_metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
        )

    def get_plan(self, obs_dict, **kwargs):
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict,
                    num_action_samples=1,
                    class_free_guide_w=0.0, 
                    guide_as_filter_only=False,
                    guide_with_gt=False,
                    guide_clean=False,
                    **kwargs):
        plan = kwargs.get("plan", None)

        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy

            # sanity chech that policies are different
            # for current_params, ma_params in zip(self.nets["policy"].parameters(), self.ema_policy.parameters()):
            #     old_weight, up_weight = ma_params.data, current_params.data
            #     print(torch.sum(old_weight - up_weight))
            # exit()

        # already called in policy_composer, but just for good measure...
        cur_policy.eval()

        # update with current "global" timestep
        cur_policy.update_guidance(global_t=kwargs['step_index'])
        
        preds = self(obs_dict, plan, num_samp=num_action_samples,
                    class_free_guide_w=class_free_guide_w, guide_as_filter_only=guide_as_filter_only,
                    guide_clean=guide_clean, global_t=kwargs['step_index']) 
        # [B, N, T, 2]
        B, N, _, _ = preds["positions"].size()

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((B), dtype=torch.long, device=preds["positions"].device)
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif cur_policy.current_perturbation_guidance.current_guidance is not None:
            # choose sample closest to desired guidance
            guide_losses = preds.pop("guide_losses", None)
            
            # from tbsim.models.diffuser_helpers import choose_act_using_guide_loss
            # act_idx = choose_act_using_guide_loss(guide_losses, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, act_idx)
            act_idx = choose_action_from_guidance(preds, obs_dict, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, guide_losses)          
                    
        action_preds = TensorUtils.map_tensor(preds, lambda x: x[torch.arange(B), act_idx])

        preds_positions = preds["positions"]
        preds_yaws = preds["yaws"]

        action_preds_positions = action_preds["positions"]
        action_preds_yaws = action_preds["yaws"]

        if self.disable_control_on_stationary and self.stationary_mask is not None:
            stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, N)
            
            preds_positions[stationary_mask_expand] = 0
            preds_yaws[stationary_mask_expand] = 0

            action_preds_positions[self.stationary_mask] = 0
            action_preds_yaws[self.stationary_mask] = 0

        
        info = dict(
            action_samples=Action(
                positions=preds_positions,
                yaws=preds_yaws
            ).to_dict(),
            # diffusion_steps={
            #     'traj' : action_preds["diffusion_steps"] # full state for the first sample
            # },
        )
        action = Action(
            positions=action_preds_positions,
            yaws=action_preds_yaws
        )
        return action, info

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.clear_guidance()


    def set_constraints(self, constraint_config):
        '''
        Resets the test-time hard constraints to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_constraints(constraint_config)

    def set_guidance_optimization_params(self, guidance_optimization_params):
        '''
        Resets the test-time guidance_optimization_params.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_optimization_params(guidance_optimization_params)
    
    def set_diffusion_specific_params(self, diffusion_specific_params):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_diffusion_specific_params(diffusion_specific_params)

class STRIVETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(STRIVETrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = STRIVEVaeModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )

        print(self.nets["policy"])

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss", "minADE": "val/metrics_ego_avg_ADE"}

    def forward(self, obs_dict):
        return self.nets["policy"].predict(obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on posterior params
        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, recon_preds, avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, recon_preds, avail
        ).mean()

        # compute ADE & FDE based on trajectory samples
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_action(self, obs_dict, num_action_samples=1,
                    guide_as_filter_only=False,
                    guide_with_gt=False, **kwargs):
        obs_dict = dict(obs_dict)

        # already called in policy_composer, but just for good measure...
        self.nets["policy"].eval()
        # update with current "global" timestep
        self.nets["policy"].update_guidance(global_t=kwargs['step_index'])

        preds = self.nets["policy"].sample(obs_dict, n=num_action_samples,
                                            guide_as_filter_only=guide_as_filter_only)
        guide_losses = preds.pop("guide_losses", None)                
        preds = preds["predictions"]  # [B, N, T, 3]
        B, N, _, _ = preds["positions"].shape

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((B), dtype=torch.long, device=preds["positions"].device)
        # apply GT or guidance filtering if desired
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif self.nets["policy"].current_guidance is not None:
            act_idx = choose_action_from_guidance(preds, obs_dict, self.nets["policy"].current_guidance.guide_configs, guide_losses)


        action_preds = TensorUtils.map_tensor(preds, lambda x: x[torch.arange(B), act_idx])  
        info = dict(
            action_samples=Action(
                positions=preds["positions"],
                yaws=preds["yaws"]
            ).to_dict()
        )

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        cur_policy.set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        cur_policy.clear_guidance()

class SceneDiffuserTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name, do_log=True, guidance_config=None, constraint_config=None):
        """
        Creates networks and places them into @self.nets.
        """
        super(SceneDiffuserTrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        # assigned at run-time according to the given data batch
        self.data_centric = None
        # ['agent_centric', 'scene_centric']
        self.coordinate = algo_config.coordinate
        # used only when data_centric == 'scene' and coordinate == 'agent'
        self.scene_agent_max_neighbor_dist = algo_config.scene_agent_max_neighbor_dist
        # to help control stationary agent's behavior
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
        self.stationary_mask = None

        # "Observations" are inputs to diffuser that are not outputs
        # "Actions" are inputs and outputs
        # "transition" dim = observation + action this is the input at each step of denoising
        # "output" is final output of the entired denoising process.

        # TBD: extract these and modify the later logics
        if algo_config.diffuser_input_mode == 'state':
            observation_dim = 0
            action_dim = 3 # x, y, yaw
            output_dim = 3 # x, y, yaw
        elif algo_config.diffuser_input_mode == 'action':
            observation_dim = 0
            action_dim = 2 # acc, yawvel
            output_dim = 2 # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action':
            observation_dim = 4 # x, y, vel, yaw
            action_dim = 2 # acc, yawvel
            output_dim = 2 # acc, yawvel
        elif algo_config.diffuser_input_mode == 'state_and_action_no_dyn':
            observation_dim = 4 # x, y, vel, yaw
            action_dim = 2 # acc, yawvel
            output_dim = 6 # x, y, vel, yaw, acc, yawvel
        else:
            raise
        
        print('registered_name', registered_name)
        diffuser_norm_info = ([-17.5, 0, 0, 0, 0, 0],[22.5, 10, 40, 3.14, 500, 31.4])
        agent_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_hist_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        neighbor_fut_norm_info = ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0])
        if 'nusc' in registered_name:
            diffuser_norm_info = algo_config.nusc_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nusc_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.nusc_norm_info:
                neighbor_hist_norm_info = algo_config.nusc_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.nusc_norm_info:
                neighbor_fut_norm_info = algo_config.nusc_norm_info['neighbor_fut']
        elif 'l5' in registered_name:
            diffuser_norm_info = algo_config.lyft_norm_info['diffuser']
            agent_hist_norm_info = algo_config.lyft_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.lyft_norm_info:
                neighbor_hist_norm_info = algo_config.lyft_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.lyft_norm_info:
                neighbor_fut_norm_info = algo_config.lyft_norm_info['neighbor_fut']
        elif 'nuplan' in registered_name:
            diffuser_norm_info = algo_config.nuplan_norm_info['diffuser']
            agent_hist_norm_info = algo_config.nuplan_norm_info['agent_hist']
            if 'neighbor_hist' in algo_config.nuplan_norm_info:
                neighbor_hist_norm_info = algo_config.nuplan_norm_info['neighbor_hist']
            if 'neighbor_fut' in algo_config.nuplan_norm_info:
                neighbor_fut_norm_info = algo_config.nuplan_norm_info['neighbor_fut']
        else:
            raise


        self.cond_drop_map_p = algo_config.conditioning_drop_map_p
        self.cond_drop_neighbor_p = algo_config.conditioning_drop_neighbor_p
        min_cond_drop_p = min([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        max_cond_drop_p = max([self.cond_drop_map_p, self.cond_drop_neighbor_p])
        assert min_cond_drop_p >= 0.0 and max_cond_drop_p <= 1.0
        self.use_cond = self.cond_drop_map_p < 1.0 and self.cond_drop_neighbor_p < 1.0 # no need for conditioning arch if always dropping
        self.cond_fill_val = algo_config.conditioning_drop_fill

        self.use_rasterized_map = algo_config.rasterized_map
        self.use_rasterized_hist = algo_config.rasterized_history

        if self.use_cond:
            if self.cond_drop_map_p > 0:
                print('DIFFUSER: Dropping map input conditioning with p = %f during training...' % (self.cond_drop_map_p))
            if self.cond_drop_neighbor_p > 0:
                print('DIFFUSER: Dropping neighbor traj input conditioning with p = %f during training...' % (self.cond_drop_neighbor_p))


        self.nets["policy"] = SceneDiffuserModel(
            rasterized_map=algo_config.rasterized_map,
            use_map_feat_global=algo_config.use_map_feat_global,
            use_map_feat_grid=algo_config.use_map_feat_grid,
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,

            rasterized_hist=algo_config.rasterized_history,
            hist_num_frames=algo_config.history_num_frames+1, # the current step is concat to the history
            hist_feature_dim=algo_config.history_feature_dim,

            diffuser_model_arch=algo_config.diffuser_model_arch,
            horizon=algo_config.horizon,

            observation_dim=observation_dim, 
            action_dim=action_dim,

            output_dim=output_dim,

            n_timesteps=algo_config.n_diffusion_steps,
            
            loss_type=algo_config.loss_type, 
            clip_denoised=algo_config.clip_denoised,

            predict_epsilon=algo_config.predict_epsilon,
            action_weight=algo_config.action_weight, 
            loss_discount=algo_config.loss_discount, 
            loss_weights=algo_config.loss_weights,
            loss_decay_rates=algo_config.loss_decay_rates,

            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,

            action_loss_only = algo_config.action_loss_only,
            
            diffuser_input_mode=algo_config.diffuser_input_mode,
            use_reconstructed_state=algo_config.use_reconstructed_state,

            use_conditioning=self.use_cond,
            cond_fill_value=self.cond_fill_val,

            diffuser_norm_info=diffuser_norm_info,
            agent_hist_norm_info=agent_hist_norm_info,
            neighbor_hist_norm_info=neighbor_hist_norm_info,
            neighbor_fut_norm_info=neighbor_fut_norm_info,

            agent_hist_embed_method=algo_config.agent_hist_embed_method,
            neigh_hist_embed_method=algo_config.neigh_hist_embed_method,
            map_embed_method=algo_config.map_embed_method,
            interaction_edge_speed_repr=algo_config.interaction_edge_speed_repr,
            normalize_rel_states=algo_config.normalize_rel_states,
            mask_social_interaction=algo_config.mask_social_interaction,
            mask_edge=algo_config.mask_edge,
            neighbor_inds=algo_config.neighbor_inds,
            edge_attr_separation=algo_config.edge_attr_separation,
            social_attn_radius=algo_config.social_attn_radius,
            use_last_hist_step=algo_config.use_last_hist_step,
            use_noisy_fut_edge=algo_config.use_noisy_fut_edge,
            use_const_speed_edge=algo_config.use_const_speed_edge,
            all_interactive_social=algo_config.all_interactive_social,
            mask_time=algo_config.mask_time,
            layer_num_per_edge_decoder=algo_config.layer_num_per_edge_decoder,
            attn_combination=algo_config.attn_combination,
            single_cond_feat=algo_config.single_cond_feat,

            disable_control_on_stationary=self.disable_control_on_stationary,

            coordinate=self.coordinate,
        )

        # set up initial guidance and constraints
        if guidance_config is not None:
            self.set_guidance(guidance_config)
        if constraint_config is not None:
            self.set_constraints(constraint_config)

        # set up EMA
        self.use_ema = algo_config.use_ema
        if self.use_ema:
            print('DIFFUSER: using EMA... val and get_action will use ema model')
            self.ema = EMA(algo_config.ema_decay)
            self.ema_policy = copy.deepcopy(self.nets["policy"])
            self.ema_policy.requires_grad_(False)
            self.ema_update_every = algo_config.ema_step
            self.ema_start_step = algo_config.ema_start_step
            self.reset_parameters()

        self.cur_train_step = 0

    @property
    def checkpoint_monitor_keys(self):
        if self.use_ema:
            return {"valLoss": "val/ema_losses_diffusion_loss"}
        else:
            return {"valLoss": "val/losses_diffusion_loss"}

    def forward(self, obs_dict, plan=None, step_index=0, num_samp=1, class_free_guide_w=0.0, guide_as_filter_only=False, guide_clean=False, global_t=0):
        if self.disable_control_on_stationary and global_t == 0:
            self.stationary_mask = get_stationary_mask(obs_dict, self.disable_control_on_stationary, self.moving_speed_th)
            B, M = self.stationary_mask.shape
            # (B, M) -> (B, N, M) -> (B*N, M)
            stationary_mask_expand =  self.stationary_mask.unsqueeze(1).expand(B, num_samp, M).reshape(B*num_samp, M)
        else:
            stationary_mask_expand = None
             

        cur_policy = self.nets["policy"]
        # this function is only called at validation time, so use ema
        if self.use_ema:
            cur_policy = self.ema_policy
        return cur_policy(obs_dict, plan, num_samp, return_diffusion=True,
                                   return_guidance_losses=True, class_free_guide_w=class_free_guide_w,
                                   apply_guidance=(not guide_as_filter_only),
                                   guide_clean=guide_clean, global_t=global_t, stationary_mask=stationary_mask_expand)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        sample_preds = predictions["positions"]
        B, N, M, T, _ = sample_preds.shape
        # (B, N, M, T, 2) -> (B, M, N, T, 2) -> (B*M, N, T, 2)
        sample_preds = TensorUtils.to_numpy(sample_preds.permute(0, 2, 1, 3, 4).reshape(B*M, N, T, -1))
        # (B, M, T, 2) -> (B*M, T, 2)
        gt = TensorUtils.to_numpy(data_batch["target_positions"].reshape(B*M, T, -1))
        # (B, M, T) -> (B*M, T)
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"].reshape(B*M, T))
        
        # compute ADE & FDE based on trajectory samples
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_APD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FPD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def reset_parameters(self):
        self.ema_policy.load_state_dict(self.nets["policy"].state_dict())

    def step_ema(self, step):
        if step < self.ema_start_step:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_policy, self.nets["policy"])

    def training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number (relative to the CURRENT epoch) - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        if self.use_ema and self.cur_train_step % self.ema_update_every == 0:
            self.step_ema(self.cur_train_step)
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
        elif self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            batch = add_scene_dim_to_agent_data(batch)
        elif self.data_centric == 'scene' and self.coordinate == 'scene_centric':
            pass
        else:
            raise NotImplementedError
       
        # drop out conditioning if desired
        if self.use_cond:
            if self.use_rasterized_map:
                num_sem_layers = batch['maps'].size(1) # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                if self.cond_drop_map_p > 0:
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_map_p
                    # only fill the last num_sem_layers as these correspond to semantic map
                    batch["image"][drop_mask, -num_sem_layers:] = self.cond_fill_val

            if self.use_rasterized_hist:
                # drop layers of map corresponding to histories
                # NOTE: this assumes a trajdata-based loader. Will not work with lyft-specific loader.
                num_sem_layers = batch['maps'].size(1) if batch['maps'] is not None else None
                if self.cond_drop_neighbor_p > 0:
                    # sample different mask so sometimes both get dropped, sometimes only one
                    drop_mask = torch.rand((batch["image"].size(0))) < self.cond_drop_neighbor_p
                    if num_sem_layers is None:
                        batch["image"][drop_mask] = self.cond_fill_val
                    else:
                        # only fill the layers before semantic map corresponding to trajectories (neighbors and ego)
                        batch["image"][drop_mask, :-num_sem_layers] = self.cond_fill_val
            else:
                if self.cond_drop_neighbor_p > 0:
                    # drop actual neighbor trajectories instead
                    # set availability to False, will be zeroed out in model
                    B = batch["history_availabilities"].size(0)
                    drop_mask = torch.rand((B)) < self.cond_drop_neighbor_p
                    batch["history_availabilities"][drop_mask] = 0

        # diffuser only take the data to estimate loss
        losses = self.nets["policy"].compute_losses(batch)

        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        # metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        # for mk, m in metrics.items():
        #     self.log("train/metrics_" + mk, m)

        return {
            "loss": total_loss,
            "all_losses": losses,
            # "all_metrics": metrics
        }
    
    def validation_step(self, batch, batch_idx):
        cur_policy = self.nets["policy"]
        if self.data_centric is None:
            if "num_agents" in batch:
                self.data_centric = 'scene'
            else:
                self.data_centric = 'agent'

        batch = batch_utils().parse_batch(batch)

        if self.data_centric == 'scene' and self.coordinate == 'agent_centric':
            batch = convert_scene_data_to_agent_coordinates(batch, max_neighbor_dist=self.scene_agent_max_neighbor_dist, keep_order_of_neighbors=True)
        elif self.data_centric == 'agent' and self.coordinate == 'agent_centric':
            batch = add_scene_dim_to_agent_data(batch)
        elif self.data_centric == 'scene' and self.coordinate == 'scene_centric':
            pass
        else:
            raise NotImplementedError
        
        losses = TensorUtils.detach(cur_policy.compute_losses(batch))
        
        pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False, mode='training')
        metrics = self._compute_metrics(pout, batch)
        return_dict =  {"losses": losses, "metrics": metrics}

        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
            ema_losses = TensorUtils.detach(cur_policy.compute_losses(batch))
            pout = cur_policy(batch,
                        num_samp=self.algo_config.diffuser.num_eval_samples,
                        return_diffusion=False,
                        return_guidance_losses=False, mode='training')
            ema_metrics = self._compute_metrics(pout, batch)
            return_dict["ema_losses"] = ema_losses
            return_dict["ema_metrics"] = ema_metrics

        return return_dict


    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)
        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)
        
        if self.use_ema:
            for k in outputs[0]["ema_losses"]:
                m = torch.stack([o["ema_losses"][k] for o in outputs]).mean()
                self.log("val/ema_losses_" + k, m)
            for k in outputs[0]["ema_metrics"]:
                m = np.stack([o["ema_metrics"][k] for o in outputs]).mean()
                self.log("val/ema_metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
        )

    def get_plan(self, obs_dict, **kwargs):
        plan = kwargs.get("plan", None)
        preds = self(obs_dict, plan)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict,
                    num_action_samples=1,
                    class_free_guide_w=0.0, 
                    guide_as_filter_only=False,
                    guide_with_gt=False,
                    guide_clean=False,
                    **kwargs):
        plan = kwargs.get("plan", None)

        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy

            # sanity chech that policies are different
            # for current_params, ma_params in zip(self.nets["policy"].parameters(), self.ema_policy.parameters()):
            #     old_weight, up_weight = ma_params.data, current_params.data
            #     print(torch.sum(old_weight - up_weight))
            # exit()

        # already called in policy_composer, but just for good measure...
        cur_policy.eval()

        # update with current "global" timestep
        cur_policy.update_guidance(global_t=kwargs['step_index'])
        
        # visualize rollout batch for debugging
        visualize_agent_batch = False
        ind_to_vis = 5
        if visualize_agent_batch:
            from matplotlib import pyplot as plt
            from tbsim.utils.trajdata_utils import plot_agent_batch_dict
            import os
            if 'agent_name' not in obs_dict:
                obs_dict['agent_name'] = [[str(i) for i in range(obs_dict['target_positions'].shape[1])]]
            ax = plot_agent_batch_dict(obs_dict, batch_idx=ind_to_vis, legend=False, show=False, close=False)
            os.makedirs('nusc_results/agent_batch_vec_map_vis', exist_ok=True)
            plt.savefig('nusc_results/agent_batch_vec_map_vis/agent_batch_'+str(kwargs['step_index'])+'.png')
            plt.close()

        preds = self(obs_dict, plan, num_samp=num_action_samples,
                    class_free_guide_w=class_free_guide_w, guide_as_filter_only=guide_as_filter_only,
                    guide_clean=guide_clean, global_t=kwargs['step_index']) 

        # [B, N, M, T, 2]
        B, N, M, _, _ = preds["positions"].shape

        # arbitrarily use the first sample as the action by default
        act_idx = torch.zeros((M), dtype=torch.long, device=preds["positions"].device)
        if guide_with_gt and "target_positions" in obs_dict:
            act_idx = choose_action_from_gt(preds, obs_dict)
        elif cur_policy.current_perturbation_guidance.current_guidance is not None:
            # choose sample closest to desired guidance
            guide_losses = preds.pop("guide_losses", None)
            
            # from tbsim.models.diffuser_helpers import choose_act_using_guide_loss
            # act_idx = choose_act_using_guide_loss(guide_losses, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, act_idx)
            act_idx = choose_action_from_guidance(preds, obs_dict, cur_policy.current_perturbation_guidance.current_guidance.guide_configs, guide_losses)          
        def map_act_idx(x):
            # Assume B == 1 during generation. TBD: need to change this to support general batchsize
            if len(x.shape) == 4:
                # [N, T, M1, M2] -> [M1, N, T, M2]
                x = x.permute(2,0,1,3)
            elif len(x.shape) == 5:
                # [B, N, M, T, 2] -> [N, M, T, 2] -> [M, N, T, 2]
                x = x[0].permute(1,0,2,3)
            elif len(x.shape) == 6: # for "diffusion_steps"
                x = x[0].permute(1,0,2,3,4)
            else:
                raise NotImplementedError
            # [M, N, T, 2] -> [M, T, 2]
            x = x[torch.arange(M), act_idx]
            # [M, T, 2] -> [B, M, T, 2]
            x = x.unsqueeze(0)
            return x
        
        # action_preds = TensorUtils.map_tensor(preds, lambda x: x[torch.arange(B), act_idx])
        # action_preds = TensorUtils.map_tensor(preds, map_act_idx)

        preds_positions = preds["positions"]
        preds_yaws = preds["yaws"]
        preds_trajectories = preds["trajectories"]

        # action_preds_positions = action_preds["positions"]
        # action_preds_yaws = action_preds["yaws"]
        action_preds_positions = map_act_idx(preds_positions)
        action_preds_yaws = map_act_idx(preds_yaws)
        
        # [N, T, M1, M2] -> [M1, N, T, M2]
        # attn_weights = preds["attn_weights"].permute(2,0,1,3)
        # only keep the selected action's accosicated attention weights
        # [N, T, M1, M2] -> [B, M1, T, M2] -> [M1, T, M2]
        attn_weights = map_act_idx(preds["attn_weights"]).squeeze(0)

        if self.disable_control_on_stationary and self.stationary_mask is not None:
            
            stationary_mask_expand = self.stationary_mask.unsqueeze(1).expand(B, N, M)
            
            preds_positions[stationary_mask_expand] = 0
            preds_yaws[stationary_mask_expand] = 0
            preds_trajectories[stationary_mask_expand] = 0

            action_preds_positions[self.stationary_mask] = 0
            action_preds_yaws[self.stationary_mask] = 0

        info = dict(
            action_samples=Action(
                positions=preds_positions, # (B, N, M, T, 2)
                yaws=preds_yaws
            ).to_dict(),
            # diffusion_steps={
            #     'traj' : action_preds["diffusion_steps"] # full state for the first sample
            # },
            trajectories=preds_trajectories,
            act_idx=act_idx,
            dyn=self.nets["policy"].dyn,
            attn_weights=attn_weights,
        )
        action = Action(
            positions=action_preds_positions, # (B, M, T, 2)
            yaws=action_preds_yaws
        )
        return action, info

    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Resets the test-time guidance functions to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance(guidance_config, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.clear_guidance()


    def set_constraints(self, constraint_config):
        '''
        Resets the test-time hard constraints to follow during prediction.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_constraints(constraint_config)

    def set_guidance_optimization_params(self, guidance_optimization_params):
        '''
        Resets the test-time guidance_optimization_params.
        '''
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_guidance_optimization_params(guidance_optimization_params)
    
    def set_diffusion_specific_params(self, diffusion_specific_params):
        cur_policy = self.nets["policy"]
        # use EMA for val
        if self.use_ema:
            cur_policy = self.ema_policy
        cur_policy.set_diffusion_specific_params(diffusion_specific_params)