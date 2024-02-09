import torch
from torch.nn.utils.rnn import pad_sequence
from typing import OrderedDict, Tuple, Dict

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import calc_distance_map
from tbsim.utils.planning_utils import ego_sample_planning
from tbsim.policies.common import Action, Plan, RolloutAction
from tbsim.algos.algo_utils import yaw_from_pos
from tbsim.utils.trajdata_utils import convert_scene_obs_to_agent_prep, convert_scene_data_to_agent_coordinates, check_consistency, full_keywords, major_keywords, prep_keywords, add_scene_dim_to_agent_data, check_action_consistency
import numpy as np
from trajdata.utils.arr_utils import angle_wrap

class HierarchicalWrapper(object):
    """A wrapper policy that feeds subgoal from a planner to a controller"""

    def __init__(self, planner, controller):
        self.device = planner.device
        self.planner = planner
        self.controller = controller

    def eval(self):
        self.planner.eval()
        self.controller.eval()

    def get_action(self, obs, with_guidance=False, **kwargs) -> Tuple[Action, Dict]:
        plan, plan_info = self.planner.get_plan(obs)
        actions, action_info = self.controller.get_action(
            obs,
            plan=plan,
            # init_u=plan.controls
            with_guidance=with_guidance,
        )
        action_info["plan"] = plan.to_dict()
        plan_info.pop("plan_samples", None)
        action_info["plan_info"] = plan_info
        return actions, action_info
    
    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.controller.set_guidance_optimization_params(guidance_optimization_params)

    def set_guidance(self, guidance_config_list, example_batch=None):
        self.controller.set_guidance(guidance_config_list, example_batch)
    
    def clear_guidance(self):
        cur_policy = self.controller
        cur_policy.clear_guidance()

class HierarchicalSamplerWrapper(HierarchicalWrapper):
    """A wrapper policy that feeds plan samples from a stochastic planner to a controller"""

    def get_action(self, obs, with_guidance=False, **kwargs) -> Tuple[None, Dict]:
        _, plan_info = self.planner.get_plan(obs)
        plan_samples = plan_info.pop("plan_samples")
        b, n = plan_samples.positions.shape[:2]

        actions_tiled, _ = self.controller.get_action(
            obs,
            plan_samples=plan_samples,
            init_u=plan_samples.controls,
            with_guidance=with_guidance,
            **kwargs,
        )

        action_samples = TensorUtils.reshape_dimensions(
            actions_tiled.to_dict(), begin_axis=0, end_axis=1, target_dims=(b, n)
        )
        action_samples = Action.from_dict(action_samples)

        action_info = dict(
            plan_samples=plan_samples,
            action_samples=action_samples,
            plan_info=plan_info,
        )
        if "log_likelihood" in plan_info:
            action_info["log_likelihood"] = plan_info["log_likelihood"]
        return None, action_info


class SamplingPolicyWrapper(object):
    def __init__(self, ego_action_sampler, agent_traj_predictor):
        """

        Args:
            ego_action_sampler: a policy that generates N action samples
            agent_traj_predictor: a model that predicts the motion of non-ego agents
        """
        self.device = ego_action_sampler.device
        self.sampler = ego_action_sampler
        self.predictor = agent_traj_predictor

    def eval(self):
        self.sampler.eval()
        self.predictor.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # actions of shape [B, num_samples, ...]
        _, action_info = self.sampler.get_action(obs)
        action_samples = action_info["action_samples"]
        agent_preds, _ = self.predictor.get_prediction(
            obs)  # preds of shape [B, A - 1, ...]

        if isinstance(action_samples, dict):
            action_samples = Action.from_dict(action_samples)

        ego_trajs = action_samples.trajectories
        agent_pred_trajs = agent_preds.trajectories

        agent_extents = obs["all_other_agents_history_extents"][..., :2].max(
            dim=-2)[0]
        drivable_map = batch_utils().get_drivable_region_map(obs["image"]).float()
        dis_map = calc_distance_map(drivable_map)
        log_likelihood = action_info.get("log_likelihood", None)

        action_idx = ego_sample_planning(
            ego_trajectories=ego_trajs,
            agent_trajectories=agent_pred_trajs,
            ego_extents=obs["extent"][:, :2],
            agent_extents=agent_extents,
            raw_types=obs["all_other_agents_types"],
            raster_from_agent=obs["raster_from_agent"],
            dis_map=dis_map,
            log_likelihood=log_likelihood,
            weights=kwargs["cost_weights"],
        )

        ego_trajs_best = torch.gather(
            ego_trajs,
            dim=1,
            index=action_idx[:, None, None, None].expand(-1, 1, *ego_trajs.shape[2:])
        ).squeeze(1)

        ego_actions = Action(
            positions=ego_trajs_best[..., :2], yaws=ego_trajs_best[..., 2:])
        action_info["action_samples"] = action_samples.to_dict()
        if "plan_samples" in action_info:
            action_info["plan_samples"] = action_info["plan_samples"].to_dict()
        return ego_actions, action_info


class PolicyWrapper(object):
    """A convenient wrapper for specifying run-time keyword arguments"""

    def __init__(self, model, get_action_kwargs=None, get_plan_kwargs=None):
        self.model = model
        self.device = model.device
        self.action_kwargs = get_action_kwargs
        self.plan_kwargs = get_plan_kwargs

    def eval(self):
        self.model.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        return self.model.get_action(obs, **self.action_kwargs, **kwargs)

    def get_plan(self, obs, **kwargs) -> Tuple[Plan, Dict]:
        return self.model.get_plan(obs, **self.plan_kwargs, **kwargs)

    @classmethod
    def wrap_controller(cls, model, **kwargs):
        return cls(model=model, get_action_kwargs=kwargs)

    @classmethod
    def wrap_planner(cls, model, **kwargs):
        return cls(model=model, get_plan_kwargs=kwargs)

class RefineWrapper(object):
    """A wrapper that feeds coarse motion plan to a optimization-based planner for refinement"""
    def __init__(self, initial_planner, refiner, device):
        """
        Args:
            planner: a policy that generates a coarse motion plan
            refiner: a policy (optimization based) that takes the coarse motion plan and refine it
            device: device for torch
        """
        self.initial_planner = initial_planner
        self.refiner = refiner
        self.device = device
    def eval(self):
        self.initial_planner.eval()
        self.refiner.eval()
    def get_action(self, obs, **kwargs):
        coarse_plan,_ = self.initial_planner.get_action(obs,**kwargs)
        action, action_info = self.refiner.get_action(obs,coarse_plan = coarse_plan)
        return action, {"coarse_plan":coarse_plan.to_dict(), **action_info}
    

class Pos2YawWrapper(object):
    """A wrapper that computes action yaw from action positions"""
    def __init__(self, policy, dt, yaw_correction_speed):
        """

        Args:
            policy: policy to be wrapped
            dt:
            speed_filter:
        """
        self.device = policy.device
        self.policy = policy
        self._dt = dt
        self._yaw_correction_speed = yaw_correction_speed

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs):
        action, action_info = self.policy.get_action(obs, **kwargs)
        curr_pos = torch.zeros_like(action.positions[..., [0], :])
        pos_seq = torch.cat((curr_pos, action.positions), dim=-2)
        yaws = yaw_from_pos(pos_seq, dt=self._dt, yaw_correction_speed=self._yaw_correction_speed)
        action.yaws = yaws
        return action, action_info


class RolloutWrapper(object):
    """A wrapper policy that can (optionally) control both ego and other agents in a scene"""

    def __init__(self, ego_policy=None, agents_policy=None, pass_agent_obs=True):
        self.device = ego_policy.device if agents_policy is None else agents_policy.device
        self.ego_policy = ego_policy
        self.agents_policy = agents_policy
        self.pass_agent_obs = pass_agent_obs

    def eval(self):
        self.ego_policy.eval()
        self.agents_policy.eval()

    def get_action(self, obs, step_index) -> RolloutAction:
        ego_action = None
        ego_action_info = None
        agents_action = None
        agents_action_info = None
        if self.ego_policy is not None:
            assert obs["ego"] is not None
            with torch.no_grad():
                if self.pass_agent_obs:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index,agent_obs = obs["agents"])
                else:
                    ego_action, ego_action_info = self.ego_policy.get_action(
                        obs["ego"], step_index = step_index)
        if self.agents_policy is not None:
            assert obs["agents"] is not None
            with torch.no_grad():
                agents_action, agents_action_info = self.agents_policy.get_action(
                    obs["agents"], step_index = step_index)
        return RolloutAction(ego_action, ego_action_info, agents_action, agents_action_info)


class PerturbationWrapper(object):
    """A wrapper policy that perturbs the policy action with Ornstein Uhlenbeck noise"""

    def __init__(self, policy, noise):
        self.device = policy.device
        self.noise = noise
        self.policy = policy

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        actions, action_info = self.policy.get_action(obs, **kwargs)
        actions_dict = OrderedDict(target_positions=actions.positions,target_yaws=actions.yaws) 
        perturbed_action_dict = self.noise.perturb(TensorUtils.to_numpy(actions_dict))
        perturbed_action_dict = TensorUtils.to_torch(perturbed_action_dict,self.device)
        perturbed_actions = Action(perturbed_action_dict["target_positions"],perturbed_action_dict["target_yaws"])
        return perturbed_actions, action_info

# --------------------------------------------------------------------------------
from tbsim.utils.guidance_loss import PerturbationGuidance
from tbsim.models.diffuser_helpers import state_grad_general_transform
class NewSamplingPolicyWrapper(object):
    def __init__(self, ego_action_sampler, guidance_config_list, transform=state_grad_general_transform, transform_params={'dt': 0.1}):
        """

        Args:
            -ego_action_sampler: a policy that generates N action samples
            -guidance_config_list: guidance config for loss estimation
            -transform: transform applied on the trajectory before estimating loss
            -transform_params: extra parameters taken by transform
        """
        self.device = ego_action_sampler.device
        self.sampler = ego_action_sampler

        self.guidance = PerturbationGuidance(transform, transform_params)
        self.guidance_config_list = guidance_config_list
        self.guidance.set_guidance(self.guidance_config_list)

    def eval(self):
        self.sampler.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # actions of shape [B, num_samples, ...]
        _, action_info = self.sampler.get_action(obs, **kwargs)
        action_samples = action_info["action_samples"]

        if isinstance(action_samples, dict):
            action_samples = Action.from_dict(action_samples)

        ego_trajs = action_samples.trajectories
        B, N, *_ = ego_trajs.shape        
        ego_trajs_BN = TensorUtils.join_dimensions(ego_trajs, begin_axis=0, end_axis=2) # B*N, T, D
        
        _, guide_losses = self.guidance.compute_guidance_loss(ego_trajs_BN, obs, num_samp=N)

        act_idx = torch.zeros((B), dtype=torch.long) # arbitrarily use the first sample as the action if no guidance is given
        if any(self.guidance_config_list):
            from tbsim.models.diffuser_helpers import choose_act_using_guide_loss
            act_idx = choose_act_using_guide_loss(guide_losses, self.guidance.current_guidance.guide_configs, act_idx)    
   
        ego_trajs_best = TensorUtils.map_tensor(ego_trajs, lambda x: x[torch.arange(B), act_idx])


        ego_actions = Action(
            positions=ego_trajs_best[..., :2], yaws=ego_trajs_best[..., 2:])
        action_info["action_samples"] = action_samples.to_dict()
        # print('action_info["action_samples"]["positions"]', action_info["action_samples"]["positions"].shape, action_info["action_samples"]["positions"][0, 0, 0, 0].data)
        

        if "plan_samples" in action_info and not isinstance(action_info["plan_samples"], dict):
            action_info["plan_samples"] = action_info["plan_samples"].to_dict()
        return ego_actions, action_info
    
    def update_guidance_config(self, guidance_config_list):
        self.guidance.set_guidance(guidance_config_list)

# --------------------------------------------------------------------------------
class AgentCentricToSceneCentricWrapper(object):
    '''
    Note: only support batch size 1 for now.

    1.takes in agent-centric data and convert it to scene-centric data
    2.apply policy get_action
    3.convert scene-centric actions back to agent-centric actions and return
    '''
    def __init__(self, policy):
        self.device = policy.device
        self.policy = policy
        self.model = policy.model

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        # convert agent-centric data to scene-centric data
        ego_obs, agent_obs = self.split_ego_and_agent_obs(obs)
        scene_obs = self.convert_agent_centric_obs_to_scene_centric(ego_obs, agent_obs)

        # TBD: temporary hack for debugging
        actions_scene_centric, action_info_scene_centric = self.policy.get_action(scene_obs, agent_obs_gt=obs, **kwargs)

        # convert scene-centric actions to agent-centric actions
        actions, action_info = self.convert_scene_centric_action_to_agent_centric(actions_scene_centric, action_info_scene_centric, ego_obs, agent_obs)

        # TBD: for debugging
        # check_action_consistency(actions, action_info_scene_centric['action_agent_gt'])
        
        return actions, action_info
    
    @staticmethod
    def split_ego_and_agent_obs(obs):
        # split ego and agent observation
        # Assume the first element in obs is ego (we can also use other criteria to set ego)
        # TBD: support multiple scenes
        ego_obs = dict()
        agent_obs = dict()
        for k,v in obs.items():
            if v is not None:
                if k not in ['history_pad_dir', 'extras']:
                    ego_obs[k] = v[:1]
                    agent_obs[k] = v[1:]
                elif k == 'extras':
                    ego_obs[k], agent_obs[k] = dict(), dict()
                    for sub_k, sub_v in v.items():
                        ego_obs[k][sub_k] = sub_v[:1]
                        agent_obs[k][sub_k] = sub_v[1:]
        return ego_obs, agent_obs

    @staticmethod
    def convert_agent_centric_obs_to_scene_centric(ego_obs, agent_obs, max_num_agents=None):
        # TBD: handle scene_index for multiple scenes in a batch
        num_agents = [sum(agent_obs["scene_index"]==i)+1 for i in ego_obs["scene_index"]]
        num_agents = torch.tensor(num_agents,device=ego_obs["scene_index"].device)
        hist_pos_b = list()
        hist_yaw_b = list()
        hist_speeds_b = list()
        hist_avail_b = list()
        fut_pos_b = list()
        fut_yaw_b = list()
        fut_avail_b = list()
        agent_from_center_b = list()
        center_from_agent_b = list()
        raster_from_center_b = list()
        center_from_raster_b = list()
        raster_from_world_b = list()
        maps_b = list()
        curr_speed_b = list()
        type_b = list()
        extents_b = list()
        scene_index_b = list()
        neighbor_indices_b = list()
        closest_lanes_points_b = list()
        extras = dict()

        full_fut_traj_b = list()
        full_fut_valid_b = list()
        agent_hist_b = list()

        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            center_from_agent = ego_obs["agent_from_world"][i].unsqueeze(0) @ agent_obs["world_from_agent"][agent_idx]
            center_from_agents = torch.cat((torch.eye(3,device=center_from_agent.device).unsqueeze(0),center_from_agent),0)
            ego_and_agent_yaw = torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0) # [M]
            hist_pos_raw = torch.cat((ego_obs["history_positions"][i:i+1],agent_obs["history_positions"][agent_idx]),0)
            hist_yaw_raw = torch.cat((ego_obs["history_yaws"][i:i+1],agent_obs["history_yaws"][agent_idx]),0)
            
            agents_hist_avail = torch.cat((ego_obs["history_availabilities"][i:i+1],agent_obs["history_availabilities"][agent_idx]),0)
            agents_hist_pos = GeoUtils.transform_points_tensor(hist_pos_raw,center_from_agents)*agents_hist_avail.unsqueeze(-1)
            agents_hist_yaw = (hist_yaw_raw+ego_and_agent_yaw[:, None, None]-ego_obs["yaw"][i:i+1])*agents_hist_avail.unsqueeze(-1)
            agents_hist_speeds = torch.cat((ego_obs["history_speeds"][i:i+1],agent_obs["history_speeds"][agent_idx]),0)*agents_hist_avail

            if 'closest_lane_point' in ego_obs['extras']:
                closest_lane_point = torch.cat((ego_obs['extras']['closest_lane_point'][i:i+1], agent_obs['extras']['closest_lane_point'][agent_idx]),0)
                M, S_seg, S_p, _ = closest_lane_point.shape
                # [M, S_seg, S_p, 3] -> [M, S_seg*S_p, 3]
                closest_lane_point = closest_lane_point.reshape(M, S_seg*S_p, -1)
                closest_lane_point_pos = closest_lane_point[..., :2] # [M, S_seg*S_p, 2]
                closest_lane_point_yaw = closest_lane_point[..., [2]] # [M, S_seg*S_p, 1]

                closest_lane_point_pos_transformed = GeoUtils.transform_points_tensor(closest_lane_point_pos,center_from_agents)
                closest_lane_point_yaw_transformed = angle_wrap(closest_lane_point_yaw+ego_and_agent_yaw[:, None, None]-ego_obs["yaw"][i:i+1])

                closest_lanes_points = torch.cat((closest_lane_point_pos_transformed, closest_lane_point_yaw_transformed), -1).reshape(M, S_seg, S_p, -1)
                
                closest_lanes_points_b.append(closest_lanes_points)
            
            # TBD: warning: currently we only do the conversion for x and y in full_fut_traj but not other fields as those are not used but those should be converted as well!!!
            if 'full_fut_valid' in ego_obs['extras'] and 'full_fut_traj' in ego_obs['extras']:
                full_fut_valid = torch.cat((ego_obs['extras']['full_fut_valid'][i:i+1],agent_obs['extras']['full_fut_valid'][agent_idx]),0)
                full_fut_valid_b.append(full_fut_valid)
                full_fut_traj = torch.cat((ego_obs['extras']['full_fut_traj'][i:i+1],agent_obs['extras']['full_fut_traj'][agent_idx]),0)

                full_fut_traj[...,:2] = GeoUtils.transform_points_tensor(full_fut_traj[...,:2],center_from_agents)*full_fut_valid.unsqueeze(-1)
                full_fut_traj_b.append(full_fut_traj)
            
            # TBD: warning: currently we only do the conversion for x, y, and yaw in agent_hist but not other fields as those are not used but those should be converted as well!!!
            agent_hist = torch.cat((ego_obs["agent_hist"][i:i+1],agent_obs["agent_hist"][agent_idx]),0)
            agent_hist[...,:2] = agents_hist_pos
            agent_hist[...,[2]] = agents_hist_yaw 
            agent_hist_b.append(agent_hist)
            
            
            hist_pos_b.append(agents_hist_pos)
            hist_yaw_b.append(agents_hist_yaw)
            hist_speeds_b.append(agents_hist_speeds)
            hist_avail_b.append(agents_hist_avail)
            if agent_obs["target_availabilities"].shape[1]<ego_obs["target_availabilities"].shape[1]:
                pad_shape=(agent_obs["target_availabilities"].shape[0],ego_obs["target_availabilities"].shape[1]-agent_obs["target_availabilities"].shape[1])
                agent_obs["target_availabilities"] = torch.cat((agent_obs["target_availabilities"],torch.zeros(pad_shape,device=agent_obs["target_availabilities"].device)),1)
            agents_fut_avail = torch.cat((ego_obs["target_availabilities"][i:i+1],agent_obs["target_availabilities"][agent_idx]),0)
            if agent_obs["target_positions"].shape[1]<ego_obs["target_positions"].shape[1]:
                pad_shape=(agent_obs["target_positions"].shape[0],ego_obs["target_positions"].shape[1]-agent_obs["target_positions"].shape[1],*agent_obs["target_positions"].shape[2:])
                agent_obs["target_positions"] = torch.cat((agent_obs["target_positions"],torch.zeros(pad_shape,device=agent_obs["target_positions"].device)),1)
                pad_shape=(agent_obs["target_yaws"].shape[0],ego_obs["target_yaws"].shape[1]-agent_obs["target_yaws"].shape[1],*agent_obs["target_yaws"].shape[2:])
                agent_obs["target_yaws"] = torch.cat((agent_obs["target_yaws"],torch.zeros(pad_shape,device=agent_obs["target_yaws"].device)),1)
            fut_pos_raw = torch.cat((ego_obs["target_positions"][i:i+1],agent_obs["target_positions"][agent_idx]),0)
            fut_yaw_raw = torch.cat((ego_obs["target_yaws"][i:i+1],agent_obs["target_yaws"][agent_idx]),0)
            agents_fut_pos = GeoUtils.transform_points_tensor(fut_pos_raw,center_from_agents)*agents_fut_avail.unsqueeze(-1)
            agents_fut_yaw = (fut_yaw_raw+torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)[:,None,None]-ego_obs["yaw"][i])*agents_fut_avail.unsqueeze(-1)
            fut_pos_b.append(agents_fut_pos)
            fut_yaw_b.append(agents_fut_yaw)
            fut_avail_b.append(agents_fut_avail)

            curr_yaw = agents_hist_yaw[:,-1]
            curr_pos = agents_hist_pos[:,-1]
            agents_from_center = GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros_like(curr_pos))@GeoUtils.transform_matrices(torch.zeros_like(curr_yaw).flatten(),-curr_pos)
                             
            # raster_from_center = centered_raster_from_agent @ agents_from_center
            center_from_raster = center_from_agents @ ego_obs["agent_from_raster"]
            # raster_from_world = torch.cat((ego_obs["raster_from_world"][i:i+1],agent_obs["raster_from_world"][agent_idx]),0)


            agent_from_center_b.append(agents_from_center)
            center_from_agent_b.append(center_from_agents)
            # raster_from_center_b.append(raster_from_center)
            center_from_raster_b.append(center_from_raster)
            # raster_from_world_b.append(raster_from_world)

            maps = torch.cat((ego_obs["image"][i:i+1],agent_obs["image"][agent_idx]),0)
            curr_speed = torch.cat((ego_obs["curr_speed"][i:i+1],agent_obs["curr_speed"][agent_idx]),0)
            agents_type = torch.cat((ego_obs["type"][i:i+1],agent_obs["type"][agent_idx]),0)
            agents_extent = torch.cat((ego_obs["extent"][i:i+1],agent_obs["extent"][agent_idx]),0)
            scene_index = torch.cat((ego_obs["scene_index"][i:i+1],agent_obs["scene_index"][agent_idx]),0)
            neighbor_indices = torch.cat((ego_obs["all_other_agents_indices"][i:i+1],agent_obs["all_other_agents_indices"][agent_idx]),0)
            maps_b.append(maps)
            curr_speed_b.append(curr_speed)
            type_b.append(agents_type)
            extents_b.append(agents_extent)
            scene_index_b.append(scene_index)
            neighbor_indices_b.append(neighbor_indices)

        extras['closest_lane_point'] = pad_sequence(closest_lanes_points_b,batch_first=True,padding_value=0)
        extras['full_fut_traj'] = pad_sequence(full_fut_traj_b,batch_first=True,padding_value=0)
        extras['full_fut_valid'] = pad_sequence(full_fut_valid_b,batch_first=True,padding_value=0)
        agent_hist = pad_sequence(agent_hist_b,batch_first=True,padding_value=0)
        scene_obs = dict(
            num_agents=num_agents,
            image=pad_sequence(maps_b,batch_first=True,padding_value=0),
            map_names=ego_obs["map_names"],
            drivable_map=ego_obs["drivable_map"],
            target_positions=pad_sequence(fut_pos_b,batch_first=True,padding_value=0),
            target_yaws=angle_wrap(pad_sequence(fut_yaw_b,batch_first=True,padding_value=0)),
            target_availabilities=pad_sequence(fut_avail_b,batch_first=True,padding_value=0),
            history_positions=pad_sequence(hist_pos_b,batch_first=True,padding_value=0),
            history_yaws=angle_wrap(pad_sequence(hist_yaw_b,batch_first=True,padding_value=0)),
            history_speeds=pad_sequence(hist_speeds_b,batch_first=True,padding_value=0),
            history_availabilities=pad_sequence(hist_avail_b,batch_first=True,padding_value=0),
            curr_speed=pad_sequence(curr_speed_b,batch_first=True,padding_value=0),
            centroid=ego_obs["centroid"],
            yaw=angle_wrap(ego_obs["yaw"]),
            type=pad_sequence(type_b,batch_first=True,padding_value=0),
            extent=pad_sequence(extents_b,batch_first=True,padding_value=0),
            raster_from_agent=ego_obs["raster_from_agent"].squeeze(0),
            agent_from_raster=ego_obs["agent_from_raster"].squeeze(0),
            # raster_from_center=pad_sequence(raster_from_center_b,batch_first=True,padding_value=0),
            # center_from_raster=pad_sequence(center_from_raster_b,batch_first=True,padding_value=0),
            agents_from_center=pad_sequence(agent_from_center_b,batch_first=True,padding_value=0),
            center_from_agents=pad_sequence(center_from_agent_b,batch_first=True,padding_value=0),
            # raster_from_world=pad_sequence(raster_from_world_b,batch_first=True,padding_value=0),
            agent_from_world=ego_obs["agent_from_world"],
            world_from_agent=ego_obs["world_from_agent"],
            extras=extras,
            agent_hist=agent_hist,

            scene_index=pad_sequence(scene_index_b,batch_first=True,padding_value=-1),
            neighbor_indices=pad_sequence(neighbor_indices_b,batch_first=True,padding_value=0),

        )
        if max_num_agents is not None and scene_obs["num_agents"].max()>max_num_agents:
            dis = torch.norm(scene_obs["history_positions"][:,:,-1],dim=-1)
            dis = dis.masked_fill_(~scene_obs["history_availabilities"][...,-1],np.inf)
            idx = torch.argsort(dis,dim=1)[:,:max_num_agents]
            for k,v in scene_obs.items():
                if v.shape[:2]==dis.shape:
                    scene_obs[k] = TensorUtils.gather_from_start_single(scene_obs[k],idx)

        return scene_obs
    
    @staticmethod
    def convert_scene_centric_action_to_agent_centric(action, info, ego_obs, agent_obs):

        # Approach 1: Transform based on state
        pos_all = action.positions # (B, M, T, 2)
        yaw_all = action.yaws # (B, M, T, 1)
        pos_info_all = info['action_samples']['positions'] # (B, N, M, T, 2)
        yaw_info_all = info['action_samples']['yaws'] # (B, N, M, T, 1)

        pos_agent_list = []
        yaw_agent_list = []
        pos_info_agent_list = []
        yaw_info_agent_list = []


        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            pos = pos_all[i] # (M, T, 2)
            yaw = yaw_all[i] # (M, T, 1)
            pos_info = pos_info_all[i] # (N, M, T, 2)
            yaw_info = yaw_info_all[i] # (N, M, T, 1)

            N, M, T, _ = pos_info.shape

            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            agent_from_center = agent_obs["agent_from_world"][agent_idx] @ ego_obs["world_from_agent"][i].unsqueeze(0)
            agents_from_center = torch.cat((torch.eye(3,device=agent_from_center.device).unsqueeze(0),agent_from_center),0) # (M, 3, 3)

            ego_and_agent_yaw = torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)
            
            pos_agent = GeoUtils.transform_points_tensor(pos,agents_from_center)
            yaw_agent = yaw-ego_and_agent_yaw[:,None,None]+ego_obs["yaw"][i:i+1]
            
            agents_from_center_N = agents_from_center.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 3, 3)
            pos_info_agent = GeoUtils.transform_points_tensor(pos_info.reshape(N*M, T, 2),agents_from_center_N)
            yaw_info_agent = yaw_info.reshape(N*M, T, 1)-ego_and_agent_yaw[:,None,None].unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 1, 1)+ego_obs["yaw"][i:i+1]
            
            pos_info_agent = pos_info_agent.reshape(N, M, T, 2)
            yaw_info_agent = yaw_info_agent.reshape(N, M, T, 1)

            pos_agent_list.append(pos_agent)
            yaw_agent_list.append(yaw_agent)
            pos_info_agent_list.append(pos_info_agent)
            yaw_info_agent_list.append(yaw_info_agent)

        pos_agent_list = torch.cat(pos_agent_list,0)        
        yaw_agent_list = torch.cat(yaw_agent_list,0)
        pos_info_agent_list = torch.cat(pos_info_agent_list,0)
        yaw_info_agent_list = torch.cat(yaw_info_agent_list,0)
        actions = Action(
            positions=pos_agent_list,
            yaws=yaw_agent_list
        )
        infos = dict(
            action_samples=Action(
                positions=pos_info_agent_list.permute(1, 0, 2, 3), # (N, M, T, 2) -> (M, N, T, 2)
                yaws=yaw_info_agent_list.permute(1, 0, 2, 3), # (N, M, T, 1) -> (M, N, T, 1)
            ).to_dict(),
        )
        
        return actions, infos
        


        # # Approach 2: Transform based on action
        # traj = info['trajectories']
        # act_idx = info['act_idx']
        # dyn = info['dyn']
        # # acc, yawvel
        # x_action = traj[..., [4, 5]]

        # # form current state for agent-centric based on unicyle model
        # curr_speed = torch.cat((ego_obs["curr_speed"],agent_obs["curr_speed"]),0)
        # NM = curr_speed.shape
        # current_states = torch.zeros(*NM, 4).to(curr_speed.device)  # [x, y, vel, yaw]
        # current_states[..., 2] = curr_speed

        # x_action_dims = len(x_action.shape)
        # if x_action_dims == 5:
        #     B, N, M, T, _ = x_action.shape
        #     x_action = x_action.reshape(B * N, M, T, -1)
        #     # (M, 4) -> (B, M, 4) -> (B*N, M, 4) (assume B is 1)
        #     current_states = current_states.unsqueeze(0).repeat(N, 1, 1)

        # from tbsim.models.diffuser_helpers import unicyle_forward_dynamics
        # x_state = unicyle_forward_dynamics(
        #     dyn_model=dyn,
        #     initial_states=current_states,
        #     actions=x_action,
        #     step_time=0.1,
        #     mode='parallel'
        # )
        # # x, y, vel, yaw, acc, yawvel
        # x_all = torch.cat([x_state, x_action], dim=-1)
        # if x_action_dims == 5:
        #     x_all = x_all.reshape(B, N, M, T, -1)
        
        
        # B = x_all.shape[0]
        # x_all_selected = x_all[torch.arange(B), act_idx]
        
        # # squeeze out the batch dimension
        # x_pos = x_all[..., :2].squeeze(0)
        # x_yaw = x_all[..., 3:4].squeeze(0)
        # x_pos_selected = x_all_selected[..., :2].squeeze(0)
        # x_yaw_selected = x_all_selected[..., 3:4].squeeze(0)
        # new_info = dict(
        #     action_samples=Action(
        #         positions=x_pos.permute(1, 0, 2, 3), # (N, M, T, 2) -> (M, N, T, 2)
        #         yaws=x_yaw.permute(1, 0, 2, 3), # (N, M, T, 1) -> (M, N, T, 1)
        #     ).to_dict(),
        # )
        # new_action = Action(
        #     positions=x_pos_selected,
        #     yaws=x_yaw_selected
        # )

        
        # print('actions.positions[-1, -1]', actions.positions[-1, -1])
        # print('actions.yaws[-1, -1]', actions.yaws[-1, -1])

        # print('new_action.positions[-1, -1]', new_action.positions[-1, -1])
        # print('new_action.yaws[-1, -1]', new_action.yaws[-1, -1])

        # print('infos["action_samples"]["positions"][-1, -1, -1]', infos["action_samples"]["positions"][-1, -1, -1])
        # print('infos["action_samples"]["yaws"][-1, -1, -1]', infos["action_samples"]["yaws"][-1, -1, -1])

        # print('new_info["action_samples"]["positions"][-1, -1, -1]', new_info["action_samples"]["positions"][-1, -1, -1])
        # print('new_info["action_samples"]["yaws"][-1, -1, -1]', new_info["action_samples"]["yaws"][-1, -1, -1])

        # return new_action, new_info




# TBD: implement this (currently a placeholder)
class SceneCentricToAgentCentricWrapper(object):
    '''
    Note: only support batch size 1 for now.
    TBD: for debugging purpose. Because of reassign_values, this wrapper does not work when using independently (i.e., w.o. using the AgentCentricToSceneCentricWrapper)

    1.takes in scene-centric data and convert it to agent-centric data
    2.apply policy get_action
    3.convert agent-centric actions back to scene-centric actions and return
    '''
    def __init__(self, policy):
        self.device = policy.device
        self.policy = policy

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        from copy import deepcopy
        agent_centric_obs2 = convert_scene_data_to_agent_coordinates(obs)

        # reassign some values
        obs = self.reassign_values(obs)
        # convert scene-centric data to agent-centric data
        ego_obs, agent_obs = self.split_ego_and_agent_obs(obs)
        agent_centric_obs = self.convert_scene_centric_obs_to_agent_centric(ego_obs, agent_obs)

        # check if double conversion is correct
        if 'agent_obs_gt' in kwargs:
            agent_obs_gt = kwargs['agent_obs_gt']
            check_consistency(major_keywords+prep_keywords, agent_centric_obs2, agent_obs_gt, agent_obs_in_scene_format=True)
            check_consistency(full_keywords, agent_centric_obs, agent_obs_gt)
                        
        actions_agent_centric, action_info_agent_centric = self.policy.get_action(agent_centric_obs, **kwargs)

        # convert agent-centric actions to scene-centric actions
        actions, action_info = self.convert_agent_centric_action_to_scene_centric(actions_agent_centric, action_info_agent_centric, ego_obs, agent_obs)

        # TBD: for debugging
        action_info['action_agent_gt'] = actions_agent_centric
        
        return actions, action_info
    @staticmethod
    def reassign_values(obs):
        new_obs = convert_scene_obs_to_agent_prep(obs)
        for k, v in new_obs.items():
            obs[k] = v
        
        # TBD: hacky picking the only first scene in the batch
        for k, v in obs.items():
            if k == 'extras':
                for sub_k, sub_v in v.items():
                    obs[k][sub_k] = sub_v[0]
            # TBD: hacky way to manually avoid scene_index
            else:
                obs[k] = obs[k][0]
        return obs

    @staticmethod
    def split_ego_and_agent_obs(obs):
        # split ego and agent observation
        # Assume the first element in obs is ego (we can also use other criteria to set ego)
        ego_obs = dict()
        agent_obs = dict()
        for k,v in obs.items():
            if v is not None:
                if k not in ['history_pad_dir', 'extras', 'num_agents']:
                    ego_obs[k] = v[:1]
                    agent_obs[k] = v[1:]
                elif k == 'extras':
                    for sub_k in v:
                        ego_obs[k], agent_obs[k] = dict(), dict()
                        ego_obs[k][sub_k] = v[sub_k][:1]
                        agent_obs[k][sub_k] = v[sub_k][1:]
        return ego_obs, agent_obs

    @staticmethod
    def convert_scene_centric_obs_to_agent_centric(ego_obs,agent_obs):
        hist_pos_b = list()
        hist_yaw_b = list()
        hist_speeds_b = list()
        hist_avail_b = list()
        fut_pos_b = list()
        fut_yaw_b = list()
        fut_avail_b = list()
        maps_b = list()
        curr_speed_b = list()
        type_b = list()
        extents_b = list()
        closest_lanes_points_b = list()
        extras = dict()

        neigh_indices_b = list()
        hist_neigh_pos_b = list()
        hist_neigh_yaw_b = list()
        hist_neigh_speeds_b = list()
        hist_neigh_avail_b = list()
        neigh_curr_speed_b = list()
        fut_neigh_pos_b = list()
        fut_neigh_yaw_b = list()
        fut_neigh_avail_b = list()
        neigh_type_b = list()
        neigh_extents_b = list()

        
        # TBD: implement the case when assuming one scene
        for i,scene_idx in enumerate(ego_obs["scene_index"]):
            agent_idx = torch.where(agent_obs["scene_index"]==scene_idx)[0]
            agent_from_center = agent_obs["agent_from_world"][agent_idx] @ ego_obs["world_from_agent"][i].unsqueeze(0)
            agents_from_center = torch.cat((torch.eye(3,device=agent_from_center.device).unsqueeze(0),agent_from_center),0) # (M, 3, 3)
            ego_and_agent_yaw = torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0) # [M]
            hist_pos_raw = torch.cat((ego_obs["history_positions"][i:i+1],agent_obs["history_positions"][agent_idx]),0)
            hist_yaw_raw = torch.cat((ego_obs["history_yaws"][i:i+1],agent_obs["history_yaws"][agent_idx]),0)
            
            agents_hist_avail = torch.cat((ego_obs["history_availabilities"][i:i+1],agent_obs["history_availabilities"][agent_idx]),0)
            agents_hist_pos = GeoUtils.transform_points_tensor(hist_pos_raw,agents_from_center)*agents_hist_avail.unsqueeze(-1)
            agents_hist_yaw = (hist_yaw_raw-ego_and_agent_yaw[:, None, None]+ego_obs["yaw"][i:i+1])*agents_hist_avail.unsqueeze(-1)
            agents_hist_speeds = torch.cat((ego_obs["history_speeds"][i:i+1],agent_obs["history_speeds"][agent_idx]),0)*agents_hist_avail

            if 'closest_lane_point' in ego_obs['extras']:
                closest_lane_point = torch.cat((ego_obs['extras']['closest_lane_point'][i:i+1], agent_obs['extras']['closest_lane_point'][agent_idx]),0)
                M, S_seg, S_p, _ = closest_lane_point.shape
                # [M, S_seg, S_p, 3] -> [M, S_seg*S_p, 3]
                closest_lane_point = closest_lane_point.reshape(M, S_seg*S_p, -1)
                closest_lane_point_pos = closest_lane_point[..., :2] # [M, S_seg*S_p, 2]
                closest_lane_point_yaw = closest_lane_point[..., [2]] # [M, S_seg*S_p, 1]

                closest_lane_point_pos_transformed = GeoUtils.transform_points_tensor(closest_lane_point_pos,agents_from_center)
                closest_lane_point_yaw_transformed = angle_wrap(closest_lane_point_yaw-ego_and_agent_yaw[:, None, None]+ego_obs["yaw"][i:i+1])

                closest_lanes_points = torch.cat((closest_lane_point_pos_transformed, closest_lane_point_yaw_transformed), -1).reshape(M, S_seg, S_p, -1)
                
                closest_lanes_points_b.append(closest_lanes_points)

            
            hist_pos_b.append(agents_hist_pos)
            hist_yaw_b.append(agents_hist_yaw)
            hist_speeds_b.append(agents_hist_speeds)
            hist_avail_b.append(agents_hist_avail)
            if agent_obs["target_availabilities"].shape[1]<ego_obs["target_availabilities"].shape[1]:
                pad_shape=(agent_obs["target_availabilities"].shape[0],ego_obs["target_availabilities"].shape[1]-agent_obs["target_availabilities"].shape[1])
                agent_obs["target_availabilities"] = torch.cat((agent_obs["target_availabilities"],torch.zeros(pad_shape,device=agent_obs["target_availabilities"].device)),1)
            agents_fut_avail = torch.cat((ego_obs["target_availabilities"][i:i+1],agent_obs["target_availabilities"][agent_idx]),0)
            if agent_obs["target_positions"].shape[1]<ego_obs["target_positions"].shape[1]:
                pad_shape=(agent_obs["target_positions"].shape[0],ego_obs["target_positions"].shape[1]-agent_obs["target_positions"].shape[1],*agent_obs["target_positions"].shape[2:])
                agent_obs["target_positions"] = torch.cat((agent_obs["target_positions"],torch.zeros(pad_shape,device=agent_obs["target_positions"].device)),1)
                pad_shape=(agent_obs["target_yaws"].shape[0],ego_obs["target_yaws"].shape[1]-agent_obs["target_yaws"].shape[1],*agent_obs["target_yaws"].shape[2:])
                agent_obs["target_yaws"] = torch.cat((agent_obs["target_yaws"],torch.zeros(pad_shape,device=agent_obs["target_yaws"].device)),1)
            fut_pos_raw = torch.cat((ego_obs["target_positions"][i:i+1],agent_obs["target_positions"][agent_idx]),0)
            fut_yaw_raw = torch.cat((ego_obs["target_yaws"][i:i+1],agent_obs["target_yaws"][agent_idx]),0)
            agents_fut_pos = GeoUtils.transform_points_tensor(fut_pos_raw,agents_from_center)*agents_fut_avail.unsqueeze(-1)
            agents_fut_yaw = (fut_yaw_raw-torch.cat((ego_obs["yaw"][i:i+1],agent_obs["yaw"][agent_idx]),0)[:,None,None]+ego_obs["yaw"][i])*agents_fut_avail.unsqueeze(-1)
            fut_pos_b.append(agents_fut_pos)
            fut_yaw_b.append(agents_fut_yaw)
            fut_avail_b.append(agents_fut_avail)

            curr_yaw = agents_hist_yaw[:,-1]
            curr_pos = agents_hist_pos[:,-1]
            agents_from_center = GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros_like(curr_pos))@GeoUtils.transform_matrices(torch.zeros_like(curr_yaw).flatten(),-curr_pos)
                                 
            maps = torch.cat((ego_obs["image"][i:i+1],agent_obs["image"][agent_idx]),0)
            curr_speed = torch.cat((ego_obs["curr_speed"][i:i+1],agent_obs["curr_speed"][agent_idx]),0)
            agents_type = torch.cat((ego_obs["type"][i:i+1],agent_obs["type"][agent_idx]),0)
            agents_extent = torch.cat((ego_obs["extent"][i:i+1],agent_obs["extent"][agent_idx]),0)
            maps_b.append(maps)
            curr_speed_b.append(curr_speed)
            type_b.append(agents_type)
            extents_b.append(agents_extent)

            # neighbor
            num_agents_i = agent_idx.shape[0]+1
            ego_and_agent_idx = torch.cat((torch.tensor([0], device=agent_idx.device), agent_idx+1), 0)
            neighbor_indices = torch.cat((ego_obs["neighbor_indices"][i:i+1], agent_obs["neighbor_indices"][agent_idx]), 0)
            agent_from_world = torch.cat([ego_obs["agent_from_world"][i].unsqueeze(0), agent_obs["agent_from_world"][agent_idx]], 0)
            world_from_agent = torch.cat([ego_obs["world_from_agent"][i].unsqueeze(0), agent_obs["world_from_agent"][agent_idx]], 0)
            
            hist_neigh_pos_b_sub = []
            hist_neigh_yaw_b_sub = []
            hist_neigh_speeds_b_sub = []
            hist_neigh_avail_b_sub = []
            neigh_curr_speed_b_sub = []
            fut_neigh_pos_b_sub = []
            fut_neigh_yaw_b_sub = []
            fut_neigh_avail_b_sub = []
            neigh_type_b_sub = []
            neigh_extents_b_sub = []

            for ci in ego_and_agent_idx:
                neigh_inds = neighbor_indices[ci]
                chosen_neigh_inds = neigh_inds[neigh_inds>=0].long()
                # print('chosen_neigh_inds', chosen_neigh_inds)

                index_agent = lambda x: x[ci] if x is not None else None
                index_neighbors = lambda x: x[chosen_neigh_inds]
                center_from_world = index_agent(agent_from_world)
                world_from_neigh = index_neighbors(world_from_agent)

                # (Q. 3. 3)
                center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh

                hist_neigh_pos_b_sub_ci = index_neighbors(agents_hist_pos)
                hist_neigh_yaw_b_sub_ci = index_neighbors(agents_hist_yaw)
                hist_neigh_speeds_b_sub_ci = index_neighbors(agents_hist_speeds)
                hist_neigh_avail_b_sub_ci = index_neighbors(agents_hist_avail)
                neigh_curr_speed_b_sub_ci = index_neighbors(curr_speed)
                fut_neigh_pos_b_sub_ci = index_neighbors(agents_fut_pos)
                fut_neigh_yaw_b_sub_ci = index_neighbors(agents_fut_yaw)
                fut_neigh_avail_b_sub_ci = index_neighbors(agents_fut_avail)
                neigh_type_b_sub_ci = index_neighbors(agents_type)
                neigh_extents_b_sub_ci = index_neighbors(agents_extent)



                hist_neigh_pos_b_sub_ci = GeoUtils.transform_points_tensor(hist_neigh_pos_b_sub_ci,center_from_neigh)*hist_neigh_avail_b_sub_ci.unsqueeze(-1)
                
                ego_yaw_ci, agent_yaw_ci = index_agent(ego_and_agent_yaw), index_neighbors(ego_and_agent_yaw)
                
                hist_neigh_yaw_b_sub_ci = (hist_neigh_yaw_b_sub_ci+agent_yaw_ci[:, None, None]-ego_yaw_ci)*hist_neigh_avail_b_sub_ci.unsqueeze(-1)
                hist_neigh_speeds_b_sub_ci = hist_neigh_speeds_b_sub_ci*hist_neigh_avail_b_sub_ci
                neigh_curr_speed_b_sub_ci = neigh_curr_speed_b_sub_ci
                
                fut_neigh_pos_b_sub_ci = GeoUtils.transform_points_tensor(fut_neigh_pos_b_sub_ci,center_from_neigh)*fut_neigh_avail_b_sub_ci.unsqueeze(-1)
                fut_neigh_yaw_b_sub_ci = (fut_neigh_yaw_b_sub_ci+agent_yaw_ci[:, None, None]-ego_yaw_ci)*fut_neigh_avail_b_sub_ci.unsqueeze(-1)
                neigh_type_b_sub_ci = neigh_type_b_sub_ci
                neigh_extents_b_sub_ci = neigh_extents_b_sub_ci

                hist_neigh_pos_b_sub.append(hist_neigh_pos_b_sub_ci)
                hist_neigh_yaw_b_sub.append(hist_neigh_yaw_b_sub_ci)
                hist_neigh_speeds_b_sub.append(hist_neigh_speeds_b_sub_ci)
                hist_neigh_avail_b_sub.append(hist_neigh_avail_b_sub_ci)
                neigh_curr_speed_b_sub.append(neigh_curr_speed_b_sub_ci)
                fut_neigh_pos_b_sub.append(fut_neigh_pos_b_sub_ci)
                fut_neigh_yaw_b_sub.append(fut_neigh_yaw_b_sub_ci)
                fut_neigh_avail_b_sub.append(fut_neigh_avail_b_sub_ci)
                neigh_type_b_sub.append(neigh_type_b_sub_ci)
                neigh_extents_b_sub.append(neigh_extents_b_sub_ci)

            neigh_indices_b.append(neighbor_indices)
            hist_neigh_pos_b.append(pad_sequence(hist_neigh_pos_b_sub, batch_first=True, padding_value=np.nan))
            hist_neigh_yaw_b.append(pad_sequence(hist_neigh_yaw_b_sub, batch_first=True, padding_value=np.nan))
            hist_neigh_speeds_b.append(pad_sequence(hist_neigh_speeds_b_sub, batch_first=True, padding_value=np.nan))
            hist_neigh_avail_b.append(pad_sequence(hist_neigh_avail_b_sub, batch_first=True, padding_value=0))
            neigh_curr_speed_b.append(pad_sequence(neigh_curr_speed_b_sub, batch_first=True, padding_value=np.nan))
            fut_neigh_pos_b.append(pad_sequence(fut_neigh_pos_b_sub, batch_first=True, padding_value=np.nan))
            fut_neigh_yaw_b.append(pad_sequence(fut_neigh_yaw_b_sub, batch_first=True, padding_value=np.nan))
            fut_neigh_avail_b.append(pad_sequence(fut_neigh_avail_b_sub, batch_first=True, padding_value=0))
            neigh_type_b.append(pad_sequence(neigh_type_b_sub, batch_first=True, padding_value=0))
            neigh_extents_b.append(pad_sequence(neigh_extents_b_sub, batch_first=True, padding_value=np.nan))
            # # # #

        extras['closest_lane_point'] = torch.cat(closest_lanes_points_b,0)
        agent_obs = dict(
            image=torch.cat(maps_b,0),
            target_positions=torch.cat(fut_pos_b,0),
            target_yaws=angle_wrap(torch.cat(fut_yaw_b,0)),
            target_availabilities=torch.cat(fut_avail_b,0),
            history_positions=torch.cat(hist_pos_b,0),
            history_yaws=angle_wrap(torch.cat(hist_yaw_b,0)),
            history_speeds=torch.cat(hist_speeds_b,0),
            history_availabilities=torch.cat(hist_avail_b,0),
            curr_speed=torch.cat(curr_speed_b,0),
            centroid=torch.cat([ego_obs["centroid"], agent_obs["centroid"]], dim=0),
            yaw=angle_wrap(torch.cat([ego_obs["yaw"], agent_obs["yaw"]], dim=0),
            type=torch.cat(type_b,0)),
            extent=torch.cat(extents_b,0),
            raster_from_agent=torch.cat([ego_obs["raster_from_agent"], agent_obs["raster_from_agent"]], dim=0),
            agent_from_raster=torch.cat([ego_obs["agent_from_raster"], agent_obs["agent_from_raster"]], dim=0),
            agent_from_world=torch.cat([ego_obs["agent_from_world"], agent_obs["agent_from_world"]], dim=0),
            world_from_agent=torch.cat([ego_obs["world_from_agent"], agent_obs["world_from_agent"]], dim=0),
            extras=extras,

            # [M, Q, T, 2]
            all_other_agents_indices=torch.cat(neigh_indices_b,0),
            all_other_agents_history_positions=torch.cat(hist_neigh_pos_b,0),
            # [M, Q, T, 1]
            all_other_agents_history_yaws=angle_wrap(torch.cat(hist_neigh_yaw_b,0)),
            # [M, Q, T]
            all_other_agents_history_speeds=torch.cat(hist_neigh_speeds_b,0),
            # [M, Q, T]
            all_other_agents_history_availabilities=torch.cat(hist_neigh_avail_b,0),
            all_other_agents_history_availability=torch.cat(hist_neigh_avail_b,0),
            
            # [M, Q]
            all_other_agents_curr_speed=torch.cat(neigh_curr_speed_b,0),
            
            # [M, Q, T, 2]
            all_other_agents_future_positions=torch.cat(fut_neigh_pos_b,0),
            # [M, Q, T]
            all_other_agents_future_yaws=angle_wrap(torch.cat(fut_neigh_yaw_b,0)),
            # [M, Q, T]
            all_other_agents_future_availability=torch.cat(fut_neigh_avail_b,0),
            
            # [M, Q]
            all_other_agents_types=torch.cat(neigh_type_b,0),
            # [M, Q, E]
            all_other_agents_extents=torch.cat(neigh_extents_b,0),
            # [M]
            scene_index=torch.cat([ego_obs["scene_index"], agent_obs["scene_index"]],0),
        )

        return agent_obs
    
    @staticmethod
    def convert_agent_centric_action_to_scene_centric(action, info, ego_obs, agent_obs):
        # TBD: assume one scene
        pos = action.positions # (M, T, 2)
        yaw = action.yaws # (M, T, 1)
        pos_info = info['action_samples']['positions'].permute(1, 0, 2, 3) # (N, M, T, 2)
        yaw_info = info['action_samples']['yaws'].permute(1, 0, 2, 3) # (N, M, T, 1)
        N, M, T, _ = pos_info.shape

        # (1, 3, 3) @ (M-1, 3, 3) -> (M-1, 3, 3)
        center_from_agent = ego_obs["agent_from_world"] @ agent_obs["world_from_agent"]
        # (M, 3, 3)
        center_from_agents = torch.cat((torch.eye(3,device=center_from_agent.device).unsqueeze(0),center_from_agent),0) 

        ego_and_agent_yaw = torch.cat((ego_obs["yaw"],agent_obs["yaw"]),0)
        
        pos_agent = GeoUtils.transform_points_tensor(pos,center_from_agents)
        yaw_agent = yaw+ego_and_agent_yaw[:,None,None]-ego_obs["yaw"]
        
        center_from_agents_N = center_from_agents.unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 3, 3)
        pos_info_agent = GeoUtils.transform_points_tensor(pos_info.reshape(N*M, T, 2),center_from_agents_N)
        yaw_info_agent = yaw_info.reshape(N*M, T, 1)+ego_and_agent_yaw[:,None,None].unsqueeze(0).repeat(N, 1, 1, 1).reshape(N*M, 1, 1)-ego_obs["yaw"]
        
        pos_info_agent = pos_info_agent.reshape(N, M, T, 2)
        yaw_info_agent = yaw_info_agent.reshape(N, M, T, 1)

        actions = Action(
            positions=pos_agent.unsqueeze(0), # (M, T, 2) -> (B, M, T, 2)
            yaws=yaw_agent.unsqueeze(0), # (M, T, 2) -> (B, M, T, 2)
        )
        infos = dict(
            action_samples=Action(
                positions=pos_info_agent.unsqueeze(0), # (N, M, T, 2) -> (B, N, M, T, 2)
                yaws=yaw_info_agent.unsqueeze(0), # (N, M, T, 1) -> (B, N, M, T, 1)
            ).to_dict(),
        )
        
        return actions, infos
        


class NaiveAgentCentricToSceneCentricWrapper(object):
    '''
    Use agent-centric coordinates for scene-centric model
    Used to debug scene-centric model with agent-centric data

    Note: only support batch size 1 for now.

    1.takes in agent-centric data and convert it to scene-centric data without coordinate transformation
    2.apply policy get_action
    3.convert scene-centric actions back to agent-centric actions without coordinate transformation and return
    '''
    def __init__(self, policy):
        self.device = policy.device
        self.policy = policy
        self.model = policy.model

    def eval(self):
        self.policy.eval()

    def get_action(self, obs, **kwargs) -> Tuple[Action, Dict]:
        scene_obs = self.convert_agent_centric_obs_to_scene_centric(obs)

        actions, action_info = self.policy.get_action(scene_obs, **kwargs)

        # convert scene-centric actions to agent-centric actions
        actions, action_info = self.convert_scene_centric_action_to_agent_centric(actions, action_info)
        
        return actions, action_info

    @staticmethod
    def convert_agent_centric_obs_to_scene_centric(agent_centric_obs):
        scene_obs = add_scene_dim_to_agent_data(agent_centric_obs)
        scene_obs["num_agents"] = torch.tensor([agent_centric_obs['target_positions'].shape[0]],device=agent_centric_obs['target_positions'].device)

        return scene_obs
    
    @staticmethod
    def convert_scene_centric_action_to_agent_centric(action, info):

        pos_all = action.positions # (B, M, T, 2)
        yaw_all = action.yaws # (B, M, T, 1)
        pos_info_all = info['action_samples']['positions'] # (B, N, M, T, 2)
        yaw_info_all = info['action_samples']['yaws'] # (B, N, M, T, 1)

        actions = Action(
            positions=pos_all[0],
            yaws=yaw_all[0],
        )
        infos = dict(
            action_samples=Action(
                positions=pos_info_all[0].permute(1, 0, 2, 3), # (N, M, T, 2) -> (M, N, T, 2)
                yaws=yaw_info_all[0].permute(1, 0, 2, 3), # (N, M, T, 1) -> (M, N, T, 1)
            ).to_dict(),
            attn_weights=info['attn_weights'],
        )
        
        return actions, infos

