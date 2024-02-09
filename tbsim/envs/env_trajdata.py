from dataclasses import asdict
from posixpath import split
from math import ceil, floor
from tbsim.utils.scene_edit_utils import scene_to_video
import torch
import numpy as np
from copy import deepcopy
from typing import List
from trajdata import UnifiedDataset, AgentBatch, AgentType
from trajdata.simulation import SimulationScene
from trajdata.simulation import sim_metrics
from trajdata.data_structures.state import StateArray  # Just for type annotations
# from scripts.parse_results import parse

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.vis_utils import render_state_trajdata
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from tbsim.policies.common import RolloutAction, Action
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.timer import Timers
from tbsim.utils.trajdata_utils import parse_trajdata_batch, get_drivable_region_map, verify_map
from tbsim.utils.rollout_logger import RolloutLogger
from torch.nn.utils.rnn import pad_sequence
from trajdata.data_structures.state import StateArray

agent_types=[AgentType.UNKNOWN,AgentType.VEHICLE,AgentType.PEDESTRIAN,AgentType.BICYCLE,AgentType.MOTORCYCLE]

class EnvUnifiedSimulation(BaseEnv, BatchedEnv):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            log_data=True,
            renderer=None,
            save_action_samples=False,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = dataset.num_scenes()
        self._num_scenes = num_scenes

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes
        self._current_scene_indices = None

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self._cached_raw_observation = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._persistent_metrics = self._metrics
        self._log_data = log_data
        self.logger = None

        self.save_action_samples = save_action_samples

    def update_random_seed(self, seed):
        self._npr = np.random.RandomState(seed=seed)

    @property
    def current_scene_names(self):
        return deepcopy([scene.scene.name for scene in self._current_scenes])

    @property
    def current_num_agents(self):
        return sum(len(scene.agents) for scene in self._current_scenes)

    def reset_multi_episodes_metrics(self):
        for v in self._metrics.values():
            v.multi_episode_reset()

    @property
    def current_agent_scene_index(self):
        si = []
        for scene_i, scene in zip(self.current_scene_index, self._current_scenes):
            si.extend([scene_i] * len(scene.agents))
        return np.array(si, dtype=np.int64)

    @property
    def current_agent_track_id(self):
        return np.arange(self.current_num_agents)

    @property
    def current_scene_index(self):
        return self._current_scene_indices.copy()

    @property
    def current_agent_names(self):
        names = []
        for scene in self._current_scenes:
            names.extend([a.name for a in scene.agents])
        return names

    @property
    def num_instances(self):
        return self._num_scenes

    @property
    def total_num_scenes(self):
        return self._num_total_scenes

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    @property
    def horizon(self):
        return self._env_config.simulation.num_simulation_steps

    def _disable_offroad_agents(self, scene):
        obs = scene.get_obs()
        obs = parse_trajdata_batch(obs)
        obs_maps = verify_map(obs["maps"])
        drivable_region = get_drivable_region_map(obs_maps)
        raster_pos = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        valid_agents = []
        for i, rpos in enumerate(raster_pos):
            if scene.agents[i].name == "ego" or drivable_region[i, int(rpos[1]), int(rpos[0])].item() > 0:
                valid_agents.append(scene.agents[i])

        scene.agents = valid_agents
    
    def add_new_agents(self,agent_data_by_scene):
        for sim_scene,agent_data in agent_data_by_scene.items():
            if sim_scene not in self._current_scenes:
                continue
            if len(agent_data)>0:
                sim_scene.add_new_agents(agent_data)

    def reset(self, scene_indices: List = None, start_frame_index = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
            start_frame_index (int or list of ints) : either a single frame number or a list of starting frames corresponding to the given scene_indices
        """
        if scene_indices is None:
            # randomly sample a batch of scenes for close-loop rollouts
            all_indices = np.arange(self._num_total_scenes)
            scene_indices = self._npr.choice(
                all_indices, size=(self.num_instances,), replace=False
            )

        scene_info = [self.dataset.get_scene(i) for i in scene_indices]

        self._num_scenes = len(scene_info)
        self._current_scene_indices = scene_indices
        assert (
                np.max(scene_indices) < self._num_total_scenes
                and np.min(scene_indices) >= 0
        )
        if start_frame_index is None:
            start_frame_index = self._env_config.simulation.start_frame_index
        self._current_scenes = []
        scenes_valid = []
        for i, si in enumerate(scene_info):
            try:
                cur_start_frame = start_frame_index[i] if isinstance(start_frame_index, list) else start_frame_index
                sim_scene: SimulationScene = SimulationScene(
                    env_name=self._env_config.name,
                    scene_name=si.name,
                    scene=si,
                    dataset=self.dataset,
                    init_timestep=cur_start_frame,
                    freeze_agents=True,
                    return_dict=True
                )
            except Exception as e:
                print('Invalid scene %s..., skipping' % (si.name))
                print(e)
                scenes_valid.append(False)
                continue
            
            obs = sim_scene.reset()
            self._disable_offroad_agents(sim_scene)
            self._current_scenes.append(sim_scene)
            scenes_valid.append(True)

        self._frame_index = 0
        self._cached_observation = None
        self._cached_raw_observation = None
        self._done = False

        obs_keys_to_log = [
            "centroid",
            "yaw",
            "extent",
            "world_from_agent",
            "scene_index",
            "track_id",
            "map_names",
        ]
        info_keys_to_log = [
            "action_samples",
            "diffusion_steps", # memory intensive
            "attn_weights",
        ]
        self.logger = RolloutLogger(obs_keys=obs_keys_to_log,
                                    info_keys=info_keys_to_log, save_action_samples=self.save_action_samples)

        for v in self._metrics.values():
            v.reset()

        return scenes_valid

    def render(self, actions_to_take):
        scene_ims = []
        # print("self.current_agent_names", self.current_agent_names)
        ego_inds = [i for i, name in enumerate(self.current_agent_names) if name == "ego"]
        for i in ego_inds:
            im = render_state_trajdata(
                batch=self.get_observation()["agents"],
                batch_idx=i,
                action=actions_to_take,
                rgb_idx_groups=self._env_config.rasterizer.rgb_idx_groups if "rgb_idx_groups" in self._env_config.rasterizer
                                      else None # backwards compat with old nusc
            )
            scene_ims.append(im)
        # print('len(scene_ims)', len(scene_ims))
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        agents = Action(
            positions=ac[:, :, :2],
            yaws=ac[:, :, 2:3]
        )

        return RolloutAction(agents=agents)

    def get_info(self):
        info = dict(scene_index=self.current_scene_names)
        if self._log_data:
            sim_buffer = self.logger.get_serialized_scene_buffer()
            sim_buffer = [sim_buffer[k] for k in self.current_scene_index]
            info["buffer"] = sim_buffer
            # self.logger.get_trajectory()
        return info

    def get_multi_episode_metrics(self):
        metrics = dict()
        for met_name, met in self._metrics.items():
            met_vals = met.get_multi_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            elif met_vals is not None:
                metrics[met_name] = met_vals

        return TensorUtils.detach(metrics)

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """
        metrics = dict()
        # get ADE and FDE from SimulationScene
        metrics["ade"] = np.zeros(self.num_instances)
        metrics["fde"] = np.zeros(self.num_instances)
        for i, scene in enumerate(self._current_scenes):
            mets_per_agent = scene.get_metrics([sim_metrics.ADE(), sim_metrics.FDE()])
            metrics["ade"][i] = np.array(list(mets_per_agent["ade"].values())).mean()
            metrics["fde"][i] = np.array(list(mets_per_agent["fde"].values())).mean()

        # aggregate per-step metrics
        for met_name, met in self._metrics.items():
            met_vals = met.get_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            else:
                metrics[met_name] = met_vals

        for k in metrics:
            assert metrics[k].shape == (self.num_instances,)
        return TensorUtils.detach(metrics)

    def get_observation_by_scene(self):
        obs = self.get_observation()["agents"]
        obs_by_scene = []
        obs_scene_index = self.current_agent_scene_index
        for i in range(self.num_instances):
            obs_by_scene.append(TensorUtils.map_ndarray(obs, lambda x: x[obs_scene_index == i]))
        return obs_by_scene

    def get_observation(self):
        def prepad_history(agent_obs, BM):
            # pad with zeros and set to unavaible
            agent_obs["history_positions"] = np.concatenate([np.zeros((*BM, pad_len, 2), dtype=agent_obs["history_positions"].dtype), agent_obs["history_positions"]], axis=1)
            agent_obs["history_yaws"] = np.concatenate([np.zeros((*BM, pad_len, 1), dtype=agent_obs["history_yaws"].dtype), agent_obs["history_yaws"]], axis=1)
            agent_obs["history_speeds"] = np.concatenate([np.zeros((*BM, pad_len), dtype=agent_obs["history_speeds"].dtype), agent_obs["history_speeds"]], axis=1)
            agent_obs["history_availabilities"] = np.concatenate([np.zeros((*BM, pad_len), dtype=agent_obs["history_availabilities"].dtype), agent_obs["history_availabilities"]], axis=1)

            N = agent_obs["all_other_agents_history_positions"].shape[1]
            agent_obs["all_other_agents_history_positions"] = np.concatenate([np.zeros((*BM, N, pad_len, 2), dtype=agent_obs["all_other_agents_history_positions"].dtype), agent_obs["all_other_agents_history_positions"]], axis=2)
            agent_obs["all_other_agents_history_yaws"] = np.concatenate([np.zeros((*BM, N, pad_len, 1), dtype=agent_obs["all_other_agents_history_yaws"].dtype), agent_obs["all_other_agents_history_yaws"]], axis=2)
            agent_obs["all_other_agents_history_speeds"] = np.concatenate([np.zeros((*BM, N, pad_len), dtype=agent_obs["all_other_agents_history_speeds"].dtype), agent_obs["all_other_agents_history_speeds"]], axis=2)
            agent_obs["all_other_agents_history_availabilities"] = np.concatenate([np.zeros((*BM, N, pad_len), dtype=agent_obs["all_other_agents_history_availabilities"].dtype), agent_obs["all_other_agents_history_availabilities"]], axis=2)
            agent_obs["all_other_agents_history_availability"] = np.concatenate([np.zeros((*BM, N, pad_len), dtype=agent_obs["all_other_agents_history_availability"].dtype), agent_obs["all_other_agents_history_availability"]], axis=2)
            agent_obs["all_other_agents_history_extents"] = np.concatenate([np.zeros((*BM, N, pad_len, 3), dtype=agent_obs["all_other_agents_history_extents"].dtype), agent_obs["all_other_agents_history_extents"]], axis=2)

        if self._cached_observation is not None:
            return self._cached_observation

        self.timers.tic("get_obs")

        raw_obs = []
        # print('len(self._current_scenes)', len(self._current_scenes))
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))

        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_trajdata_batch(agent_obs, overwrite_nan=False)
        agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
        agent_obs["scene_index"] = self.current_agent_scene_index
        agent_obs["track_id"] = self.current_agent_track_id
        agent_obs["env_name"] = [self._current_scenes[self.current_scene_index.index(i)].env_name for i in agent_obs["scene_index"]]

        # corner case where no agents in the scene are visible up to full history.
        #       so need to pad
        expected_hist_len = floor(self.dataset.history_sec[1] / self.dataset.desired_dt) + 1
        if "num_agents" in agent_obs:
            # scene centric
            # B, M, T, 2
            pad_len = expected_hist_len - agent_obs["history_positions"].shape[2]
            B, M = agent_obs["history_positions"].shape[:2]
            if pad_len > 0:
                prepad_history(agent_obs, [B, M])
        else:
            # agent centric
            # B, T, 2
            pad_len = expected_hist_len - agent_obs["history_positions"].shape[1]
            B = agent_obs["history_positions"].shape[0]
            if pad_len > 0:
                prepad_history(agent_obs, [B])

        # cache observations
        self._cached_observation = dict(agents=agent_obs)
        self.timers.toc("get_obs")

        return self._cached_observation


    def get_observation_skimp(self):
        self.timers.tic("obs_skimp")
        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False, get_map=False))
        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_trajdata_batch(agent_obs, overwrite_nan=False)
        agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
        agent_obs["scene_index"] = self.current_agent_scene_index
        agent_obs["track_id"] = self.current_agent_track_id
        self.timers.toc("obs_skimp")
        return dict(agents=agent_obs)

    def _add_per_step_metrics(self, obs, frame_index=None):
        for k, v in self._metrics.items():
            v.update_global_t(global_t=frame_index)
            # comment out since we now directly store np.nan in dataframe
            # hacky way to deal with invalid timesteps in GT
            # if self._frame_index > 0:
            #     xyh = np.concatenate([obs["centroid"], obs["yaw"][:,None]], axis=-1)
            #     nan_inds = np.where(xyh == np.zeros(3))[0]
            #     if len(nan_inds) > 0:
            #         import copy
            #         obs = copy.deepcopy(obs)
            #         obs["centroid"][nan_inds] = np.nan
            #         obs["yaw"][nan_inds] = np.nan
            v.add_step(obs, self.current_scene_index)

    def _step(self, step_actions: RolloutAction, num_steps_to_take):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")

        obs = self.get_observation()["agents"]   
        # record metrics
        # self._add_per_step_metrics(obs)

        action = step_actions.agents.to_dict()
        # action_samples = None if "action_samples" not in step_actions.agents_info else step_actions.agents_info["action_samples"]
        # action_info = {k : v for k, v in step_actions.agents_info.items() if k != "action_samples"}
        for action_index in range(num_steps_to_take):
            # print('action_index', action_index)
            # print("action['positions'].shape", action['positions'].shape)
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                return
            # # log state and action
            obs_skimp = self.get_observation_skimp()
            obs_skimp["agents"]["image"] = obs["image"]
            obs_skimp["agents"]["raster_from_world"] = obs["raster_from_world"]
            obs_skimp["agents"]["map_names"] = obs["map_names"]
            self._add_per_step_metrics(obs_skimp["agents"], self._frame_index+action_index)
            if self._log_data:
                # log_agents_info = action_info.copy()
                # if action_samples is not None:
                #     # need to truncate samples as well
                #     #       assuming action_samples is given as (B,N,T,D)
                #     #       swaps to (B,T,N,D) for logging
                #     log_agents_info["action_samples"] = TensorUtils.map_ndarray(action_samples, lambda x: np.swapaxes(x[:, :, action_index:], 1, 2))
                # if "diffusion_steps" in action_info:
                #     # have to swap N and T
                #     # NOTE: this does not truncate to the current action, only will visualize at planned steps
                #     # log_agents_info["diffusion_steps"] = TensorUtils.map_ndarray(log_agents_info["diffusion_steps"], 
                #     #                                                                       lambda x: np.swapaxes(x, 1, 2))
                #     pass

                action_to_log = RolloutAction(
                    agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                    agents_info=step_actions.agents_info
                )
                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    curr_yaw = obs["yaw"][idx]
                    curr_pos = obs["centroid"][idx]
                    h1, h2 = obs["yaw"][idx], obs["curr_agent_state"][idx, -1]
                    p1, p2 = obs["centroid"][idx], obs["curr_agent_state"][idx, :2]
                    assert np.all(h1[~np.isnan(h1)] == h2[~np.isnan(h2)])
                    assert np.all(p1[~np.isnan(p1)] == p2[~np.isnan(p2)])

                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                    if not np.any(np.isnan(action["positions"][idx, action_index])) and not np.any(np.isnan(action["yaws"][idx, action_index, 0])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                        next_state[2] = curr_yaw + action["yaws"][idx, action_index, 0]
                    else:
                        next_state = np.ones(3, dtype=obs["agent_fut"].dtype)*np.nan
                        print("invalid action!", idx, action_index, action["positions"][idx, action_index], action["yaws"][idx, action_index, 0])
                    scene_action[agent.name] = StateArray.from_array(next_state, "x,y,h")
                    idx += 1
                scene.step(scene_action, return_obs=False)

        self._cached_observation = None

        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take

    def step(self, actions: RolloutAction, num_steps_to_take: int = 1, render=False):
        """
        Step the simulation with control inputs

        Args:
            actions (RolloutAction): action for controlling ego and/or agents
            num_steps_to_take (int): how many env steps to take. Must be less or equal to length of the input actions
            render (bool): whether to render state and actions and return renderings
        """
        actions = actions.to_numpy()
        renderings = []
        if render:
            renderings.append(self.render(actions))
        self._step(step_actions=actions, num_steps_to_take=num_steps_to_take)
        return renderings
    
    def adjust_scene(self,adjust_plan):
        agent_data_by_scene = dict()
        for simscene in self._current_scenes:
            if simscene.scene.name in adjust_plan:
                adjust_plan_i = adjust_plan[simscene.scene.name]
                if adjust_plan_i["remove_existing_neighbors"]["flag"] and not adjust_plan_i["remove_existing_neighbors"]["executed"]:
                    simscene.agents = [agent for agent in simscene.agents if agent.name=="ego"]
                    adjust_plan_i["remove_existing_neighbors"]["executed"]=True
                agent_data=list()
                for agent_data_i in adjust_plan_i["agents"]:
                    if not agent_data_i["executed"]:
                        agent_data.append([agent_data_i["name"],
                                        np.array(agent_data_i["agent_state"]),
                                        agent_data_i["initial_timestep"],
                                        agent_types[agent_data_i["agent_type"]],
                                        np.array(agent_data_i["extent"]),
                                        ])
                        agent_data_i["executed"]=True
                agent_data_by_scene[simscene] = agent_data

        self.add_new_agents(agent_data_by_scene)
    

        
class EnvSplitUnifiedSimulation(EnvUnifiedSimulation):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            log_data=True,
            split_ego = False,
            renderer=None,
            parse_obs = True,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset, with the capability of spliting ego and agent observations

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
            split_ego (bool): if set to True, split ego out as the ego observation
            parse_obs (bool or dict): whether to parse the ego and agent observation or not
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = dataset.num_scenes()
        self._num_scenes = num_scenes
        self.split_ego = split_ego
        self.parse_obs = parse_obs

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes
        self._current_scene_indices = None

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self._cached_raw_observation = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._log_data = log_data
        self.logger = None


    def reset(self, scene_indices: List = None, start_frame_index = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        super(EnvSplitUnifiedSimulation,self).reset(scene_indices,start_frame_index)
        self._cached_raw_observation = None

    def render(self, actions_to_take):
        scene_ims = []
        ego_inds = [i for i, name in enumerate(self.current_agent_names) if name == "ego"]
        for i in ego_inds:
            im = render_state_trajdata(
                batch=self.get_observation(split_ego=False)["agents"],
                batch_idx=i,
                action=actions_to_take
            )
            scene_ims.append(im)
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        if self.split_ego:
            ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
            agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
            ego_action = Action(
                positions=ac[ego_idx, :, :2],
                yaws=ac[ego_idx, :, 2:3]
            )
            agent_action = Action(
                positions=ac[agent_idx, :, :2],
                yaws=ac[agent_idx, :, 2:3]
            )
            return RolloutAction(ego=ego_action,agents=agent_action)
        else:
            agents = Action(
                positions=ac[:, :, :2],
                yaws=ac[:, :, 2:3]
            )

            return RolloutAction(agents=agents)


    def get_observation(self,split_ego=None,return_raw=False):
        if split_ego is None:
            split_ego = self.split_ego
        if return_raw:
            if self._cached_raw_observation is not None:
                return self._cached_raw_observation
        else:
            if self._cached_observation is not None:
                if split_ego and "ego" in self._cached_observation:
                    return self._cached_observation
                elif not split_ego and "ego" not in self._cached_observation:
                    return self._cached_observation
                else:
                    self._cached_observation = None
                    self._cached_raw_observation = None

        self.timers.tic("get_obs")

        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))
        self._cached_raw_observation = raw_obs
        if return_raw:
            return raw_obs
        if split_ego:
            # obtain index of ego and agents    
            ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
            agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
            # raw_obs is the raw trajdata batch_element object without collation
            ego_obs_raw = [raw_obs[idx] for idx in ego_idx]
            # call the collate function to turn batch_element into trajdata batch object
            ego_obs_collated = self.dataset.get_collate_fn(return_dict=True)(ego_obs_raw)
            agent_obs_raw = [raw_obs[idx] for idx in agent_idx]
            # call the collate function to turn batch_element into trajdata batch object
            agent_obs_collated = self.dataset.get_collate_fn(return_dict=True)(agent_obs_raw)
            
            # parse_obs can be True (parse both ego and agent), or False (parse neither), or dictionary that determines whether to parse ego or agent observation
            if self.parse_obs==True:
                parse_plan = dict(ego=True,agent=True)
            elif self.parse_obs==False:
                parse_plan = dict(ego=False,agent=False)
            elif isinstance(self.parse_obs,dict):
                parse_plan = self.parse_obs
            if parse_plan["ego"]:
                ego_obs = parse_trajdata_batch(ego_obs_collated)
                ego_obs = TensorUtils.to_numpy(ego_obs,ignore_if_unspecified=True)
                ego_obs["scene_index"] = self.current_agent_scene_index[ego_idx]
                ego_obs["track_id"] = self.current_agent_track_id[ego_idx]
            else:
                # put collated observation into AgentBatch object from trajdata
                ego_obs = AgentBatch(**ego_obs_collated)
            if parse_plan["agent"]:
                agent_obs = parse_trajdata_batch(agent_obs_collated)
                agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                agent_obs["scene_index"] = self.current_agent_scene_index[agent_idx]
                agent_obs["track_id"] = self.current_agent_track_id[agent_idx]
            else:
                # put collated observation into AgentBatch object from trajdata
                agent_obs = AgentBatch(**agent_obs_collated)
            self._cached_observation = dict(ego=ego_obs,agents=agent_obs)
        else:
            # if ego is not splitted out, then either parse all observation or do not parse any observation.
            assert isinstance(self.parse_obs,bool)
            agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
            if self.parse_obs:
                agent_obs = parse_trajdata_batch(agent_obs)
                agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                agent_obs["scene_index"] = self.current_agent_scene_index
                agent_obs["track_id"] = self.current_agent_track_id
            else:
                agent_obs = AgentBatch(**agent_obs)
            self._cached_observation = dict(agents=agent_obs)

        self.timers.toc("get_obs")
        return self._cached_observation

    
    def combine_action(self,step_actions):
        # combine ego and agent actions
        ego_action = step_actions.ego.to_dict()
        agent_action = step_actions.agents.to_dict()
        ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
        agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
        min_length = min(ego_action["positions"].shape[1],agent_action["positions"].shape[1])
        combined_positions = np.zeros([len(self.current_agent_names),min_length,2])
        combined_yaws = np.zeros([len(self.current_agent_names),min_length,1])
        combined_positions[ego_idx] = ego_action["positions"][:,:min_length]
        combined_positions[agent_idx] = agent_action["positions"][:,:min_length]
        combined_yaws[ego_idx] = ego_action["yaws"][:,:min_length]
        combined_yaws[agent_idx] = agent_action["yaws"][:,:min_length]
        return RolloutAction(agents=Action(positions=combined_positions,yaws=combined_yaws),agents_info=step_actions.agents_info)


    def combine_obs(self,ego_obs,agent_obs):
        # combining ego and agent observation, not really used.
        ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
        agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
        bs = len(self.current_agent_names)
        combined_obs = dict()
        for k,v in ego_obs.items():
            if k in agent_obs and v is not None:
                combined_v = np.zeros([bs,*v.shape[1:]])
                combined_v[ego_idx]=ego_obs[k]
                combined_v[agent_idx]=agent_obs[k]
                combined_obs[k]=combined_v
        return combined_obs
    def get_observation_skimp(self,split_ego=True):
        self.timers.tic("obs_skimp")
        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False, get_map=False))
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id
        self.timers.toc("obs_skimp")
        if split_ego:
            ego_mask = [name=="ego" for name in self.current_agent_names]
            agents_mask = [name!="ego" for name in self.current_agent_names]
            ego_obs = TensorUtils.map_ndarray(obs, lambda x: x[ego_mask])
            agents_obs = TensorUtils.map_ndarray(obs, lambda x: x[agents_mask])
            
            return dict(ego=ego_obs,agents=agents_obs)
        else:
            return dict(agents=obs)
    def _add_per_step_metrics(self, obs):

        ego_mask = [name=="ego" for name in self.current_agent_names]
        agents_mask = [name!="ego" for name in self.current_agent_names]
        ego_obs = TensorUtils.map_ndarray(obs, lambda x: x[ego_mask])
        agents_obs = TensorUtils.map_ndarray(obs, lambda x: x[agents_mask])
        for k, v in self._metrics.items():
            if k.startswith("ego"):
                v.add_step(ego_obs, self._current_scene_indices)
            elif k.startswith("agents"):
                v.add_step(agents_obs, self._current_scene_indices)
            elif k.startswith("all"):
                v.add_step(obs, self._current_scene_indices)
            else:
                raise KeyError("Invalid metrics name {}".format(k))

    def _step(self, step_actions: RolloutAction, num_steps_to_take):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")
        self.timers.tic("_step")
        # to bypass all the ego split, collation and parsing, directly get raw obs
        raw_obs = self.get_observation(split_ego=False,return_raw=True)
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        # always parse when stepping
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id
        obs = {k:v for k,v in obs.items() if not isinstance(v,list)}

        # record metrics
        #TODO: fix the bugs in metrics when using diffstack
        self._add_per_step_metrics(obs)
        if step_actions.has_ego:
            combined_step_actions = self.combine_action(step_actions)
            action = combined_step_actions.agents.to_dict()
        else:
            action = step_actions.agents.to_dict()
        
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        for action_index in range(num_steps_to_take):
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                self._cached_raw_observation = None
                return
            # # log state and action
            obs_skimp = self.get_observation_skimp(split_ego=True)
            # self._add_per_step_metrics(obs_skimp["agents"])
            if self._log_data:
                if step_actions.has_ego:
                    ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
                    agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
                    action_t = TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(dict(positions=action_t["positions"][agent_idx],yaws=action_t["yaws"][agent_idx])),
                        agents_info=step_actions.agents_info,
                        ego = Action.from_dict(dict(positions=action_t["positions"][ego_idx],yaws=action_t["yaws"][ego_idx])),
                        ego_info=step_actions.ego_info,
                    )
                else:
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                        agents_info=step_actions.agents_info
                    )

                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    curr_yaw = obs["curr_agent_state"][idx, -1]
                    curr_pos = obs["curr_agent_state"][idx, :2]
                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.ones(4, dtype=obs["agent_fut"].dtype) * np.nan
                    if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                        next_state[-1] = curr_yaw + action["yaws"][idx, action_index, 0]
                    else:
                        pass
                    scene_action[agent.name] = StateArray.from_array(next_state, "x,y,z,h")
                    idx += 1
                scene.step(scene_action, return_obs=False)

        self._cached_observation = None
        self._cached_raw_observation = None
        self.timers.toc("_step")
        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take
        print(self.timers)
        

