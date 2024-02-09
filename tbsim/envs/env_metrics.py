import abc
import numpy as np
from typing import List, Dict, OrderedDict
import math

import torch
from l5kit.geometry import transform_points, angular_distance

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor, transform_yaw, detect_collision, CollisionType
from tbsim.utils.trajdata_utils import get_raster_pix2m
import tbsim.utils.metrics as Metrics
from collections import defaultdict
from tbsim.models.cnn_roi_encoder import rasterized_ROI_align
from torchvision.ops.roi_align import RoIAlign
import tbsim.utils.geometry_utils as GeoUtils
from pyemd import emd
import matplotlib.pyplot as plt
import pandas as pd
import tbsim.utils.geometry_utils as GeoUtils

class EnvMetrics(abc.ABC):
    def __init__(self):
        self._df = None
        self._scene_ts = defaultdict(lambda:0)
        
        self.reset()
        self.global_t = 0

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def add_step(self, state_info: Dict, all_scene_index: np.ndarray):
        pass

    @abc.abstractmethod
    def get_episode_metrics(self) -> Dict[str, np.ndarray]:
        pass

    def get_multi_episode_metrics(self) -> Dict[str, np.ndarray]:
        pass
    
    def multi_episode_reset(self):
        pass

    def __len__(self):
        return max(self._scene_ts.values()) if len(self._scene_ts)>0 else 0

    def update_global_t(self, global_t=None):
        '''
        Update any persistant state needed by guidance loss functions.
        - global_t : the current global timestep of rollout
        '''
        if global_t is not None:
            self.global_t = global_t


def step_aggregate_per_scene(agent_met, agent_scene_index, all_scene_index, agg_func=np.mean):
    """
    Aggregate per-step metrics for each scene.

    1. if there are more than one agent per scene, aggregate their metrics for each scene using @agg_func.
    2. if there are zero agent per scene, the returned mask should have 0 for that scene

    Args:
        agent_met (np.ndarray): metrics for all agents and scene [num_agents, ...]
        agent_scene_index (np.ndarray): scene index for each agent [num_agents]
        all_scene_index (list, np.ndarray): a list of scene indices [num_scene]
        agg_func: function to aggregate metrics value across all agents in a scene

    Returns:
        met_per_scene (np.ndarray): [num_scene]
        met_per_scene_mask (np.ndarray): [num_scene]
    """
    met_per_scene = split_agents_by_scene(agent_met, agent_scene_index, all_scene_index)
    met_agg_per_scene = []
    for met in met_per_scene:
        if len(met) > 0:
            met_agg_per_scene.append(agg_func(met))
        else:
            met_agg_per_scene.append(np.zeros_like(agent_met[0]))
    met_mask_per_scene = [len(met) > 0 for met in met_per_scene]
    return np.stack(met_agg_per_scene, axis=0), np.array(met_mask_per_scene)


def split_agents_by_scene(agent, agent_scene_index, all_scene_index):

    assert agent.shape[0] == agent_scene_index.shape[0]
    agent_split = []
    for si in all_scene_index:
        agent_split.append(agent[agent_scene_index == si])
    return agent_split


def agent_index_by_scene(agent_scene_index, all_scene_index):
    agent_split = []
    for si in all_scene_index:
        agent_split.append(np.where(agent_scene_index == si)[0])
    return agent_split


def masked_average_per_episode(met, met_mask):
    """
    Compute average metrics across timesteps given an availability mask
    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_met (np.ndarray): [num_scene]
    """
    assert met.shape == met_mask.shape
    return (met * met_mask).sum(axis=1) / (met_mask.sum(axis=1) + 1e-8)


def masked_sum_per_episode(met, met_mask):
    """
    Compute sum metrics across timesteps given an availability mask
    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_met (np.ndarray): [num_scene]
    """
    assert met.shape == met_mask.shape
    return (met * met_mask).sum(axis=1)


def masked_max_per_episode(met, met_mask):
    """

    Args:
        met (np.ndarray): measurements, [num_scene, num_steps]
        met_mask (np.ndarray): measurement masks [num_scene, num_steps]

    Returns:
        avg_max (np.ndarray): [num_scene]
    """
    assert met.shape == met_mask.shape
    return (met * met_mask).max(axis=1)


class OffRoadRate(EnvMetrics):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])


        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        
        # if True, use centroid to check offroad (same as BITS); if False, use bounding box to check offroad. 
        use_center = True
        if use_center:
            off_road_out = np.ones((centroid_raster.size(0))) * np.nan
            valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
            if torch.sum(valid_mask) == 0:
                return off_road_out
            off_road = Metrics.batch_detect_off_road(centroid_raster[valid_mask], drivable_region[valid_mask])  # [num_agents]
            off_road = TensorUtils.to_numpy(off_road)
            off_road_out[valid_mask.cpu().numpy()] = off_road
            
            # from matplotlib import pyplot as plt
            # i2pl = 3
            # cur_layer = drivable_region[i2pl].cpu().numpy() # h, w
            # plt.imshow(cur_layer)
            # plt.plot(centroid_raster[i2pl, 0:1].cpu().numpy(), centroid_raster[i2pl, 1:2].cpu().numpy(), 'ro')
            # plt.savefig('tmp.png')
            # plt.close()
            
            return off_road_out
        else:
            # TBD: this seems to not give correct results on the nusc dataset; will fix it later
            # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
            cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
            extent = obs["extent"][:,:2]
            # TODO: this is super hacky and assumes trajdata is being used.
            #       should really just transform corners in world frame then convert to raster.
            extent = get_raster_pix2m()*extent # convert to raster frame
            # off_road = Metrics.batch_detect_off_road_boxes(centroid_raster, cur_yaw, extent, drivable_region)
        
            off_road_out = np.ones((centroid_raster.size(0))) * np.nan

            valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
            if torch.sum(valid_mask) == 0:
                return off_road_out
            off_road = Metrics.batch_detect_off_road_boxes(centroid_raster[valid_mask],
                                                            cur_yaw[valid_mask],
                                                            extent[valid_mask],
                                                            drivable_region[valid_mask])
            off_road = TensorUtils.to_numpy(off_road)
            off_road_out[valid_mask.cpu().numpy()] = off_road
            # import matplotlib
            # from matplotlib import pyplot as plt
            # i2pl = 1
            # cur_layer = drivable_region[i2pl].cpu().numpy() # h, w
            # plt.imshow(cur_layer)
            # plt.plot(centroid_raster[i2pl, 0:1].cpu().numpy(), centroid_raster[i2pl, 1:2].cpu().numpy(), 'ro')
            # plt.savefig('tmp.png')
            # exit()
            # print(off_road)

            return off_road_out

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = dict(scene_index=state_info["scene_index"],
                       track_id=state_info["track_id"],
                       ts=ts,
                       met=met)
        step_df = pd.DataFrame(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):
        # df sums and means automatically ignore nan
        self._df.set_index(["scene_index","track_id","ts"])
        metric_by_agt = self._df.groupby(["scene_index","track_id"])["met"].sum()
        metric_nframe = metric_by_agt.groupby(["scene_index"]).mean().to_numpy()
        metric_rate = self._df.groupby(["scene_index"])["met"].mean().to_numpy()
        return {
            "rate" : metric_rate,
            "nframe" : metric_nframe
        }

class DiskOffRoadRate(EnvMetrics):
    """Compute the fraction of the time that the agent is in undrivable regions"""
    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        # CHANGE: set ignore_if_unspecified to deal with a string type on scene_index
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)
        drivable_region = batch_utils().get_drivable_region_map(obs["image"])
        # print(obs["centroid"])
        # print(obs["raster_from_world"])
        centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        # cur_yaw = transform_yaw(obs["yaw"], obs["agent_from_world"])[:,None] # agent frame is same as raster, just scaled
        cur_yaw = transform_yaw(obs["yaw"], obs["raster_from_world"])[:,None] # have to use raster tf mat because the raster may not be up to date with the agent (i.e. the raster may be from an older frame)
        extent = obs["extent"][:,:2]
        # TODO: this is super hacky and assumes trajdata is being used.
        #       should really just transform corners in world frame then convert to raster.
        extent = get_raster_pix2m()*extent # convert to raster frame

        # import matplotlib
        # from matplotlib import pyplot as plt
        # i2pl = 1
        # cur_layer = drivable_region[i2pl].cpu().numpy() # h, w
        # plt.imshow(cur_layer)
        # plt.plot(centroid_raster[i2pl, 0:1].cpu().numpy(), centroid_raster[i2pl, 1:2].cpu().numpy(), 'ro')
        # plt.show()
        # exit()
        # off_road = Metrics.batch_detect_off_road(centroid_raster, drivable_region)  # [num_agents]

        off_road_out = np.ones((centroid_raster.size(0))) * np.nan

        valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
        if torch.sum(valid_mask) == 0:
            return off_road_out
        # off_road = Metrics.batch_detect_off_road_boxes(centroid_raster[valid_mask],
        #                                                 cur_yaw[valid_mask],
        #                                                 extent[valid_mask],
        #                                                 drivable_region[valid_mask])
        off_road = Metrics.batch_detect_off_road_disk(centroid_raster[valid_mask],
                                                        extent[valid_mask],
                                                        drivable_region[valid_mask])
        # print(off_road)
        off_road = TensorUtils.to_numpy(off_road)
        off_road_out[valid_mask.cpu().numpy()] = off_road
        return off_road_out

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = dict(scene_index=state_info["scene_index"],
                       track_id=state_info["track_id"],
                       ts=ts,
                       met=met)
        step_df = pd.DataFrame(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):
        # df sums and means automatically ignore nan
        self._df.set_index(["scene_index","track_id","ts"])
        metric_by_agt = self._df.groupby(["scene_index","track_id"])["met"].sum()
        metric_nframe = metric_by_agt.groupby(["scene_index"]).mean().to_numpy()
        metric_rate = self._df.groupby(["scene_index"])["met"].mean().to_numpy()
        return {
            "rate" : metric_rate,
            "nframe" : metric_nframe
        }

class SemLayerRate(EnvMetrics):
    """
    Compute the fraction of the time that the agent centroid is on each layer of the semantic map
    Note: assumes same agent/scenes over time.
    """
    def reset(self):
        # self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])
        # self._scene_ts = defaultdict(lambda:0)
        self._per_step = []
        self._scene_masks = None

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        obs = TensorUtils.to_tensor(state_info, ignore_if_unspecified=True)

        sem_map = obs["image"]
        num_layers = sem_map.size(1)

        on_layers = []
        for li in range(num_layers+1):
            # print(obs["centroid"])
            # print(obs["raster_from_world"])
            centroid_raster = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]

            # import matplotlib
            # from matplotlib import pyplot as plt
            # i2pl = 1
            # cur_layer = drivable_region[i2pl].cpu().numpy() # h, w
            # plt.imshow(cur_layer)
            # plt.plot(centroid_raster[i2pl, 0:1].cpu().numpy(), centroid_raster[i2pl, 1:2].cpu().numpy(), 'ro')
            # plt.show()
            # exit()

            cur_on_layer_out = np.ones((centroid_raster.size(0))) * np.nan
            valid_mask = torch.sum(torch.isnan(centroid_raster), dim=-1) == 0
            if torch.sum(valid_mask) == 0:
                on_layers.append(cur_on_layer_out)
                continue

            if li == num_layers:
                # check if on no layers
                cur_layer_map = ~(torch.amax(sem_map[valid_mask], dim=-3).bool())
            else:
                cur_layer_map = sem_map[valid_mask][:,li].bool()
            cur_on_layer = 1.0 - Metrics.batch_detect_off_road(centroid_raster[valid_mask], cur_layer_map)  # [num_agents]
            # print(off_road)
            cur_on_layer = TensorUtils.to_numpy(cur_on_layer)
            cur_on_layer_out[valid_mask.cpu().numpy()] = cur_on_layer
            on_layers.append(cur_on_layer_out)

        on_layers = np.stack(on_layers, axis=1) # num_agents x num_layers

        return on_layers

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met)
        if self._scene_masks is None:
            agent_scene_index = state_info["scene_index"]
            self._scene_masks = agent_index_by_scene(agent_scene_index, all_scene_index)

    def get_episode_metrics(self):
        met_all_steps = np.stack(self._per_step, axis=1) # num_agents x T x num_layers
        met_per_scene = []
        for scene_mask in self._scene_masks:
            cur_scene_met = met_all_steps[scene_mask]
            sem_rate = np.nanmean(cur_scene_met, axis=1) # num_agents x num_layers
            sem_scene = np.mean(sem_rate, axis=0) # num_layers
            met_per_scene.append(sem_scene)
        # track each layer separately
        num_layers = len(met_per_scene[0]) - 1
        met_dict = dict()
        for li in range(num_layers+1):
            layer_met = [smet[li] for smet in met_per_scene]
            layer_str = "no_layer" if li == num_layers else "layer%02d" % (li)
            met_dict[layer_str] = np.array(layer_met)
        return met_dict

class CollisionRate(EnvMetrics):
    """Compute collision rate across all agents in a batch of data."""
    def __init__(self):
        super(CollisionRate, self).__init__()
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene collision rate and type"""
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)

        coll_rates = dict()
        for k in CollisionType:
            coll_rates[k] = np.zeros(num_agents)
        coll_rates["coll_any"] = np.zeros(num_agents)

        # for each scene, compute collision rate
        for i in range(num_scenes):
            num_agents_in_scene = pos_per_scene[i].shape[0]
            for j in range(num_agents_in_scene):
                other_agent_mask = np.arange(num_agents_in_scene) != j
                valid_mask = np.logical_not(np.sum(np.isnan(pos_per_scene[i]), axis=-1) > 0)
                if not valid_mask[j]:
                    continue
                other_agent_mask = np.logical_and(other_agent_mask, valid_mask)
                if np.sum(other_agent_mask) == 0:
                    continue
                coll = detect_collision(
                    ego_pos=pos_per_scene[i][j],
                    ego_yaw=yaw_per_scene[i][j],
                    ego_extent=extent_per_scene[i][j],
                    other_pos=pos_per_scene[i][other_agent_mask],
                    other_yaw=yaw_per_scene[i][other_agent_mask],
                    other_extent=extent_per_scene[i][other_agent_mask]
                )
                
                if coll is not None:
                    coll_rates[coll[0]][agent_index_per_scene[i][j]] = 1.
                    coll_rates["coll_any"][agent_index_per_scene[i][j]] = 1.

        # compute per-scene collision counts (for visualization purposes)
        coll_counts = dict()
        for k in coll_rates:
            coll_counts[k], _ = step_aggregate_per_scene(
                coll_rates[k],
                agent_scene_index,
                all_scene_index,
                agg_func=np.sum
            )

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = []
        for k in met_all:
            if k=="coll_any":
                type=-1
            else:
                type=k
            step_df_k = dict(scene_index=state_info["scene_index"],
                        track_id=state_info["track_id"],
                        ts=ts,
                        type=type,
                        met=met_all[k])
            step_df.append(pd.DataFrame(step_df_k))
        step_df = pd.concat(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):

        self._df.set_index(["scene_index","track_id","type","ts"])
        coll_whole_horizon = self._df.groupby(["scene_index","track_id","type"])["met"].max()
        met_all = dict()
        
        for k in CollisionType:
            coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([k],level=2)]
            met_all[str(k)] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([-1],level=2)]
        met_all["coll_any"] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        return met_all

class DiskCollisionRate(EnvMetrics):
    """Compute collision rate across all agents in a batch of data."""
    def __init__(self):
        super(DiskCollisionRate, self).__init__()
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    def reset(self):
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', 'type', "met"])
        self._scene_ts = defaultdict(lambda:0)

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        """Compute per-agent and per-scene collision rate and type"""
        agent_scene_index = state_info["scene_index"]
        pos_per_scene = split_agents_by_scene(state_info["centroid"], agent_scene_index, all_scene_index)
        yaw_per_scene = split_agents_by_scene(state_info["yaw"], agent_scene_index, all_scene_index)
        extent_per_scene = split_agents_by_scene(state_info["extent"][..., :2], agent_scene_index, all_scene_index)
        agent_index_per_scene = agent_index_by_scene(agent_scene_index, all_scene_index)

        num_scenes = len(all_scene_index)
        num_agents = len(agent_scene_index)

        coll_rates = dict()
        for k in CollisionType:
            coll_rates[k] = np.zeros(num_agents)
        coll_rates["coll_any"] = np.zeros(num_agents)

        # for each scene, compute collision rate
        for i in range(num_scenes):
            num_agents_in_scene = pos_per_scene[i].shape[0]
            rad_cur_scene = np.nanmin(extent_per_scene[i], axis=-1) / 2.0
            for j in range(num_agents_in_scene):
                other_agent_mask = np.arange(num_agents_in_scene) != j
                valid_mask = np.logical_not(np.sum(np.isnan(pos_per_scene[i]), axis=-1) > 0)
                if not valid_mask[j]:
                    continue
                other_agent_mask = np.logical_and(other_agent_mask, valid_mask)
                if np.sum(other_agent_mask) == 0:
                    continue

                neighbor_dist = np.linalg.norm(pos_per_scene[i][j:j+1] - pos_per_scene[i][other_agent_mask], axis=-1)
                min_allowed_dist = rad_cur_scene[j] + rad_cur_scene[other_agent_mask]
                coll = np.sum(neighbor_dist < min_allowed_dist) > 0
                if coll:
                    coll_rates["coll_any"][agent_index_per_scene[i][j]] = 1.

        # compute per-scene collision counts (for visualization purposes)
        coll_counts = dict()
        for k in coll_rates:
            coll_counts[k], _ = step_aggregate_per_scene(
                coll_rates[k],
                agent_scene_index,
                all_scene_index,
                agg_func=np.sum
            )

        return coll_rates, coll_counts

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        
        met_all, _ = self.compute_per_step(state_info, all_scene_index)
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = []
        for k in met_all:
            if k=="coll_any":
                type=-1
            else:
                type=k
            step_df_k = dict(scene_index=state_info["scene_index"],
                        track_id=state_info["track_id"],
                        ts=ts,
                        type=type,
                        met=met_all[k])
            step_df.append(pd.DataFrame(step_df_k))
        step_df = pd.concat(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1

    def get_episode_metrics(self):

        self._df.set_index(["scene_index","track_id","type","ts"])
        coll_whole_horizon = self._df.groupby(["scene_index","track_id","type"])["met"].max()
        met_all = dict()
        
        for k in CollisionType:
            coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([k],level=2)]
            met_all[str(k)] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        coll_data = coll_whole_horizon[coll_whole_horizon.index.isin([-1],level=2)]
        met_all["coll_any"] = coll_data.groupby(["scene_index"]).mean().to_numpy()
        return met_all

class CriticalFailure(EnvMetrics):
    """Metrics that report failures caused by either collision or offroad"""
    def __init__(self, num_collision_frames=1, num_offroad_frames=3):
        super(CriticalFailure, self).__init__()
        self._df = pd.DataFrame(columns=["scene_index","track_id","ts","offroad","collision"])
        self._scene_ts = defaultdict(lambda:0)

    def reset(self):
        self._df = pd.DataFrame(columns=["scene_index","track_id","ts","offroad","collision"])
        self._scene_ts = defaultdict(lambda:0)


    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met_all = dict(
            offroad=OffRoadRate.compute_per_step(state_info, all_scene_index),
            collision=CollisionRate.compute_per_step(state_info, all_scene_index)[0]["coll_any"]
        )
        ts = np.array([self._scene_ts[sid] for sid in state_info["scene_index"]])
        step_df = dict(scene_index=state_info["scene_index"],
                       track_id=state_info["track_id"],
                       ts=ts,
                       offroad=met_all["offroad"],
                       collision = met_all["collision"])
        step_df = pd.DataFrame(step_df)
        self._df = pd.concat((self._df,step_df))
        for sid in np.unique(state_info["scene_index"]):
            self._scene_ts[sid]+=1
    
    def get_per_agent_metrics(self):
        coll_fail_cases = self._df.groupby(["scene_index","track_id"])["collision"].any()
        offroad_fail_cases = self._df.groupby(["scene_index","track_id"])["offroad"].any()
        any_fail_cases = coll_fail_cases|offroad_fail_cases
        return dict(offroad=offroad_fail_cases,collision=coll_fail_cases,any=any_fail_cases)


    def get_episode_metrics(self) -> Dict[str, np.ndarray]:
        num_steps = len(self)
        grid_points = np.arange(5,num_steps,5)

        coll_fail_cases = self._df.groupby(["scene_index","track_id"])["collision"].any()
        coll_by_scene = coll_fail_cases.groupby(["scene_index"])
        coll_fail_rate = (coll_by_scene.sum()/coll_by_scene.count()).to_numpy()
        offroad_fail_cases = self._df.groupby(["scene_index","track_id"])["offroad"].any()
        offroad_by_scene = offroad_fail_cases.groupby(["scene_index"])
        offroad_fail_rate = (offroad_by_scene.sum()/offroad_by_scene.count()).to_numpy()
        any_fail_cases = coll_fail_cases | offroad_fail_cases
        any_fail_by_scene = any_fail_cases.groupby(["scene_index"])
        any_fail_rate = (any_fail_by_scene.sum()/any_fail_by_scene.count()).to_numpy()

        met = dict(failure_offroad=offroad_fail_rate,failure_collision=coll_fail_rate,failure_any=any_fail_rate)
        # for t in grid_points:
        #     df_sel = self._df.loc[self._df["ts"]<t]
        #     coll_fail_cases = df_sel.groupby(["scene_index","track_id"])["collision"].any()
        #     coll_by_scene = coll_fail_cases.groupby(["scene_index"])
        #     coll_fail_rate = (coll_by_scene.sum()/coll_by_scene.count()).to_numpy()
        #     offroad_fail_cases = df_sel.groupby(["scene_index","track_id"])["offroad"].any()
        #     offroad_by_scene = offroad_fail_cases.groupby(["scene_index"])
        #     offroad_fail_rate = (offroad_by_scene.sum()/offroad_by_scene.count()).to_numpy()
        #     any_fail_cases = coll_fail_cases | offroad_fail_cases
        #     any_fail_by_scene = any_fail_cases.groupby(["scene_index"])
        #     any_fail_rate = (any_fail_by_scene.sum()/any_fail_by_scene.count()).to_numpy()
        #     met["failure_offroad@{}".format(t)]=offroad_fail_rate
        #     met["failure_collision@{}".format(t)]=coll_fail_rate
        #     met["failure_any@{}".format(t)]=any_fail_rate
        return met


class LearnedMetric(EnvMetrics):
    def __init__(self, metric_algo, perturbations=None):
        super(LearnedMetric, self).__init__()
        self.metric_algo = metric_algo
        self.traj_len = metric_algo.algo_config.future_num_frames
        self.state_buffer = []
        self.perturbations = dict() if perturbations is None else perturbations
        self.total_steps = 0
        self._df = pd.DataFrame(columns = ['scene_index', 'track_id', 'ts', "met"])

    def reset(self):
        self.state_buffer = []
        self._per_step_mask = []
        self.total_steps = 0

    def __len__(self):
        return self.total_steps

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        state_info = dict(state_info)
        state_info["image"] = (state_info["image"] * 255.).astype(np.uint8)
        self.state_buffer.append(state_info)
        while len(self.state_buffer) > self.traj_len + 1:
            self.state_buffer.pop(0)
        if len(self.state_buffer) == self.traj_len + 1:
            step_metrics, agent_selected = self.compute_per_step(self.state_buffer, all_scene_index)
            self._per_step.append(step_metrics)

        self.total_steps += 1

    def compute_per_step(self, state_buffer, all_scene_index):
        assert len(state_buffer) == self.traj_len + 1

        # assemble score function input
        appearance_idx = obtain_active_agent_index(state_buffer)
        agent_selected = np.where((appearance_idx>=0).all(axis=1))[0]
        state = dict(state_buffer[0])  # avoid changing the original state_dict
        for k,v in state.items():
            state[k]=v[agent_selected]
        state["image"] = (state["image"] / 255.).astype(np.float32)
        agent_from_world = state["agent_from_world"]
        yaw_current = state["yaw"]

        # transform traversed trajectories into the ego frame of a given state
        traj_inds = range(1, self.traj_len + 1)
        traj_pos = [state_buffer[traj_i]["centroid"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_yaw = [state_buffer[traj_i]["yaw"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_pos = np.stack(traj_pos, axis=1)  # [B, T, 2]

        traj_yaw = np.stack(traj_yaw, axis=1)  # [B, T]
        assert traj_pos.shape[0] == traj_yaw.shape[0]

        agent_traj_pos = transform_points(points=traj_pos, transf_matrix=agent_from_world)
        agent_traj_yaw = angular_distance(traj_yaw, yaw_current[:, None])

        traj_to_eval = dict()
        traj_to_eval["target_positions"] = agent_traj_pos
        traj_to_eval["target_yaws"] = agent_traj_yaw[:, :, None]

        state_torch = TensorUtils.to_torch(state, self.metric_algo.device)
        metrics = dict()

        # evaluate score of the ground truth state
        m = self.metric_algo.get_metrics(state_torch)
        for mk in m:
            metrics["gt_{}".format(mk)] = m[mk]

        with torch.no_grad():
            traj_torch = TensorUtils.to_torch(traj_to_eval, self.metric_algo.device)
            state_to_eval = dict(state_torch)
            state_to_eval.update(traj_torch)
            state_to_eval = TensorUtils.recursive_dict_list_tuple_apply(
                state_to_eval,
                {
                    torch.Tensor: lambda x:x.type(torch.float),
                    type(None): lambda x: x,
                },
            )
            m = self.metric_algo.get_metrics(state_to_eval)
            for mk in m:
                metrics["comp_{}".format(mk)] = (metrics["gt_{}".format(mk)] < m[mk]).float()
            metrics.update(m)
        for k, v in self.perturbations.items():
            traj_perturbed = TensorUtils.to_torch(v.perturb(traj_to_eval), self.metric_algo.device)
            state_perturbed = dict(state_torch)
            state_perturbed.update(traj_perturbed)
            m = self.metric_algo.get_metrics(state_perturbed)
            for mk in m:
                metrics["{}_{}".format(k, mk)] = m[mk]

        metrics= TensorUtils.to_numpy(metrics)

        step_metrics = dict()
        for k in metrics:
            met, met_mask = step_aggregate_per_scene(metrics[k], state["scene_index"], all_scene_index)
            assert np.all(met_mask > 0)  # since we will always use it for all agents
            step_metrics[k] = met

        return step_metrics, agent_selected

    def get_episode_metrics(self):
        ep_metrics = dict()

        for step_metrics in self._per_step:
            for k in step_metrics:
                if k not in ep_metrics:
                    ep_metrics[k] = []
                ep_metrics[k].append(step_metrics[k])

        ep_metrics_agg = dict()
        for k in ep_metrics:
            met = np.stack(ep_metrics[k], axis=1)  # [num_scene, T, ...]
            ep_metrics_agg[k] = np.mean(met, axis=1)
            for met_horizon in [10, 50, 100, 150]:
                if met.shape[1] >= met_horizon:
                    ep_metrics_agg[k + "@{}".format(met_horizon)] = np.mean(met[:, :met_horizon], axis=1)
        return ep_metrics_agg


class LearnedCVAENLL(EnvMetrics):
    def __init__(self, metric_algo, perturbations=None):
        super(LearnedCVAENLL, self).__init__()
        self.metric_algo = metric_algo
        self.traj_len = metric_algo.algo_config.future_num_frames
        self.state_buffer = []
        self.perturbations = dict() if perturbations is None else perturbations
        self.total_steps = 0

    def reset(self):
        self.state_buffer = []
        self._per_step = []
        self._per_step_mask = []
        self.total_steps = 0

    def __len__(self):
        return self.total_steps

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        state_info = dict(state_info)
        state_info["image"] = (state_info["image"] * 255.).astype(np.uint8)
        self.state_buffer.append(state_info)
        self.total_steps += 1

    def compute_metric(self, state_buffer, all_scene_index):
        assert len(state_buffer) == self.traj_len + 1
        appearance_idx = obtain_active_agent_index(state_buffer)
        agent_selected = np.where((appearance_idx>=0).all(axis=1))[0]
        # assemble score function input
        state = dict(state_buffer[0])  # avoid changing the original state_dict
        for k,v in state.items():
            state[k]=v[agent_selected]
        state["image"] = (state["image"] / 255.).astype(np.float32)
        agent_from_world = state["agent_from_world"]
        yaw_current = state["yaw"]

        # transform traversed trajectories into the ego frame of a given state
        traj_inds = range(1, self.traj_len + 1)
        

        traj_pos = [state_buffer[traj_i]["centroid"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_yaw = [state_buffer[traj_i]["yaw"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_pos = np.stack(traj_pos, axis=1)  # [B, T, 2]

        traj_yaw = np.stack(traj_yaw, axis=1)  # [B, T]
        assert traj_pos.shape[0] == traj_yaw.shape[0]
        
        agent_traj_pos = transform_points(points=traj_pos, transf_matrix=agent_from_world)
        agent_traj_yaw = angular_distance(traj_yaw, yaw_current[:, None])

        traj_to_eval = dict()
        traj_to_eval["target_positions"] = agent_traj_pos
        traj_to_eval["target_yaws"] = agent_traj_yaw[:, :, None]

        state_torch = TensorUtils.to_torch(state, self.metric_algo.device)
        metrics = dict()

        # evaluate score of the ground truth state
        # m = self.metric_algo.get_metrics(state_torch)
        # for mk in m:
        #     metrics["gt_{}".format(mk)] = m[mk]
        traj_torch = TensorUtils.to_torch(traj_to_eval, self.metric_algo.device)
        m = self.metric_algo.get_metrics(state_torch,traj_torch)
        for mk in m:
            metrics[mk] = m[mk]

        for k, v in self.perturbations.items():
            
            traj_perturbed = TensorUtils.to_torch(v.perturb(traj_to_eval), self.metric_algo.device)
            state_perturbed = dict(state_torch)
            state_perturbed.update(traj_perturbed)
            m = self.metric_algo.get_metrics(state_perturbed)
            for mk in m:
                metrics["{}_{}".format(k, mk)] = m[mk]

        metrics= TensorUtils.to_numpy(metrics)
        step_metrics = dict()
        for k in metrics:
            met, met_mask = step_aggregate_per_scene(metrics[k], state["scene_index"], all_scene_index)
            assert np.all(met_mask > 0)  # since we will always use it for all agents
            step_metrics[k] = met
        
        return step_metrics

    def get_episode_metrics(self):
        assert len(self.state_buffer) >= self.traj_len+1
        all_scene_index = np.unique(self.state_buffer[-self.traj_len-1]["scene_index"])
        ep_metrics = self.compute_metric(self.state_buffer[-self.traj_len-1:], all_scene_index)
        return ep_metrics


class LearnedCVAENLLRolling(LearnedCVAENLL):
    def __init__(self, metric_algo, rolling_horizon=None, perturbations=None):
        super(LearnedCVAENLLRolling, self).__init__(metric_algo,perturbations)
        self.rolling_horizon = rolling_horizon

    def reset(self):
        self.state_buffer = []
        self._per_step = []
        self._per_step_mask = []
        self.total_steps = 0

    def __len__(self):
        return self.total_steps

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        state_info = dict(state_info)
        state_info["image"] = (state_info["image"] * 255.).astype(np.uint8)
        self.state_buffer.append(state_info)
        self.total_steps += 1
        step_metrics = self.compute_per_step(all_scene_index)
        if step_metrics is not None:
            self._per_step.append(step_metrics)

    def compute_per_step(self, all_scene_index):
        if len(self.state_buffer)<self.traj_len + 1:
            return None
        # assert len(self.state_buffer) == self.traj_len + 1
        self.state_buffer = self.state_buffer[-self.traj_len-1:]
        appearance_idx = obtain_active_agent_index(self.state_buffer)
        agent_selected = np.where((appearance_idx>=0).all(axis=1))[0]
        # assemble score function input
        state = dict(self.state_buffer[0])  # avoid changing the original state_dict
        for k,v in state.items():
            if isinstance(v,np.ndarray):
                state[k]=v[agent_selected]
        state["image"] = (state["image"] / 255.).astype(np.float32)
        agent_from_world = state["agent_from_world"]
        yaw_current = state["yaw"]

        # transform traversed trajectories into the ego frame of a given state
        traj_inds = range(1, self.traj_len + 1)
        

        traj_pos = [self.state_buffer[traj_i]["centroid"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_yaw = [self.state_buffer[traj_i]["yaw"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_pos = np.stack(traj_pos, axis=1)  # [B, T, 2]

        traj_yaw = np.stack(traj_yaw, axis=1)  # [B, T]
        assert traj_pos.shape[0] == traj_yaw.shape[0]
        
        agent_traj_pos = transform_points(points=traj_pos, transf_matrix=agent_from_world)
        agent_traj_yaw = angular_distance(traj_yaw, yaw_current[:, None])

        traj_to_eval = dict()
        traj_to_eval["target_positions"] = agent_traj_pos
        traj_to_eval["target_yaws"] = agent_traj_yaw[:, :, None]

        state_torch = TensorUtils.to_torch(state, self.metric_algo.device)
        metrics = dict()

        traj_torch = TensorUtils.to_torch(traj_to_eval, self.metric_algo.device)

        if isinstance(self.rolling_horizon,int):
            m = self.metric_algo.get_metrics(state_torch,traj_torch,horizon=self.rolling_horizon)
            for mk in m:
                metrics[mk] = m[mk]
        elif isinstance(self.rolling_horizon,list):
            for horizon in self.rolling_horizon:
                m = self.metric_algo.get_metrics(state_torch,traj_torch,horizon=horizon)
                for mk in m:
                    metrics["{}_horizon_{}".format(mk,horizon)] = m[mk]
        
        for k, v in self.perturbations.items():
            traj_perturbed = TensorUtils.to_torch(v.perturb(traj_to_eval), self.metric_algo.device)
            for kk,vv in traj_perturbed.items():
                traj_perturbed[kk]=vv.type(torch.float32)
            if isinstance(self.rolling_horizon,int):
                rolling_horizon = self.rolling_horizon
            elif isinstance(self.rolling_horizon,list):
                rolling_horizon = self.rolling_horizon[1]
            m = self.metric_algo.get_metrics(state_torch,traj_perturbed,horizon=rolling_horizon)
            for mk in m:
                metrics["{}_{}".format(k, mk)] = m[mk]

        metrics= TensorUtils.to_numpy(metrics)
        step_metrics = dict()
        for k in metrics:
            met, met_mask = step_aggregate_per_scene(metrics[k], state["scene_index"], all_scene_index)
            assert np.all(met_mask > 0)  # since we will always use it for all agents
            step_metrics[k] = met
        self.state_buffer.pop(0)
        return step_metrics

    def get_episode_metrics(self):
        scene_met = dict()
        for k in self._per_step[0]:
            scene_met_k = [step_met[k] for step_met in self._per_step]
            scene_met_k = np.stack(scene_met_k,axis=0)
            scene_met_k = scene_met_k.mean(0)
            scene_met[k] = scene_met_k   
        return scene_met

def obtain_active_agent_index(state_buffer):
    agents_indices = dict()
    appearance_idx = -np.ones([state_buffer[0]["scene_index"].shape[0],len(state_buffer)])
    appearance_idx[:,0]=np.arange(appearance_idx.shape[0])
    for i in range(state_buffer[0]["scene_index"].shape[0]):
        agents_indices[(state_buffer[0]["scene_index"][i],state_buffer[0]["track_id"][i])]=i

    for t in range(1,len(state_buffer)):
        for i in range(state_buffer[t]["scene_index"].shape[0]):
            agent_idx = (state_buffer[t]["scene_index"][i],state_buffer[t]["track_id"][i])
            if agent_idx in agents_indices:
                appearance_idx[agents_indices[agent_idx],t] = i

    return appearance_idx.astype(np.int32)


class OccupancyGrid():
    def __init__(self,gridinfo,sigma=1.0):
        """Estimate occupancy with kernel density estimation under a Gaussian RBF kernel

        Args:
            gridinfo (dict): grid offset, grid step size
            sigma (float): std for the RBF kernel
        """
        self.gridinfo = gridinfo
        self.sigma = sigma
        self.occupancy_grid = defaultdict(lambda: 0)
        self.lane_flag = defaultdict(lambda: 0)
        self.agent_ids = defaultdict(lambda: set())

    def get_neighboring_grid_points(self,coords,radius):
        
        x0,y0=self.gridinfo["offset"]
        xs,ys=self.gridinfo["step"]
        bs = coords.shape[0]
        Nx = int(np.ceil(radius/xs))+1
        Ny = int(np.ceil(radius/xs))+1
        grid = np.concatenate((np.tile(np.arange(-Nx,Nx+1)[:,np.newaxis],(1,2*Ny+1))[...,np.newaxis],
                              np.tile(np.arange(-Ny,Ny+1)[np.newaxis,:],(2*Nx+1,1))[...,np.newaxis]),-1)
        grid = np.tile(grid[np.newaxis,...],(bs,1,1,1))
        xi,yi = (coords[:,0:1]-x0)/xs, (coords[:,1:]-y0)/ys
        # hacky conversion of nan to 0
        xi,yi = np.where(np.isnan(xi), 0, xi), np.where(np.isnan(yi), 0, yi)
        xi,yi = np.round(xi).astype(np.int32), np.round(yi).astype(np.int32)
        XYi = (grid+np.concatenate((xi,yi),-1).reshape(bs,1,1,2))
        grid_points = self.gridinfo["step"].reshape(1,1,1,2)*XYi+self.gridinfo["offset"].reshape(1,1,1,2)

        kernel_value= np.exp(-np.linalg.norm(coords[:,np.newaxis,np.newaxis]-grid_points,axis=-1)**2/2/self.sigma)
        return grid_points.reshape(bs,-1,2),XYi.reshape(bs,-1,2),kernel_value.reshape(bs,-1)

    def reset(self):
        self.occupancy_grid.clear()
        self.lane_flag.clear()
        self.agent_ids.clear()
    
    def obtain_lane_flag(self,grid_points,raster_from_world,lane_map):
        raster_points = GeoUtils.batch_nd_transform_points_np(grid_points,raster_from_world)
        # hacky conversion of nan to 0
        raster_points = np.where(np.isnan(raster_points), 0, raster_points)
        raster_points = raster_points.astype(np.int32)
        raster_points[...,0] = raster_points[...,0].clip(0,lane_map.shape[-2]-1)
        raster_points[...,1] = raster_points[...,1].clip(0,lane_map.shape[-1]-1)
        lane_flag = list()
        
        for k in range(raster_points.shape[0]):
            lane_flag.append(np.array([lane_map[k,y,x] for x,y in zip(raster_points[k,:,0],raster_points[k,:,1])]))
        lane_flag = np.stack(lane_flag,0)
        # clear_flag = (raster_points[:,0]>=0) & (raster_points[:,0]<drivable_area_map.shape[0])& (raster_points[:,1]>=0) & (raster_points[:,1]<drivable_area_map.shape[1])
        return lane_flag

    def update(self, coords, raster_from_world, lane_map, agent_ids, episode_index, threshold=0.1,weight=1):
        assert threshold<1.0
        radius = np.sqrt(-2*self.sigma*np.log(threshold))
        grid_points,XYi,kernel_value = self.get_neighboring_grid_points(coords,radius)
        lane_flag = self.obtain_lane_flag(grid_points,raster_from_world,lane_map)
        agent_ids = np.repeat(agent_ids[:, None], axis=1, repeats=grid_points.shape[1])

        XYi_flatten = XYi.reshape(-1,2)
        lane_flag_flatten = lane_flag.flatten()
        kernel_value_flatten = kernel_value.flatten()
        agent_ids = agent_ids.flatten()
        for i in range(XYi_flatten.shape[0]):
            self.occupancy_grid[(XYi_flatten[i,0],XYi_flatten[i,1])] += weight*kernel_value_flatten[i]
            self.lane_flag[(XYi_flatten[i,0],XYi_flatten[i,1])] = lane_flag_flatten[i]
            self.agent_ids[(XYi_flatten[i,0],XYi_flatten[i,1])].add((episode_index, agent_ids[i]))
    
    def visualize(self):
        fig, ax = plt.subplots(figsize=(20, 20))
        for k in self.occupancy_grid.keys():
            color = "gx" if self.lane_flag[k] else "ro"
            xyi = np.array(k)
            xy = xyi*self.gridinfo["step"]+self.gridinfo["offset"]
            ax.plot(xy[1],xy[0],color)
        plt.show()


class Occupancymet(EnvMetrics):
    def __init__(self, gridinfo, sigma=1.0):
        self.og = dict()
        super(Occupancymet, self).__init__()
        self.gridinfo = gridinfo
        self.sigma=sigma
        self._per_step = []
        self._per_step_mask = []

    """Compute occupancy grid on the map for agents."""
    def reset(self):
        self.og.clear()

    def add_step(self, state_info: dict, all_scene_index: np.ndarray, episode_index: int):
        self._per_step.append(0)
        self._per_step_mask.append(1)
        drivable_area = batch_utils().get_drivable_region_map(state_info["image"])
        coords = state_info["centroid"][:, :2]
        for scene_idx in all_scene_index:
            indices = np.where(state_info["scene_index"]==scene_idx)[0]
            if scene_idx not in self.og:
                self.og[scene_idx] = OccupancyGrid(self.gridinfo,self.sigma)

            self.og[scene_idx].update(
                coords=coords[indices],
                raster_from_world=state_info["raster_from_world"][indices],
                lane_map=drivable_area[indices],
                agent_ids=state_info["track_id"][indices],
                episode_index=episode_index,
                threshold=0.1,
                weight=1,
            )

    def get_episode_metrics(self):
        pass


class OccupancyCoverage(Occupancymet):
    def __init__(self, gridinfo, sigma=1.0, threshold=1e-2):
        self.failure_metric = [CriticalFailure(num_offroad_frames=2)]
        self.episode_index = 0
        self.threshold = threshold
        self._episode_started = False
        super(OccupancyCoverage, self).__init__(gridinfo, sigma)

    def reset(self):
        if self._episode_started:
            self.failure_metric.append(CriticalFailure(num_offroad_frames=2))
            self.episode_index += 1
            self._episode_started = False

    def multi_episode_reset(self):
        self.failure_metric = [CriticalFailure(num_offroad_frames=2)]
        self.episode_index = 0
        self._episode_started = False
        self.og.clear()

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        super(OccupancyCoverage, self).add_step(state_info, all_scene_index, self.episode_index)
        self._episode_started = True
        self.failure_metric[-1].add_step(state_info, all_scene_index)

    def summarize_grid(self):
        failed_agent_ids = []
        failed_scene_index = []
        assert self.episode_index + 1 == len(self.failure_metric)
        for fm in self.failure_metric:
            per_agent_failure = fm.get_per_agent_metrics()["any"]
            fail_index = per_agent_failure[per_agent_failure==True].index
            failed_scene_index.append(fail_index.get_level_values("scene_index").to_numpy())
            failed_agent_ids.append(fail_index.get_level_values("track_id").to_numpy())

        coverage_num = OrderedDict(total=[], onroad=[], success=[])
        for scene_idx, og in self.og.items():
            data = np.array(list(og.occupancy_grid.values()))
            lane = np.array(list(og.lane_flag.values())).astype(np.float32)
            success_mask = np.zeros_like(lane).astype(bool)
            for i, ep_aid in enumerate(og.agent_ids.values()):
                # if any of the successful agent in any episode covers a grid, count it as a successful coverage
                # conversely, if all the agents that cover the grid ended up failing, do not count the coverage.
                for (epi, aid) in ep_aid:
                    failed_agent_ids_in_scene = failed_agent_ids[epi][failed_scene_index[epi] == scene_idx]
                    success_mask[i] = success_mask[i] or (aid not in failed_agent_ids_in_scene)
            data_onroad = data * lane
            data_success = data * lane * success_mask.astype(np.float32)
            coverage_num["onroad"].append((data_onroad > self.threshold).sum())
            coverage_num["success"].append((data_success > self.threshold).sum())
            coverage_num["total"].append((data > self.threshold).sum())

        return {k: np.array(v) for k, v in coverage_num.items()}

    def get_multi_episode_metrics(self):
        return self.summarize_grid()

    def get_episode_metrics(self):
        # CHANGE: modify this to support coverage estimation when running one episode
        return self.summarize_grid()


class OccupancyDiversity(Occupancymet):
    def __init__(self, gridinfo, sigma=1.0):
        super(OccupancyDiversity, self).__init__(gridinfo, sigma)
        self.episode_index = 0

    def reset(self):
        self._per_step = []
        self._per_step_mask = []

    def multi_episode_reset(self):
        self.episode_index = 0
        self.og.clear()

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        self._per_step.append(0)
        self._per_step_mask.append(1)
        drivable_area = batch_utils().get_drivable_region_map(state_info["image"])
        coords = state_info["centroid"][:, :2]
        for scene_idx in all_scene_index:
            indices = np.where(state_info["scene_index"]==scene_idx)[0]
            if scene_idx not in self.og:
                self.og[scene_idx] = [OccupancyGrid(self.gridinfo,self.sigma)]
            if len(self.og[scene_idx])==self.episode_index:
                self.og[scene_idx].append(OccupancyGrid(self.gridinfo,self.sigma))
            
            assert len(self.og[scene_idx])==self.episode_index+1
            self.og[scene_idx][self.episode_index].update(
                coords=coords[indices],
                raster_from_world=state_info["raster_from_world"][indices],
                lane_map=drivable_area[indices],
                agent_ids=state_info["track_id"],
                episode_index=self.episode_index,
                threshold=0.1,
                weight=1
            )

    def get_multi_episode_metrics(self):
        result = []
        for scene_index in self.og:
            keys_union = set()
            distr = list()
            for og in self.og[scene_index]:
                keys_union = keys_union.union(set(og.occupancy_grid.keys()))
                
            coords = np.array(list(keys_union))*self.gridinfo["step"]+self.gridinfo["offset"]
            coords = np.tile(coords,(coords.shape[0],1,1))
            distance_matrix = np.linalg.norm(coords-coords.transpose(1,0,2),axis=2)
            wasser_dis = np.array([])
            for og in self.og[scene_index]:
                distr_i = np.array([og.occupancy_grid[k] for k in keys_union])
                lane_flag = np.array([og.lane_flag[k] for k in keys_union])
                distr_i = distr_i*lane_flag
                distr_i = distr_i/distr_i.sum()
                for distr_j in distr:
                    wasser_dis = np.append(wasser_dis,emd(distr_i, distr_j, distance_matrix))
                distr.append(distr_i)
            result.append(wasser_dis.mean())
            print("Wasserstein metric:",wasser_dis)
        return np.array(result)

    def get_episode_metrics(self):
        self.episode_index+=1
        return dict()


class Occupancy_likelihood(EnvMetrics):
    def __init__(self, metric_algo, perturbations=None):
        super(Occupancy_likelihood, self).__init__()
        self.metric_algo = metric_algo
        self.traj_len = metric_algo.algo_config.future_num_frames
        self.state_buffer = []
        self.perturbations = dict() if perturbations is None else perturbations
        self.total_steps = 0

    def reset(self):
        self.state_buffer = []
        self._per_step = []
        self._per_step_mask = []
        self.total_steps = 0

    def __len__(self):
        return self.total_steps

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        state_info = dict(state_info)
        state_info["image"] = (state_info["image"] * 255.).astype(np.uint8)
        self.state_buffer.append(state_info)
        self.total_steps += 1

    def compute_metric(self, state_buffer, all_scene_index):
        assert len(state_buffer) == self.traj_len + 1
        appearance_idx = obtain_active_agent_index(state_buffer)
        agent_selected = np.where((appearance_idx>=0).all(axis=1))[0]
        # assemble score function input
        state = dict(state_buffer[0])  # avoid changing the original state_dict
        for k,v in state.items():
            state[k]=v[agent_selected]
        state["image"] = (state["image"] / 255.).astype(np.float32)
        agent_from_world = state["agent_from_world"]
        yaw_current = state["yaw"]

        # transform traversed trajectories into the ego frame of a given state
        traj_inds = range(1, self.traj_len + 1)
        

        traj_pos = [state_buffer[traj_i]["centroid"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_yaw = [state_buffer[traj_i]["yaw"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_pos = np.stack(traj_pos, axis=1)  # [B, T, 2]

        traj_yaw = np.stack(traj_yaw, axis=1)  # [B, T]
        assert traj_pos.shape[0] == traj_yaw.shape[0]
        
        agent_traj_pos = transform_points(points=traj_pos, transf_matrix=agent_from_world)
        agent_traj_yaw = angular_distance(traj_yaw, yaw_current[:, None])



        state_torch = TensorUtils.to_torch(state, self.metric_algo.device)
        metrics = dict()
        state_torch["target_positions"] = TensorUtils.to_torch(agent_traj_pos,self.metric_algo.device).type(torch.float32)
        state_torch["target_yaws"] = TensorUtils.to_torch(agent_traj_yaw,self.metric_algo.device).type(torch.float32)
        traj_to_eval = dict()
        traj_to_eval["target_positions"] = agent_traj_pos
        traj_to_eval["target_yaws"] = agent_traj_yaw[:, :, None]
        # evaluate score of the ground truth state

        m = self.metric_algo.get_metrics(state_torch)
        for mk in m:
            metrics[mk] = m[mk]

        
        for k, v in self.perturbations.items():
            traj_perturbed = TensorUtils.to_torch(v.perturb(traj_to_eval), self.metric_algo.device)
            for kk,vv in traj_perturbed.items():
                traj_perturbed[kk]=vv.type(torch.float32)
            state_torch.update(traj_perturbed)
            m = self.metric_algo.get_metrics(state_torch)
            for mk in m:
                metrics["{}_{}".format(k, mk)] = m[mk]

        metrics= TensorUtils.to_numpy(metrics)
        step_metrics = dict()
        for k in metrics:
            met, met_mask = step_aggregate_per_scene(metrics[k], state["scene_index"], all_scene_index)
            assert np.all(met_mask > 0)  # since we will always use it for all agents
            step_metrics[k] = met
        return step_metrics

    def get_episode_metrics(self):
        assert len(self.state_buffer) >= self.traj_len+1
        all_scene_index = np.unique(self.state_buffer[-self.traj_len-1]["scene_index"])
        ep_metrics = self.compute_metric(self.state_buffer[-self.traj_len-1:], all_scene_index)


        return ep_metrics

class Occupancy_rolling(Occupancy_likelihood):
    def __init__(self, metric_algo, rolling_horizon, perturbations=None):
        super(Occupancy_rolling, self).__init__(metric_algo,perturbations)
        self.rolling_horizon = rolling_horizon

    def reset(self):
        self.state_buffer = []
        self._per_step = []
        self._per_step_mask = []
        self.total_steps = 0

    def __len__(self):
        return self.total_steps

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        state_info = dict(state_info)
        state_info["image"] = (state_info["image"] * 255.).astype(np.uint8)
        self.state_buffer.append(state_info)
        self.total_steps += 1
        step_metrics = self.compute_per_step(all_scene_index)
        if step_metrics is not None:
            self._per_step.append(step_metrics)


    def compute_per_step(self, all_scene_index):
        if len(self.state_buffer)<self.traj_len + 1:
            return None
        else:
            self.state_buffer = self.state_buffer[-self.traj_len-1:]

        appearance_idx = obtain_active_agent_index(self.state_buffer)
        agent_selected = np.where((appearance_idx>=0).all(axis=1))[0]
        # assemble score function input
        state = dict(self.state_buffer[0])  # avoid changing the original state_dict
        for k,v in state.items():
            if isinstance(v,np.ndarray):
                state[k]=v[agent_selected]
        state["image"] = (state["image"] / 255.).astype(np.float32)
        agent_from_world = state["agent_from_world"]
        yaw_current = state["yaw"]

        # transform traversed trajectories into the ego frame of a given state
        traj_inds = range(1, self.traj_len + 1)
        

        traj_pos = [self.state_buffer[traj_i]["centroid"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_yaw = [self.state_buffer[traj_i]["yaw"][appearance_idx[agent_selected,traj_i]] for traj_i in traj_inds]
        traj_pos = np.stack(traj_pos, axis=1)  # [B, T, 2]

        traj_yaw = np.stack(traj_yaw, axis=1)  # [B, T]
        assert traj_pos.shape[0] == traj_yaw.shape[0]
        
        agent_traj_pos = transform_points(points=traj_pos, transf_matrix=agent_from_world)
        agent_traj_yaw = angular_distance(traj_yaw, yaw_current[:, None])



        state_torch = TensorUtils.to_torch(state, self.metric_algo.device)
        if state_torch["target_positions"].shape[-2]<self.traj_len:
            return None
        metrics = dict()
        state_torch["target_positions"] = TensorUtils.to_torch(agent_traj_pos,self.metric_algo.device).type(torch.float32)
        state_torch["target_yaws"] = TensorUtils.to_torch(agent_traj_yaw,self.metric_algo.device).type(torch.float32)
        traj_to_eval = dict()
        traj_to_eval["target_positions"] = agent_traj_pos
        traj_to_eval["target_yaws"] = agent_traj_yaw[:, :, None]
        # evaluate score of the ground truth state
        if isinstance(self.rolling_horizon,int):
            m = self.metric_algo.get_metrics(state_torch,horizon=self.rolling_horizon)
            for mk in m:
                metrics[mk] = m[mk]
        elif isinstance(self.rolling_horizon,list):
            for horizon in self.rolling_horizon:
                m = self.metric_algo.get_metrics(state_torch,horizon=horizon)
                for mk in m:
                    metrics["{}_horizon_{}".format(mk,horizon)] = m[mk]
        
        for k, v in self.perturbations.items():
            traj_perturbed = TensorUtils.to_torch(v.perturb(traj_to_eval), self.metric_algo.device)
            for kk,vv in traj_perturbed.items():
                traj_perturbed[kk]=vv.type(torch.float32)
            state_torch.update(traj_perturbed)
            if isinstance(self.rolling_horizon,int):
                rolling_horizon = self.rolling_horizon
            elif isinstance(self.rolling_horizon,list):
                rolling_horizon = self.rolling_horizon[0]
            m = self.metric_algo.get_metrics(state_torch,horizon=rolling_horizon)
            for mk in m:
                metrics["{}_{}".format(k, mk)] = m[mk]

        metrics= TensorUtils.to_numpy(metrics)
        step_metrics = dict()
        for k in metrics:
            met, met_mask = step_aggregate_per_scene(metrics[k], state["scene_index"], all_scene_index)
            assert np.all(met_mask > 0)  # since we will always use it for all agents
            step_metrics[k] = met
        self.state_buffer.pop(0)
        return step_metrics

    def get_episode_metrics(self):
        
        scene_met = dict()
        for k in self._per_step[0]:
            scene_met_k = [step_met[k] for step_met in self._per_step]
            scene_met_k = np.stack(scene_met_k,axis=0)
            scene_met_k = scene_met_k.mean(0)
            scene_met[k] = scene_met_k   
        return scene_met

def obtain_active_agent_index(state_buffer):
    agents_indices = dict()
    appearance_idx = -np.ones([state_buffer[0]["scene_index"].shape[0],len(state_buffer)])
    appearance_idx[:,0]=np.arange(appearance_idx.shape[0])
    for i in range(state_buffer[0]["scene_index"].shape[0]):
        agents_indices[(state_buffer[0]["scene_index"][i],state_buffer[0]["track_id"][i])]=i

    for t in range(1,len(state_buffer)):
        for i in range(state_buffer[t]["scene_index"].shape[0]):
            agent_idx = (state_buffer[t]["scene_index"][i],state_buffer[t]["track_id"][i])
            if agent_idx in agents_indices:
                appearance_idx[agents_indices[agent_idx],t] = i

    return appearance_idx.astype(np.int32)

class Comfort(EnvMetrics):
    # compute stats at lower res than actual sim b/c for GT
    #       they are usually linearly inteprolated from low framerate data
    def __init__(self, sim_dt=0.1, stat_dt=0.5):
        super(Comfort, self).__init__()
        self.sim_dt = sim_dt
        self.stat_dt = stat_dt
        assert stat_dt >= sim_dt
        
    """Compute metrics relevant to comfort (speed, accels, jerk)"""
    def reset(self):
        self._per_step = []
        self._scene_masks = None

    @staticmethod
    def compute_per_step(state_info: dict, all_scene_index: np.ndarray):
        cur_state = np.concatenate([state_info["centroid"], state_info["yaw"][:,np.newaxis]], axis=1)
        return cur_state

    def add_step(self, state_info: dict, all_scene_index: np.ndarray):
        met = self.compute_per_step(state_info, all_scene_index)
        self._per_step.append(met)
        if self._scene_masks is None:
            agent_scene_index = state_info["scene_index"]
            self._scene_masks = agent_index_by_scene(agent_scene_index, all_scene_index)

    def get_episode_metrics(self):
        state_traj = np.stack(self._per_step, axis=1) # num_agents x T x 2
        # downsample
        dt_ratio = int(math.ceil(self.stat_dt / self.sim_dt))
        state_traj = state_traj[:,::dt_ratio]

        pos_traj = state_traj[..., :2]
        yaw_traj = state_traj[..., 2]
        vel_traj = np.diff(pos_traj, axis=1) / self.stat_dt
        speed = np.linalg.norm(vel_traj, axis=-1)

        accel_traj = np.diff(vel_traj, axis=1) / self.stat_dt
        accel_norm = np.linalg.norm(accel_traj, axis=-1)
        # magnitude of accel, not the direction
        lon_acc = np.abs(accel_norm * np.cos(yaw_traj[:,:accel_norm.shape[1]]))
        lat_acc = np.abs(accel_norm * np.sin(yaw_traj[:,:accel_norm.shape[1]]))
        jerk = np.abs(np.diff(accel_norm, axis=1) / self.stat_dt)

        # nanmean over each agent first, then group by scene
        speed = np.nanmean(speed, axis=1)
        lon_acc = np.nanmean(lon_acc, axis=1)
        lat_acc = np.nanmean(lat_acc, axis=1)
        jerk = np.nanmean(jerk, axis=1)

        met_per_scene = {
            'speed' : [],
            'lon_acc' : [],
            'lat_acc' : [],
            'jerk' : []
        }
        for scene_mask in self._scene_masks:
            met_per_scene['speed'].append(np.nanmean(speed[scene_mask]))
            met_per_scene['lon_acc'].append(np.nanmean(lon_acc[scene_mask]))
            met_per_scene['lat_acc'].append(np.nanmean(lat_acc[scene_mask]))
            met_per_scene['jerk'].append(np.nanmean(jerk[scene_mask]))

        for k,v in met_per_scene.items():
            met_per_scene[k] = np.array(v)

        return met_per_scene

if __name__=="__main__":
    gridinfo = {"offset":np.zeros(2),"step":0.3*np.ones(2)}
    occu = OccupancyGrid(gridinfo,sigma=0.5)
    pts = occu.get_neighboring_grid_points(np.array([0.5,0.6]))