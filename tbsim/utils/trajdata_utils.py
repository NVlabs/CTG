import torch
import numpy as np

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig
from trajdata.data_structures.state import StateTensor,StateArray

from trajdata import AgentBatch, AgentType
from trajdata.utils.arr_utils import angle_wrap

from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import RoadLane
from pathlib import Path
from trajdata.maps.map_api import MapAPI
from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np, transform_xyh_np
from trajdata.utils.state_utils import transform_state_np_2d
from typing import Union
from torch.nn.utils.rnn import pad_sequence

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from torch import Tensor
from trajdata.visualization.vis import draw_map, draw_agent, draw_history


full_keywords = ['image', 'target_positions', 'target_yaws', 'target_availabilities', 'history_positions', 'history_yaws', 'history_speeds', 'history_availabilities', 'curr_speed', 'centroid', 'yaw', 'type', 'extent', 'raster_from_agent', 'agent_from_raster', 'agent_from_world', 'world_from_agent', 'all_other_agents_curr_speed', 'all_other_agents_future_availability', 'all_other_agents_types', 'all_other_agents_extents', 'scene_index', 'all_other_agents_history_speeds', 'all_other_agents_history_availabilities', 'all_other_agents_history_availability', 'all_other_agents_history_positions', 'all_other_agents_future_positions', 'all_other_agents_history_yaws', 'all_other_agents_future_yaws', 'image']
major_keywords = ["history_positions", "history_yaws", "history_speeds", "extent", "history_availabilities", "curr_speed", "target_positions", "target_yaws", "target_availabilities", "all_other_agents_extents", "all_other_agents_history_speeds", "all_other_agents_history_yaws", "all_other_agents_history_positions", "all_other_agents_history_availabilities", "maps", "image"]+["type", "all_other_agents_types", "agent_hist"]
neighbor_keywords = ["all_other_agents_extents", "all_other_agents_history_speeds", "all_other_agents_history_yaws", "all_other_agents_history_positions", "all_other_agents_history_availabilities"]
major_scene_keywords = ["history_positions", "history_yaws", "history_speeds", "extent", "history_availabilities", "curr_speed", "target_positions", "target_yaws", "target_availabilities", "image", "maps"]
prep_keywords = ['agent_from_raster', 'raster_from_agent', 'agent_from_world', 'world_from_agent', 'centroid', 'yaw', 'raster_from_world'] + ['drivable_map', 'map_names']
prep_keywords_for_interaction_edge = ['raster_from_agent', 'agent_from_world', 'world_from_agent', 'yaw', 'scene_index', 'drivable_map', 'map_names'] + ['centroid', 'raster_from_world']
guidance_keywords = ['map_names', 'curr_speed', 'history_positions', 'history_yaws', 'agent_from_world', 'world_from_agent', 'scene_index', 'extent', 'drivable_map', 'raster_from_agent', 'agent_hist', 'extras']
guidance_keywords_training = ['map_names', 'curr_speed', 'history_positions', 'history_yaws', 'agent_from_world', 'world_from_agent', 'scene_index', 'extent', 'drivable_map', 'raster_from_agent', 'extras']

# TODO: these global hacks are pretty ugly, should refactor so this isn't necessary
BATCH_ENV = None
BATCH_RASTER_CFG = None

# need this for get_drivable_region_map
def set_global_trajdata_batch_env(batch_env):
    global BATCH_ENV
    BATCH_ENV = batch_env.split('-')[0] # if split is specified, remove it
# need this for rasterize_agents
def set_global_trajdata_batch_raster_cfg(raster_cfg):
    global BATCH_RASTER_CFG
    assert "include_hist" in raster_cfg
    assert "pixel_size" in raster_cfg
    assert "raster_size" in raster_cfg
    assert "ego_center" in raster_cfg
    assert "num_sem_layers" in raster_cfg
    assert "no_map_fill_value" in raster_cfg
    assert "drivable_layers" in raster_cfg
    BATCH_RASTER_CFG = raster_cfg

def get_raster_pix2m():
    return 1.0 / BATCH_RASTER_CFG["pixel_size"]

def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    
    if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
        state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
    else:
        assert state.shape[-1] == 8
    pos = state[..., :2]
    yaw = angle_wrap(torch.atan2(state[..., [-2]], state[..., [-1]]))
    speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def rasterize_agents_scene(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    
    b, a, t, _ = agent_hist_pos.shape
    _, _, _, h, w = maps.shape
    maps = maps.clone()
    agent_hist_pos = TensorUtils.unsqueeze_expand_at(agent_hist_pos,a,1)
    raster_hist_pos = transform_points_tensor(agent_hist_pos.reshape(b*a,-1,2), raster_from_agent.reshape(b*a,3,3)).reshape(b,a,a,t,2)
    raster_hist_pos[~agent_mask[:,None].repeat_interleave(a,1)] = 0.0  # Set invalid positions to 0.0 Will correct below
    
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels [B, A, A, T, 2]
    raster_hist_pos = raster_hist_pos.transpose(2,3)
    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, A, T, A]
    hist_image = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    
    ego_mask = torch.zeros_like(raster_hist_pos_flat,dtype=torch.bool)
    ego_mask[:,range(a),:,range(a)]=1
    agent_mask = torch.logical_not(ego_mask)


    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*agent_mask, src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*ego_mask, src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[..., 0] = 0  # correct the 0th index from invalid positions
    hist_image[..., -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, a, t, h, w)

    maps = torch.cat((hist_image, maps), dim=2)  # treat time as extra channels
    return maps


def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    maps = maps.clone()

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels

    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, T, A]

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
    hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, t, h, w)

    maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
    return maps


def get_drivable_region_map(maps):
    drivable_layers = BATCH_RASTER_CFG["drivable_layers"]
    if drivable_layers is None:
        # use defaults for known datasets
        env_name = BATCH_ENV
        if env_name in ['nusc_trainval', 'nusc_test', 'nusc_mini', 'main']:
            # first 3 layers are drivable
            # drivable_range = (-7, -4)
            # drivable_layers = [0, 1, 2] #[-7, -6, -5]
            drivable_layers = [0]
        elif env_name in ['lyft_train', 'lyft_train_full', 'lyft_val', 'lyft_sample']:
            # drivable_range = (-3, -2)
            drivable_layers = [0] #[-3]
        elif env_name in ['orca_maps', 'orca_no_maps']: # if using a mixed dataset, orca_no_maps may have dummy map layers to parse
            # drivable_range = (-2, -1)
            drivable_layers = [0] #[-2]
        elif env_name in ['nuplan_mini']:
            drivable_layers = [0]
        else:
            raise NotImplementedError("Must implement get_drivable_region_map for any new dataset from trajdata")

    drivable = None
    if len(drivable_layers) > 0:
        # convert to indices in the full rasterized stack of layers (which may include rasterized history)
        drivable_layers = -BATCH_RASTER_CFG["num_sem_layers"] + np.array(drivable_layers)
        if isinstance(maps, torch.Tensor):
            drivable = torch.amax(maps[..., drivable_layers, :, :], dim=-3)
            # agent-centric
            if len(drivable.shape) == 3:
                drivable_4dims = drivable.unsqueeze(1)
            else: # scene-centric
                drivable_4dims = drivable
            invalid_mask = ~compute_valid_map_mask(drivable_4dims)
            # TODO this is a bit hacky, shouldn't be computing these metrics at all if invalid
            # set batch indices with no map (infilled default value) to drivable by default for
            #       the sake of metrics
            drivable[invalid_mask] = 1.0
            drivable = drivable.bool()
        else:
            drivable = np.amax(maps[..., drivable_layers, :, :], axis=-3)
            # agent-centric
            if len(drivable.shape) == 3:
                drivable_4dims = drivable[:,np.newaxis]
            else: # scene-centric
                drivable_4dims = drivable
            invalid_mask = ~compute_valid_map_mask(drivable_4dims)
            # set batch indices with no map (infilled default value) to drivable by default for
            #       the sake of metrics
            drivable[invalid_mask] = 1.0
            drivable = drivable.astype(np.bool_)
    else:
        # the whole map is drivable
        if isinstance(maps, torch.Tensor):
            drivable_size = list(maps.size())
            drivable_size = drivable_size[:-3] + drivable_size[-2:]
            drivable = torch.ones(drivable_size, dtype=torch.bool).to(maps.device)
        else:
            drivable_size = list(maps.shape)
            drivable_size = drivable_size[:-3] + drivable_size[-2:]
            drivable = np.ones(drivable_size, dtype=bool)

    return drivable


def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch["agent_hist"].shape[1]
    fut_len = batch["agent_fut"].shape[1]
    b, a, neigh_len, _ = batch["neigh_hist"].shape
    device = batch["neigh_hist"].device
    empty_neighbor = a == 0
    device = batch["neigh_hist"].device
    if empty_neighbor:
        batch["neigh_hist"] = torch.ones(b, 1, hist_len, batch["neigh_hist"].shape[-1]).to(device) * torch.nan
        batch["neigh_fut"] = torch.ones(b, 1, fut_len, batch["neigh_fut"].shape[-1]).to(device) * torch.nan
        batch["neigh_types"] = torch.zeros(b, 1).to(device)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, batch["neigh_hist_extents"].shape[-1]).to(device)
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, batch["neigh_hist_extents"].shape[-1]).to(device)
    elif neigh_len < hist_len:
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch["neigh_hist"].shape[-1], device=device).to(device) * torch.nan
        batch["neigh_hist"] = torch.cat((hist_pad, batch["neigh_hist"]), dim=2)
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch["neigh_hist_extents"].shape[-1], device=device).to(device)
        batch["neigh_hist_extents"] = torch.cat((hist_pad, batch["neigh_hist_extents"]), dim=2)

def parse_scene_centric(batch: dict):
    num_agents = batch["num_agents"]
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])

    curr_pos = hist_pos[:,:,-1]
    curr_yaw = hist_yaw[:,:,-1]
    assert isinstance(batch["centered_agent_state"],StateTensor) or isinstance(batch["centered_agent_state"],StateArray)
    curr_speed = hist_speed[..., -1]
    centered_state = batch["centered_agent_state"]
    assert torch.all(centered_state[:, -1] == centered_state.heading[...,0])
    assert torch.all(centered_state[:, :2] == centered_state.position)
    centered_yaw = centered_state.heading[...,0]
    centered_pos = centered_state.position
    # assert torch.all(curr_yaw == centered_yaw), f"{curr_yaw} != {centered_yaw}"
    # assert torch.all(curr_pos == centered_pos), f"{curr_pos} != {centered_pos}"

    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type[agent_type < 0] = 0
    agent_type[agent_type == 1] = 3
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.


    centered_world_from_agent = torch.inverse(batch["centered_agent_from_world_tf"])



    # map-related
    if batch["maps"] is not None:
        map_res = batch["maps_resolution"][0,0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        
        centered_raster_from_agent = torch.Tensor([
            [map_res, 0, 0.25 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).to(centered_state.device)
        b,a = curr_yaw.shape[:2]
        centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
        
        agents_from_center = (GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros(b*a,2,device=curr_yaw.device))
                                @GeoUtils.transform_matrices(torch.zeros(b*a,device=curr_yaw.device),-curr_pos.reshape(-1,2))).reshape(*curr_yaw.shape[:2],3,3)
        center_from_agents = GeoUtils.transform_matrices(curr_yaw.flatten(),curr_pos.reshape(-1,2)).reshape(*curr_yaw.shape[:2],3,3)
        raster_from_center = centered_raster_from_agent @ agents_from_center
        center_from_raster = center_from_agents @ centered_agent_from_raster

        raster_from_world = batch["rasters_from_world_tf"]
        world_from_raster,_ = torch.linalg.inv_ex(raster_from_world)
        raster_from_world[torch.isnan(raster_from_world)] = 0.
        world_from_raster[torch.isnan(world_from_raster)] = 0.

        maps = rasterize_agents_scene(
            batch["maps"],
            hist_pos,
            hist_yaw,
            hist_mask,
            raster_from_center,
            map_res
        )
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0


    d = dict(
        image=maps,
        map_names=batch["map_names"],
        drivable_map=drivable_map,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_speeds=hist_speed,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=centered_pos,
        yaw=centered_yaw,
        type=agent_type,
        history_extent=agent_hist_extent * extent_scale,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=centered_raster_from_agent,
        agent_from_raster=centered_agent_from_raster,
        raster_from_center=raster_from_center,
        center_from_raster=center_from_raster,
        agents_from_center = agents_from_center,
        center_from_agents = center_from_agents,
        raster_from_world=raster_from_world,
        agent_from_world=batch["centered_agent_from_world_tf"],
        world_from_agent=centered_world_from_agent,
    )
    return d

def parse_node_centric(batch: dict, overwrite_nan=True):
    maybe_pad_neighbor(batch)
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"], nan_to_zero=overwrite_nan)
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"], nan_to_zero=overwrite_nan)
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]
    assert isinstance(curr_state,StateTensor) or isinstance(curr_state,StateArray)
    h1, h2 = curr_state[:, -1], curr_state.heading[...,0]
    p1, p2 = curr_state[:, :2], curr_state.position
    assert torch.all(h1[~torch.isnan(h1)] == h2[~torch.isnan(h2)])
    assert torch.all(p1[~torch.isnan(p1)] == p2[~torch.isnan(p2)])
    curr_yaw = curr_state.heading[...,0]
    curr_pos = curr_state.position

    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type = convert_nusc_type_to_lyft_type(agent_type)
    
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.
    neigh_indices = batch["neigh_indices"]
    neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = trajdata2posyawspeed(batch["neigh_hist"], nan_to_zero=overwrite_nan)
    neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = trajdata2posyawspeed(batch["neigh_fut"], nan_to_zero=overwrite_nan)
    neigh_curr_speed = neigh_hist_speed[..., -1]
    neigh_types = batch["neigh_types"]
    # convert nuscenes types to l5kit types
    neigh_types = convert_nusc_type_to_lyft_type(neigh_types)
    # mask out invalid extents
    neigh_hist_extents = batch["neigh_hist_extents"]
    neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

    world_from_agents = torch.inverse(batch["agents_from_world_tf"])

    raster_cfg = BATCH_RASTER_CFG
    map_res = 1.0 / raster_cfg["pixel_size"] # convert to pixels/meter
    h = w = raster_cfg["raster_size"]
    ego_cent = raster_cfg["ego_center"]

    raster_from_agent = torch.Tensor([
            [map_res, 0, ((1.0 + ego_cent[0])/2.0) * w],
            [0, map_res, ((1.0 + ego_cent[1])/2.0) * h],
            [0, 0, 1]
    ]).to(curr_state.device)
    
    bsize = batch["agents_from_world_tf"].shape[0]
    agent_from_raster = torch.inverse(raster_from_agent)
    raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=bsize, dim=0)
    agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=bsize, dim=0)
    raster_from_world = torch.bmm(raster_from_agent, batch["agents_from_world_tf"])

    all_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos.to(hist_pos.device)), dim=1)
    all_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw.to(hist_pos.device)), dim=1)
    all_hist_mask = torch.cat((hist_mask[:, None], neigh_hist_mask.to(hist_pos.device)), dim=1)

    maps_rasterize_in = batch["maps"]
    if maps_rasterize_in is None and BATCH_RASTER_CFG["include_hist"]:
        maps_rasterize_in = torch.empty((bsize, 0, h, w)).to(all_hist_pos.device)
    elif maps_rasterize_in is not None:
        maps_rasterize_in = verify_map(maps_rasterize_in)

    # num_sem_layers = maps_rasterize_in.size(1)

    if BATCH_RASTER_CFG["include_hist"]:
        # first T channels are rasterized history (single pixel where agent is)
        #       -1 for ego, 1 for others
        # last num_sem_layers are direclty the channels from data loader
        maps = rasterize_agents(
            maps_rasterize_in,
            all_hist_pos,
            all_hist_yaw,
            all_hist_mask,
            raster_from_agent,
            map_res
        )
    else:
        maps = maps_rasterize_in

    # print(maps.size())
    # import matplotlib
    # from matplotlib import pyplot as plt
    # for li in range(maps.size(1) - 3, maps.size(1)):
    #     print(li)
    #     cur_layer = maps[0,li].cpu().numpy() # h, w
    #     plt.imshow(cur_layer)
    #     plt.show()

    drivable_map = None
    if batch["maps"] is not None:
        drivable_map = get_drivable_region_map(maps_rasterize_in)

    extent_scale = 1.0
    d = dict(
        image=maps,
        map_names=batch["map_names"],
        drivable_map=drivable_map,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_speeds=hist_speed,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=curr_pos,
        yaw=curr_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=raster_from_agent,
        agent_from_raster=agent_from_raster,
        raster_from_world=raster_from_world,
        agent_from_world=batch["agents_from_world_tf"],
        world_from_agent=world_from_agents,
        all_other_agents_indices=neigh_indices,
        all_other_agents_history_positions=neigh_hist_pos,
        all_other_agents_history_yaws=neigh_hist_yaw,
        all_other_agents_history_speeds=neigh_hist_speed,
        all_other_agents_history_availabilities=neigh_hist_mask,
        all_other_agents_history_availability=neigh_hist_mask,  # dump hack to agree with l5kit's typo ...
        all_other_agents_curr_speed=neigh_curr_speed,
        all_other_agents_future_positions=neigh_fut_pos,
        all_other_agents_future_yaws=neigh_fut_yaw,
        all_other_agents_future_availability=neigh_fut_mask,
        all_other_agents_types=neigh_types,
        all_other_agents_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
        all_other_agents_history_extents=neigh_hist_extents * extent_scale,
    )
    if "agent_lanes" in batch:
        d["ego_lanes"] = batch["agent_lanes"]
    return d

def verify_map(batch_maps):
    '''
    Verify and expand map to the number of necessary channels if necessary.
    '''
    # if we use incl_map with trajdata, but the data does not contain a map, it will
    #       return 1 empty channel. Need to expand to the expected size given in config.
    if isinstance(batch_maps, torch.Tensor):
        if batch_maps.size(1) != BATCH_RASTER_CFG["num_sem_layers"]:
            assert batch_maps.size(1) == 1, "maps from trajdata have an unexpected number of layers"
            batch_maps = batch_maps.expand(-1, BATCH_RASTER_CFG["num_sem_layers"], -1, -1)
    else:
        if batch_maps.shape[1] != BATCH_RASTER_CFG["num_sem_layers"]:
            assert batch_maps.shape[1] == 1, "maps from trajdata have an unexpected number of layers"
            batch_maps = np.repeat(batch_maps, BATCH_RASTER_CFG["num_sem_layers"], axis=1)

    return batch_maps

def compute_valid_map_mask(batch_maps):
    '''
     - batch_maps (B, C, H, W)
    '''
    if isinstance(batch_maps, torch.Tensor):
        _, C, H, W = batch_maps.size()
        map_valid_mask = ~(torch.sum(torch.isclose(batch_maps, torch.tensor([BATCH_RASTER_CFG["no_map_fill_value"]], device=batch_maps.device)), dim=[1,2,3]) == C*H*W)
    else:
        B, C, H, W = batch_maps.shape
        map_valid_mask = ~(np.sum(np.isclose(batch_maps, np.array([BATCH_RASTER_CFG["no_map_fill_value"]])).reshape((B,-1)), axis=1) == C*H*W)
    return map_valid_mask

@torch.no_grad()
def parse_trajdata_batch(batch: dict, overwrite_nan=True):
    
    if "num_agents" in batch:
        # scene centric
        d = parse_scene_centric(batch)
        
    else:
        # agent centric
        d = parse_node_centric(batch, overwrite_nan=overwrite_nan)

    batch = dict(batch)
    batch.update(d)
    if overwrite_nan:
        for k,v in batch.items():
            if isinstance(v,torch.Tensor):
                batch[k]=v.nan_to_num(0)
    # batch.pop("agent_name", None)
    batch.pop("robot_fut", None)
    # batch.pop("scene_ids", None)
    return batch

TRAJDATA_AGENT_TYPE_MAP = {
    'unknown' : AgentType.UNKNOWN, 
    'vehicle' : AgentType.VEHICLE,
    'pedestrian' : AgentType.PEDESTRIAN,
    'bicycle' : AgentType.BICYCLE,
    'motorcycle' : AgentType.MOTORCYCLE
}

def get_modality_shapes(cfg: ExperimentConfig):
    num_sem_layers = 7 # backward compatibility for nuscenes
    if "num_sem_layers" in cfg.env.rasterizer.keys():
        num_sem_layers = cfg.env.rasterizer.num_sem_layers

    hist_layer_size = (cfg.algo.history_num_frames + 1) if cfg.env.rasterizer.include_hist else 0
    num_channels = hist_layer_size + num_sem_layers
    h = cfg.env.rasterizer.raster_size
    return dict(image=(num_channels, h, h))

# GPT Helper Functions ---------------------------------------------------------------
def select_agent_ind(x, i):
    return x[i]

def transform_coord_agents_to_world(pos_pred, yaw_pred, data_batch):
    from tbsim.utils.geometry_utils import transform_agents_to_world
    return transform_agents_to_world(pos_pred, yaw_pred, data_batch['world_from_agent'])

def transform_coord_world_to_agent_i(pos_pred, yaw_pred, data_batch, ind_i):
    # data_batch['agent_from_world'] has shape (B, 3, 3) (agent-centric) or (3, 3) (scene-centric)
    B = pos_pred.shape[0]
    if len(data_batch['agent_from_world'].shape) == 2:
        agent_i_from_world = data_batch['agent_from_world'].unsqueeze(0).expand(B, 3, 3)
    else:
        agent_i_from_world = data_batch['agent_from_world'][ind_i].unsqueeze(0).expand(B, 3, 3)
    from tbsim.utils.geometry_utils import transform_agents_to_world
    return transform_agents_to_world(pos_pred, yaw_pred, agent_i_from_world)

def get_left_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=''):
    return get_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=visualize_projection, mode='left')

def get_right_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=''):
    return get_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=visualize_projection, mode='right')

def get_current_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=''):
    return get_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection=visualize_projection, mode='current')

def get_lane_projection(pos_pred, yaw_pred, data_batch, visualize_projection='', mode='current'):
    '''
    Query projected lane points in agent coordinate by passing in predicted trajectories in agent coordinate.
    - param pos_pred (B, N, T, 2)
    - param yaw_pred (B, N, T, 1)
    - param data_batch (dict)
    - param mode (str) - 'left', 'right', 'current'
    - return agent_future_xyh_on_lane (B, N, T, 3)
    '''
    pos_pred = pos_pred.detach()
    yaw_pred = yaw_pred.detach()
    B, N, _, _ = pos_pred.shape

    # threshold to query a next lane
    lane_end_d_th = 10
    max_lookahead = 5

    vec_map = data_batch['vec_map']
    world_from_agent = data_batch['world_from_agent'].cpu().numpy()
    agent_from_world = data_batch['agent_from_world'].cpu().numpy()

    current_lane_list = []
    for i in range(B):
        xyh = np.array([0, 0, 0])
        xyh_world = transform_xyh_np(xyh, world_from_agent[i])
        xyzh_world = np.concatenate([xyh_world[:2], [0], xyh_world[2:]])
        lanes = vec_map.get_current_lane(xyzh_world, max_dist=80, max_heading_error=0.25*np.pi)
        
        valid = False
        if len(lanes) > 0:  
            lane = lanes[0]
            if mode == 'current':
                valid = True
                xyh_on_lane_world = lane.center.points[...,[0,1,3]]
                xyh_on_lane = transform_xyh_np(xyh_on_lane_world, agent_from_world[i])

            elif mode in ['left', 'right']:
                xyh_on_lane = get_neighbor_lane(lane, data_batch["extras"]["closest_lane_point"][i], agent_from_world[i], mode) 
                if xyh_on_lane is not None:
                    valid = True

            else:
                raise ValueError(f'Unknown mode {mode}')
            
        if valid:            
            # keep extending lanes until the furthest point is larger than lane_end_d_th
            for _ in range(max_lookahead):
                # print('xyh_on_lane', xyh_on_lane)
                xyh_on_lane_clipped = np.clip(xyh_on_lane, a_min=0, a_max=np.inf)
                # (S_point, 3) -> (S_point)
                xyh_on_lane_norm = np.linalg.norm(xyh_on_lane_clipped[...,:2], axis=-1)
                # print('xyh_on_lane_norm.shape', xyh_on_lane_norm.shape, xyh_on_lane_norm)
                # print('xyh_on_lane_norm.max()', xyh_on_lane_norm.max())
                if xyh_on_lane_norm.max() < lane_end_d_th and len(lane.next_lanes) > 0:
                    # randomly select the next lane
                    next_lane_str_list = list(lane.next_lanes)
                    if len(next_lane_str_list) > 0:
                        next_lane = vec_map.get_road_lane(next_lane_str_list[0])
                        xyh_next_lane_world = next_lane.center.points[...,[0,1,3]]
                        xyh_next_lane = transform_xyh_np(xyh_next_lane_world, agent_from_world[i])

                        # (S_point, 3) -> (S_point*2, 3)
                        xyh_on_lane = np.concatenate([xyh_on_lane, xyh_next_lane], axis=0)
                    else:
                        break
                else:
                    break
        else:
            xyh_on_lane = np.ones((1,3))*np.nan

        current_lane_list.append(xyh_on_lane)

    max_pts = max([lane.shape[0] for lane in current_lane_list])
    current_lane_list_paded = [np.pad(lane, ((0, max_pts - lane.shape[0]), (0, 0)), mode='constant', constant_values=np.nan) for lane in current_lane_list]
    current_lane = np.stack(current_lane_list_paded, axis=0)
    current_lane = torch.from_numpy(current_lane).float().to(pos_pred.device)

    # if no valid lane, return the original prediction
    if torch.isnan(current_lane).to(torch.float32).mean() == 1:
        raise
        return torch.cat([pos_pred, yaw_pred], dim=-1)
    else:
        # sort it by x
        current_lane = torch.gather(current_lane, 1, torch.argsort(current_lane[:, :, 0], dim=1).unsqueeze(-1).expand(-1, -1, 3))
        
        # (B, S_point, 3) -> (B, N, S_point, 3)
        current_lane = current_lane.unsqueeze(1).repeat(1,N,1,1)

        agent_future_xyh = torch.cat([pos_pred, yaw_pred], dim=-1)
        agent_future_xyh_on_lane = project_onto(agent_future_xyh, current_lane, visualize_projection)
        
        return agent_future_xyh_on_lane

def get_neighbor_lane(current_lane, neighbor_lanes, agent_from_world, mode):
    '''
    Get left/right lane of current lane
    - param current_lane: lane object
    - param neighbor_lanes: (S_seg, S_point, 3)
    - param agent_from_world: (3, 3)
    - param mode: 'left' or 'right'
    - return neighbor_lane: (S_point, 3)
    '''
    neighbor_lanes = neighbor_lanes.detach().clone().cpu().numpy()
    if len(neighbor_lanes) < 2:
        return None
    # remove current lane
    # neighbor_lanes = neighbor_lanes[1:]

    if mode == 'left':
        left_lanes = list(current_lane.adj_lanes_left)
        if len(left_lanes) > 0:
            return left_lanes[0].center.points[...,[0,1,3]]
        dir = 1
    elif mode == 'right':
        right_lanes = list(current_lane.adj_lanes_right)
        if len(right_lanes) > 0:
            return right_lanes[0].center.points[...,[0,1,3]]
        dir = -1
    else:
        raise ValueError('mode should be left or right')

    xyh_on_lane_world = current_lane.center.points[...,[0,1,3]]
    xyh_on_lane = transform_xyh_np(xyh_on_lane_world, agent_from_world)
    xy_current = xyh_on_lane[0,:2]

    # Find the closest left_lane / right_lane
    S_seg, S_point, k = neighbor_lanes.shape
    # Repace nan with inf
    neighbor_lanes = np.where(np.isnan(neighbor_lanes), np.inf, neighbor_lanes)
    if_left = neighbor_lanes[...,1]*dir > 1.5
    if_ahead = neighbor_lanes[...,0] > 0
    
    neighbor_lanes_left = np.where(np.expand_dims(if_left, axis=-1).repeat(repeats=k, axis=-1), neighbor_lanes, np.ones_like(neighbor_lanes)*np.inf)
    
    neighbor_lanes_left_ahead = np.where(np.expand_dims(if_ahead, axis=-1).repeat(repeats=k, axis=-1), neighbor_lanes_left, np.ones_like(neighbor_lanes_left)*np.inf)

    # [S_seg*S_point, 2] - [2] -> [S_seg*S_point]
    dist = np.linalg.norm(neighbor_lanes_left_ahead.reshape(S_seg*S_point, -1)[...,:2] - xy_current, axis=-1)
    # [S_seg, S_point] -> [S_seg] -> [1]
    # print('dist', dist.reshape(S_seg, S_point))
    min_ind = np.argmin(np.min(dist.reshape(S_seg, S_point), axis=-1))
    # print('min_ind', min_ind)
    neighbor_lane = neighbor_lanes[min_ind]
    # remove padded points in the end
    inf_inds = np.argwhere(np.isinf(neighbor_lane))
    if len(inf_inds) > 0:
        end_ind = inf_inds[0][0]
        neighbor_lane = neighbor_lane[:end_ind]
    if len(neighbor_lane) == 0:
        return None

    return neighbor_lane

def project_onto(xyh: torch.tensor, points: torch.tensor, visualize_projection: str='') -> np.ndarray:
    """Project the given points onto this Polyline.

    Args:
        xyh: Points to project, of shape (B, N, T, 3)
        points: lane points to project onto, of shape (B, N, S_point, 3)
    Returns:
        xyh_projection: The projected points, of shape (B, N, T, 3)

    """
    B, N, T, _ = xyh.shape
    # (B, N, T, 3) -> (B*N, T, 3)
    xyh = xyh.reshape(B*N, T, -1)
    # (B*N, T, 3) -> (B*N, T, 1, 2)
    xy = xyh[:, :, None, :2]

    # replace nan with 0
    # (B, N, S_point, 3)
    nan_mask = torch.isnan(points)
    # points = points.masked_fill_(nan_mask, 10000.0)

    S_point = points.shape[2]
    points = points.reshape(B*N, S_point, 3)

    # p0, p1 are (B*N, 1, S_point, 2)
    p0: torch.tensor = points[:, None, :-1, :2]
    p1: torch.tensor = points[:, None, 1:, :2]

    # 1. Compute projections of each point to each line segment in a
    #    batched manner.
    # (B*N, 1, S_point, 2), (B*N, 1, S_point, 2) -> (B*N, 1, S_point, 2)
    line_seg_diffs: torch.tensor = p1 - p0
    # (B*N, T, 1, 2), (B*N, 1, S_point, 2) -> (B*N, T, S_point, 2)
    point_seg_diffs: torch.tensor = xy - p0

    # (B*N, T, S_point, 2), (B*N, 1, S_point, 2) -> (B*N, T, S_point, 1)
    dot_products: torch.tensor = (point_seg_diffs * line_seg_diffs).sum(
        dim=-1, keepdim=True
    )
    # (B*N, 1, S_point, 2) -> (B*N, 1, S_point, 1)
    norms: torch.tensor = torch.norm(line_seg_diffs, dim=-1, keepdim=True) ** 2
    # print('norms[0]', norms[0].shape, norms[0])
    # print('dot_products[0]', dot_products[0].shape, dot_products[0])
    # Clip ensures that the projected point stays within the line segment boundaries.
    # (B*N, T, S_point, 2)
    projs: torch.tensor = (
        p0 + torch.clip(dot_products / norms, min=0, max=1) * line_seg_diffs
    )

    # 2. Find the nearest projections to the original points.
    # (B*N, T, 1, 2), (B*N, T, S_point, 2) -> (B*N, T, S_point) -> (B*N, T)
    dist = torch.norm(xy - projs, dim=-1)
    dist = torch.where(torch.isnan(dist), torch.tensor(float('inf'), dtype=dist.dtype, device=dist.device), dist)
    closest_proj_idxs = dist.argmin(dim=-1)

    # Adding in the heading of the corresponding p0 point (which makes
    # sense as p0 to p1 is a line => same heading along it).
    # (B*N, T)
    indices_dim1 = torch.arange(B * N)[:, None].repeat(1, T)
    # (1, T)
    indices_dim2 = torch.arange(T)[None, :]

    # (B*N, T, 3)
    xyh_projection = torch.cat(
        [
            projs[indices_dim1, indices_dim2, closest_proj_idxs],
            torch.gather(points[...,-1], 1, closest_proj_idxs).unsqueeze(-1),
        ],
        dim=-1,
    )

    xyh_projection = xyh_projection.reshape(B, N, T, 3)

    # sanity check
    ind = 6
    if visualize_projection:
        from matplotlib import pyplot as plt
        lane = points.reshape(B, N, S_point, 3)[ind, 0, :, :2].detach().clone().cpu()
        original = xyh.reshape(B, N, T, 3).detach().clone().cpu()[ind, 0, :, :2]
        projection = xyh_projection[ind, 0, :, :2].detach().clone().cpu()

        plt.scatter(lane[...,0], lane[...,1], label='lane')
        plt.scatter(original[...,0], original[...,1], label='original')
        plt.scatter(projection[...,0], projection[...,1], label='projection')

        # print('lane', lane.shape, lane)
        # print('original', original.shape, original)
        # print('projection', projection.shape, projection)
        
        # print('lane.shape, original.shape, projection.shape', lane.shape, original.shape, projection.shape)
        plt.legend()
        plt.savefig('nusc_results/lane_projection_'+visualize_projection+'.png')
        plt.close()
        # raise
    
    return xyh_projection

# ---------------------------------------------------------------


def get_current_lane_by_point(point_xyh, vector_map, world_from_agent_tf=None, agent_from_world_tf=None):
    '''
    param point_xyh, (3) in agent coordinate
    param vector_map, VectorMap
    output point_xyh, (S_point, 4) in agent coordinate
    '''
    S_point = 40
    map_max_dist = 2
    max_heading_error = 0.125*np.pi
    dist_weight = 1.0
    heading_weight = 0.1

    # transform coordinate from agent to world
    # point_xyh_transformed = np.zeros(3)
    # point_xyh_transformed[:2] = transform_coords_np(
    #     point_xyh[None, :][:, :2], world_from_agent_tf
    # )[0]
    # point_xyh_transformed[-1] = transform_angles_np(
    #     point_xyh[None, :][:, -1], world_from_agent_tf
    # )[0]
    
    # fill in 0.0 for "z" coordinate
    point_xyzh = np.concatenate([point_xyh[:2], [0.0], angle_wrap(point_xyh[-1:])], axis=0)
    # [4] -> List[RoadLane]
    possible_lanes = vector_map.get_current_lane(point_xyzh, max_dist=map_max_dist, max_heading_error=max_heading_error)

    if len(possible_lanes) > 0:
        
        possible_lane = possible_lanes[0]
        xyzh_on_lane_original = possible_lane.center.points
        
        dist = dist_weight * np.linalg.norm(xyzh_on_lane_original[:, :2] - point_xyzh[:2], axis=-1) + heading_weight * np.abs(xyzh_on_lane_original[:, -1] - point_xyzh[-1])
        inds = np.argsort(dist).tolist()

        # transform coordinate from world to agent
        # xyzh_on_lane = xyzh_on_lane_original.copy()
        # xyzh_on_lane[:, :2] = transform_coords_np(
        #     xyzh_on_lane_original[:, :2], agent_from_world_tf
        # )
        # xyzh_on_lane[:, -1] = transform_angles_np(
        #     xyzh_on_lane_original[:, -1], agent_from_world_tf
        # )

        xyzh_on_lane = xyzh_on_lane[inds][:np.amin([len(xyzh_on_lane), S_point])]

        # # fill to S_point with nan
        # if xyzh_on_lane.shape[0] < S_point:
        #     xyzh_on_lane = np.concatenate([xyzh_on_lane, np.full((S_point-xyzh_on_lane.shape[0], 4), np.nan)], axis=0)

    else:
        lanes_points = np.full((S_point, 4), np.nan)

    # convert to tensor
    lanes_points = torch.as_tensor(lanes_points, dtype=torch.float)
    return lanes_points

def get_closest_lane_point_for_one_agent(agent_history, vector_map, world_from_agent_tf, agent_from_world_tf, params):
        '''
        Output:
            lanes_points: (S_seg, S_point, 4)
        '''
        S_seg = params['S_seg']
        S_point = params['S_point']
        map_max_dist = params['map_max_dist']
        max_heading_error = params['max_heading_error']
        dist_weight = params['dist_weight']
        heading_weight = params['heading_weight']
        ahead_threshold = params['ahead_threshold']
        
        agent_traj = StateArray.from_array(
            agent_history,
            "x,y,xd,yd,xdd,ydd,s,c",
        )
        agent_traj_xyh_world = transform_state_np_2d(
            agent_traj, world_from_agent_tf
        ).as_format("x,y,h")

        # Use cached kdtree to find closest lane point
        
        # get the current position of the agent
        point_xyh = agent_traj_xyh_world[-1]

        # fill in 0.0 for "z" coordinate
        point_xyzh = np.concatenate([point_xyh[:2], [0.0], angle_wrap(point_xyh[-1:])], axis=0)
        # [4] -> List[RoadLane]
        possible_lanes = vector_map.get_current_lane(point_xyzh, max_dist=map_max_dist, max_heading_error=max_heading_error)

        num_lanes = np.amin([len(possible_lanes), S_seg])
        possible_lanes_shortened = possible_lanes[:num_lanes]
        
        if len(possible_lanes_shortened) > 0:
            lanes_points_list = []
            for i, possible_lane in enumerate(possible_lanes_shortened):
                xyzh_on_lane_original = possible_lane.center.points
                
                dist = dist_weight * np.linalg.norm(xyzh_on_lane_original[:, :2] - point_xyzh[:2], axis=-1) + heading_weight * np.abs(xyzh_on_lane_original[:, -1] - point_xyzh[-1])
                inds = np.argsort(dist).tolist()

                # transform to coordinate from world to agent/scene
                xyzh_on_lane = xyzh_on_lane_original.copy()
                xyzh_on_lane[:, :2] = transform_coords_np(
                    xyzh_on_lane_original[:, :2], agent_from_world_tf
                )
                xyzh_on_lane[:, -1] = transform_angles_np(
                    xyzh_on_lane_original[:, -1], agent_from_world_tf
                )

                # sort waypoints
                xyzh_on_lane = xyzh_on_lane[inds]
                # only keep waypoints ahead
                waypoints_ahead_mask = xyzh_on_lane[:, 0] > ahead_threshold
                xyzh_on_lane = xyzh_on_lane[waypoints_ahead_mask]
                xyzh_on_lane = xyzh_on_lane[:np.amin([len(xyzh_on_lane), S_point])]

                # fill to S_point with nan
                if xyzh_on_lane.shape[0] < S_point:
                    xyzh_on_lane = np.concatenate([xyzh_on_lane, np.full((S_point-xyzh_on_lane.shape[0], 4), np.nan)], axis=0)

                lanes_points_list.append(xyzh_on_lane)

            lanes_points = np.stack(lanes_points_list, axis=0)

        else:
            lanes_points = np.full((S_seg, S_point, 4), np.nan)

        # fill to S_point with nan
        if lanes_points.shape[0] < S_seg:
            lanes_points = np.concatenate([lanes_points, np.full((S_seg-lanes_points.shape[0], S_point, 4), np.nan)], axis=0)

        # convert to tensor
        lanes_points = torch.as_tensor(lanes_points, dtype=torch.float)
        return lanes_points

def get_closest_lane_point_wrapper(vec_map_params={}):
    # print('vec_map_params', vec_map_params)
    if vec_map_params == {}:
        # fill in default values
        vec_map_params = {
            'S_seg': 15,
            'S_point': 80,
            'map_max_dist': 80,
            'max_heading_error': 0.25*np.pi,
            'ahead_threshold': -40,
            'dist_weight': 1.0,
            'heading_weight': 0.1,
        }

    def get_closest_lane_point(element: Union[AgentBatchElement, SceneBatchElement]):
        """Closest lane for predicted agent.
        lane_points (torch tensor): 
        [S_seg, S_point, 3] (if agent-centric) 
        [M, S_seg, S_point, 3] (if scene-centric)
        only consider the x, y, heading of points
        """
        # Transform from agent coordinate frame to world coordinate frame.
        vector_map: VectorMap = element.vec_map


        # resolution = 1
        if isinstance(element, AgentBatchElement):
            agent_from_world_tf = element.agent_from_world_tf
            world_from_agent_tf = np.linalg.inv(agent_from_world_tf)
            # (T, 8)
            agent_history = element.agent_history_np
            
            lane_points_cur_agent = get_closest_lane_point_for_one_agent(agent_history, vector_map, world_from_agent_tf, agent_from_world_tf, vec_map_params)
            # (x, y, z, heading) -> (x, y, heading)
            lane_points = lane_points_cur_agent[..., [0, 1, 3]]
            assert len(lane_points.shape) == 3, f"lane_points.shape: {lane_points.shape}"
        
        elif isinstance(element, SceneBatchElement):
            agent_histories = element.agent_histories
            world_from_agent_tf = element.centered_world_from_agent_tf
            agent_from_world_tf = element.centered_agent_from_world_tf

            lane_points_list = []
            for i, agent_history in enumerate(agent_histories):
                lane_points_cur_agent = get_closest_lane_point_for_one_agent(agent_history, vector_map, world_from_agent_tf, agent_from_world_tf, vec_map_params)
                lane_points_list.append(lane_points_cur_agent)

            # (M, S_seg, S_p, 4)
            lane_points = pad_sequence(lane_points_list,
                batch_first=True,
                padding_value=np.nan,
            )
            # (x, y, z, heading) -> (x, y, heading)
            lane_points = lane_points[..., [0, 1, 3]]
            assert len(lane_points.shape) == 4, f"lane_points.shape: {lane_points.shape}"
        else:
            raise ValueError(f"Unknown element type: {type(element)}")

        return lane_points
    
    return get_closest_lane_point

def get_full_fut_traj(element: Union[AgentBatchElement, SceneBatchElement]):
    """Get mask for moving agents.
    """
    fut_sec = 20
    dt = 0.1
    T = int(fut_sec / dt)
    if isinstance(element, AgentBatchElement):
        # (T, 8), (T)
        fut_traj, _ = element.get_agent_future(element.agent_info, (fut_sec, fut_sec))
        t, k = fut_traj.shape
        if T > t:
            pad = np.zeros((T-t, k))
            fut_traj = np.concatenate([fut_traj, pad], axis=0)
        else:
            fut_traj = fut_traj[:T]
    elif isinstance(element, SceneBatchElement):
        # (M, T, 8), (M, T)
        fut_traj, _, _ = element.get_agents_future((fut_sec, fut_sec), element.nearby_agents)
        fut_traj_list = []
        for arr in fut_traj:
            t, k = arr.shape
            if T > t:
                pad = np.zeros((T-t, k))
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:T]
            fut_traj_list.append(arr)
        fut_traj = np.stack(fut_traj_list, axis=0)
    else:
        raise ValueError(f"Unknown element type: {type(element)}")
    fut_traj = torch.as_tensor(fut_traj, dtype=torch.float)
    return fut_traj

def get_full_fut_valid(element: Union[AgentBatchElement, SceneBatchElement]):
    """Get mask for moving agents.
    """
    fut_sec = 20.0
    dt = 0.1
    T = int(fut_sec / dt)
    if isinstance(element, AgentBatchElement):
        # (T, 3)
        _, fut_valid = element.get_agent_future(element.agent_info, (fut_sec, fut_sec))
        t, k = fut_valid.shape
        if T > t:
            pad = np.zeros((T-t, k))
            fut_valid = np.concatenate([fut_valid, pad], axis=0)
        else:
            fut_valid = fut_valid[:T]
        # (T, 3) -> (T)
        fut_valid = fut_valid[...,0]
    elif isinstance(element, SceneBatchElement):
        # (M, T)
        _, fut_valid, _ = element.get_agents_future((fut_sec, fut_sec), element.nearby_agents)
        fut_valid_list = []
        for arr in fut_valid:
            t, k = arr.shape
            if T > t:
                pad = np.zeros((T-t, k))
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:T]
            fut_valid_list.append(arr)
        fut_valid = np.stack(fut_valid_list, axis=0)
        # (M, T, 3) -> (M, T)
        fut_valid = fut_valid[...,0]
    else:
        raise ValueError(f"Unknown element type: {type(element)}")
    fut_valid = torch.as_tensor(fut_valid, dtype=torch.float)
    return fut_valid

def get_stationary_mask(data_batch, disable_control_on_stationary, moving_speed_th):
    '''
    This function is called when disable_control_on_stationary is not False.
    '''
    # (B, (M), T, 8)
    full_fut_speed = data_batch['extras']['full_fut_traj'][...,2]
    # (B, (M), T)
    full_fut_valid = data_batch['extras']['full_fut_valid']
    # mask out those stationary all the time in GT
    if disable_control_on_stationary == True or 'any_speed' in disable_control_on_stationary:
        # (B, (M), T), (B, (M), T) -> (B, (M))
        moving_mask = ((full_fut_speed > moving_speed_th).to(torch.float) * full_fut_valid).sum(dim=-1) > 0
    # mask out those stationary at the first timestep in GT
    elif 'current_speed' in disable_control_on_stationary:
        # technically we are using one timestep from the current timestep
        # (B, (M), T), (B, (M), T) -> (B, (M))
        moving_mask = ((full_fut_speed[...,0] > moving_speed_th).to(torch.float) * full_fut_valid[...,0]) > 0
    else:
        moving_mask = torch.ones(*full_fut_valid.shape[:-1], dtype=torch.bool, device=full_fut_valid.device)
    stationary_mask = ~moving_mask
    # print('0 stationary_mask', stationary_mask)
    # mask out those not on lane (in parking lot)
    if 'on_lane' in disable_control_on_stationary:
        map_max_dist = 2.0
        # assumer vec map in agent-centric coordinates
        lane_points = data_batch['extras']['closest_lane_point'].detach().clone()
        lane_points = torch.where(torch.isnan(lane_points), torch.tensor(10e8, dtype=lane_points.dtype, device=lane_points.device), lane_points)
        
        # (B, (M), S_seg, S_point, 3) -> (B, (M), S_seg, S_point)
        dist_to_lane = torch.norm(lane_points[...,:2], dim=-1)
        # (B, (M), S_seg, S_point) -> (B, (M), S_seg*S_point)
        dist_to_lane = dist_to_lane.view(*dist_to_lane.shape[:-2], -1)
        # (B, (M), S_seg*S_point) -> (B, (M))
        parking_mask = dist_to_lane.min(dim=-1)[0] > map_max_dist
        # print("dist_to_lane.min(dim=-1)[0]", dist_to_lane.min(dim=-1)[0])
        # print('parking_mask', parking_mask)
        stationary_mask = stationary_mask | parking_mask
    # This mode is in order to test the influence of fixing the center vehicle on other vehicles
    if 'center' in disable_control_on_stationary:
        # (B, M)
        if len(stationary_mask.shape) == 2:
            stationary_mask[:,0] = True
        else: # (B)
            stationary_mask[0] = True
    # print('1 stationary_mask', stationary_mask)
    return stationary_mask

#-----------------------------------------------------------------------------#
#------------------------------ data conversion ------------------------------#
#-----------------------------------------------------------------------------#
def convert_scene_data_to_agent_coordinates(data, merge_BM=False, max_neighbor_num=None, max_neighbor_dist=np.inf, keep_order_of_neighbors=False):
    '''
    This will convert a scene-centric data batch to agent-centric coordinates

    keep_order_of_neighbors: if True, the order of neighbors (including the agents themselves) will be kept, this is used for agent-agent edge for customized social attention layer. If False, the order of neighbors will be sorted by distance, this will be the same as that in agent-centric data.
    '''
    new_data = convert_scene_obs_to_agent_prep(data)

    # copy over unchanged fields
    new_data['history_speeds'] = data['history_speeds']
    new_data['extent'] = data['extent']
    new_data['history_availabilities'] = data['history_availabilities']
    new_data['image'] = data['image']
    if 'maps' in data:
        new_data['maps'] = data['maps']
    if 'extras' in data:
        new_data['extras'] = data['extras']
    new_data['curr_speed'] = data['curr_speed']
    new_data['target_availabilities'] = data['target_availabilities']
    new_data['type'] = data['type']

    # TBD: this is used for debugging but not by the model
    new_data['agent_names'] = data['agent_names']

    B, M, _, _ = data["history_positions"].shape

    for pos_k, yaw_k, avai_k in zip(["history_positions", "target_positions"], ["history_yaws", "target_yaws"], ["history_availabilities", "target_availabilities"]):
        pos_k_list, yaw_k_list = [], []
        for i in range(B):
            pos_k_i = GeoUtils.transform_points_tensor(data[pos_k][i],data["agents_from_center"][i])*new_data[avai_k][i].unsqueeze(-1)
            yaw_k_i = data[yaw_k][i]-new_data['yaw'][i][:, None, None]+new_data['yaw'][i][0]*new_data[avai_k][i].unsqueeze(-1)

            pos_k_list.append(pos_k_i)
            yaw_k_list.append(yaw_k_i)

        new_data[pos_k] = torch.stack(pos_k_list, dim=0)
        new_data[yaw_k] = angle_wrap(torch.stack(yaw_k_list, dim=0))


    if 'closest_lane_point' in data['extras']:
        B, M, S_seg, S_p, _ = data['extras']['closest_lane_point'].shape
        closest_lanes_points_list = []
        for i in range(B):
            # [M, S_seg, S_p, 3] -> [M, S_seg*S_p, 3]
            closest_lane_point = data['extras']['closest_lane_point'][i].reshape(M, S_seg*S_p, -1)
            closest_lane_point_pos = closest_lane_point[..., :2] # [M, S_seg*S_p, 2]
            closest_lane_point_yaw = closest_lane_point[..., [2]] # [M, S_seg*S_p, 1]

            closest_lane_point_pos_transformed = GeoUtils.transform_points_tensor(closest_lane_point_pos,data["agents_from_center"][i])
            closest_lane_point_yaw_transformed = angle_wrap(closest_lane_point_yaw-new_data["yaw"][i][:, None, None]+new_data["yaw"][i, 0])

            closest_lanes_points = torch.cat((closest_lane_point_pos_transformed, closest_lane_point_yaw_transformed), -1).reshape(M, S_seg, S_p, -1)

            closest_lanes_points_list.append(closest_lanes_points)
        new_data['extras'] = {'closest_lane_point': torch.stack(closest_lanes_points_list, dim=0)}

    # "all_other_agents_history_positions", "all_other_agents_history_yaws"
    # "all_other_agents_history_speeds", "all_other_agents_extents", "all_other_agents_history_availabilities", "all_other_agents_types"
    # get neighbor indices for each agent

    all_other_agents_history_positions_list = []
    all_other_agents_history_yaws_list = []
    all_other_agents_history_speeds_list = []
    all_other_agents_extent_list = []
    all_other_agents_types_list = []
    all_other_agents_history_availabilities_list = []

    for i in range(B):

        # M, T
        # print('new_data["history_availabilities"].shape', data["history_availabilities"][i].shape, data["history_availabilities"][i])
        # print('(data["history_availabilities"]==np.nan).to(float).shape', (data["history_availabilities"][i]==np.nan).to(float).shape)
        avail = torch.sum((data["history_availabilities"][i]==True).to(float), dim=-1)
        # print('avail', avail.shape, avail)
        unavail_inds = torch.where(avail==0)[0]
        # print('avail_inds', avail_inds.shape, avail_inds)

        # (M, M, 2) - (1, M, 2) -> (M, M)
        if keep_order_of_neighbors:
            nb_idx = torch.arange(M).unsqueeze(0).repeat(M, 1)
        else:
            agent_distances = torch.linalg.norm(TensorUtils.unsqueeze_expand_at(new_data['centroid'][i],M,1) - new_data['centroid'][i].unsqueeze(0), axis=-1)

            agent_distances[:, unavail_inds] = np.inf
            agent_distances[unavail_inds, :] = np.inf

            nb_idx = agent_distances.argsort(dim=1)
            agent_distances_sorted = agent_distances.sort(dim=1).values
            

            # (M, M) -> (M, Q) set those outside the distance limit to False
            if max_neighbor_num is not None:
                nb_idx = nb_idx[:, 1:max_neighbor_num+1]
                agent_distances_sorted = agent_distances_sorted[:, 1:max_neighbor_num+1]
            else:
                nb_idx = nb_idx[:, 1:]
                agent_distances_sorted = agent_distances_sorted[:, 1:]

            # mark those outside the distance limit (M, M)
            nb_idx[agent_distances_sorted >= max_neighbor_dist] = -1

        all_other_agents_history_positions_list_sub = []
        all_other_agents_history_yaws_list_sub = []
        all_other_agents_history_speeds_list_sub = []
        all_other_agents_extent_list_sub = []
        all_other_agents_types_list_sub = []
        all_other_agents_history_availabilities_list_sub = []

        agent_from_world = new_data['agent_from_world'][i]
        world_from_agent = new_data['world_from_agent'][i]
        
        for j in range(M):
            chosen_neigh_inds = nb_idx[j][nb_idx[j]>=0].tolist()

            # (Q. 3. 3)
            center_from_world = agent_from_world[j]
            world_from_neigh = world_from_agent[chosen_neigh_inds]
            center_from_neigh = center_from_world.unsqueeze(0) @ world_from_neigh
            
            hist_neigh_avail_b_sub = new_data["history_availabilities"][i][chosen_neigh_inds]

            hist_neigh_pos_b_sub = new_data["history_positions"][i][chosen_neigh_inds]
            hist_neigh_yaw_b_sub = new_data["history_yaws"][i][chosen_neigh_inds]

            all_other_agents_history_positions_list_sub.append(GeoUtils.transform_points_tensor(hist_neigh_pos_b_sub,center_from_neigh)*hist_neigh_avail_b_sub.unsqueeze(-1))
            all_other_agents_history_yaws_list_sub.append(hist_neigh_yaw_b_sub+new_data["yaw"][i][chosen_neigh_inds][:,None,None]-new_data["yaw"][i][j]*hist_neigh_avail_b_sub.unsqueeze(-1))
            all_other_agents_history_speeds_list_sub.append(new_data["history_speeds"][i][chosen_neigh_inds])
            all_other_agents_extent_list_sub.append(new_data["extent"][i][chosen_neigh_inds])
            all_other_agents_history_availabilities_list_sub.append(hist_neigh_avail_b_sub)
            all_other_agents_types_list_sub.append(new_data["type"][i][chosen_neigh_inds])

        all_other_agents_history_positions_list.append(pad_sequence(all_other_agents_history_positions_list_sub, batch_first=True, padding_value=np.nan))
        all_other_agents_history_yaws_list.append(pad_sequence(all_other_agents_history_yaws_list_sub, batch_first=True, padding_value=np.nan))
        all_other_agents_history_speeds_list.append(pad_sequence(all_other_agents_history_speeds_list_sub, batch_first=True, padding_value=np.nan))
        all_other_agents_extent_list.append(pad_sequence(all_other_agents_extent_list_sub, batch_first=True, padding_value=0))
        all_other_agents_types_list.append(pad_sequence(all_other_agents_types_list_sub, batch_first=True, padding_value=0))
        all_other_agents_history_availabilities_list.append(pad_sequence(all_other_agents_history_availabilities_list_sub, batch_first=True, padding_value=0))
        

    max_second_dim = max(a.size(1) for a in all_other_agents_history_positions_list)

    new_data["all_other_agents_history_positions"] = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_history_positions_list], dim=0)
    new_data["all_other_agents_history_yaws"] = angle_wrap(torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_history_yaws_list], dim=0))
    new_data["all_other_agents_history_speeds"] = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_history_speeds_list], dim=0)
    new_data["all_other_agents_extents"] = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_extent_list], dim=0)
    new_data["all_other_agents_types"] = torch.stack([torch.nn.functional.pad(tensor, (0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_types_list], dim=0)
    new_data["all_other_agents_history_availabilities"] = torch.stack([torch.nn.functional.pad(tensor, (0, 0, 0, max_second_dim - tensor.size(1), 0, 0)) for tensor in all_other_agents_history_availabilities_list], dim=0)

    new_data["scene_index"] = torch.arange(B).unsqueeze(-1).repeat(1, M)
    if merge_BM:
        for k in major_keywords+prep_keywords+["scene_index"]:
            new_data[k] = new_data[k].reshape(-1, *new_data[k].shape[2:])
        if "agent_names" in new_data:
            new_list = []

            max_len = max(len(sublist) for sublist in new_data["agent_names"])
            for sublist in new_data["agent_names"]:
                new_list.extend(sublist + [None] * (max_len - len(sublist)))

            new_data["agent_names"] = np.array(new_list)
        if "extras" in new_data:
            new_data["extras"] = {"closest_lane_point": new_data["extras"]["closest_lane_point"].reshape(-1, *new_data["extras"]["closest_lane_point"].shape[2:])}

    return new_data

def convert_scene_obs_to_agent_prep(obs):
    '''
    This will convert 9 fields (some of which only have center values) from scene-centric to agent-centric and return a new dict consisting only of the 6 fields
    '''
    B, M = obs["center_from_agents"].shape[:2]
    new_obs = {}

    agent_from_raster_list = []
    raster_from_agent_list = []
    agent_from_world_list = []
    world_from_agent_list = []
    raster_from_world_list = []
    centroid_list = []
    yaw_list = []
    map_names_list = []

    center_from_raster = obs["agent_from_raster"]
    raster_from_center = obs["raster_from_agent"]
    
    raster_cfg = BATCH_RASTER_CFG
    map_res = 1.0 / raster_cfg["pixel_size"] # convert to pixels/meter
    h = w = raster_cfg["raster_size"]
    ego_cent = raster_cfg["ego_center"]

    raster_from_agent = torch.Tensor([
            [map_res, 0, ((1.0 + ego_cent[0])/2.0) * w],
            [0, map_res, ((1.0 + ego_cent[1])/2.0) * h],
            [0, 0, 1]
    ]).to(center_from_raster.device)
    

    for i in range(B):
        center_from_agents = obs["center_from_agents"][i]
        agents_from_center = obs["agents_from_center"][i]

        center_from_world = obs["agent_from_world"][i]
        world_from_center = obs["world_from_agent"][i]

        agents_from_world = agents_from_center @ center_from_world
        world_from_agents = world_from_center @ center_from_agents

        raster_from_world = raster_from_agent @ agents_from_world

        agent_from_raster_list.append(center_from_raster.repeat(M, 1, 1))
        raster_from_agent_list.append(raster_from_center.repeat(M, 1, 1))
        agent_from_world_list.append(agents_from_world)
        world_from_agent_list.append(world_from_agents)
        raster_from_world_list.append(raster_from_world)

        centroid_list.append(GeoUtils.transform_points_tensor(obs["history_positions"][i], world_from_center)[:, -1])
        yaw_list.append(obs["history_yaws"][i, :, -1, 0] + obs["yaw"][i])

        map_names_list.append([obs['map_names'][i] for _ in range(M)])

    new_obs['agent_from_raster'] = torch.stack(agent_from_raster_list, dim=0)
    new_obs['raster_from_agent'] = torch.stack(raster_from_agent_list, dim=0)
    new_obs['agent_from_world'] = torch.stack(agent_from_world_list, dim=0)
    new_obs['world_from_agent'] = torch.stack(world_from_agent_list, dim=0)
    new_obs['raster_from_world'] = torch.stack(raster_from_world_list, dim=0)
    new_obs['centroid'] = torch.stack(centroid_list, dim=0)
    new_obs['yaw'] = torch.stack(yaw_list, dim=0)

    # for guidance loss estimation
    new_obs['drivable_map'] = obs['drivable_map']
    # for map related guidance
    new_obs['map_names'] = map_names_list

    return new_obs

def add_scene_dim_to_agent_data(obs):
    '''
    A dummy wrapper that add one dimension to each field.
    '''
    new_obs = {}
    for k in prep_keywords_for_interaction_edge+major_keywords:
        if isinstance(obs[k], torch.Tensor):
            new_obs[k] = obs[k].unsqueeze(0)
        else:
            new_obs[k] = [obs[k]]
    if 'extras' in obs:
        new_obs['extras'] = {}
        for k in obs['extras']:
            new_obs['extras'][k] = obs['extras'][k].unsqueeze(0)

    return new_obs

def load_vec_map(map_name, cache_path="~/.unified_data_cache"):
    cache_path = Path(cache_path).expanduser()
    mapAPI = MapAPI(cache_path)
    vec_map = mapAPI.get_map(map_name, scene_cache=None)
    return vec_map

def extract_data_batch_for_guidance(data_batch, mode='testing'):
    data_batch_for_guidance = {}
    if mode == 'testing':
        guidance_keywords_used = guidance_keywords
    elif mode == 'training':
        guidance_keywords_used = guidance_keywords_training
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    for k in guidance_keywords_used:
        if k == 'extras':
            if 'extras' in data_batch:
                data_batch_for_guidance['extras'] = {}
                for j in data_batch['extras']:
                    B, M = data_batch['extras'][j].shape[:2]
                    data_batch_for_guidance['extras'][j] = data_batch['extras'][j].reshape(B*M, *data_batch['extras'][j].shape[2:])
        elif k == 'map_names':
            if isinstance(data_batch[k][0], list):
                map_name = data_batch[k][0][0]
            else:
                map_name = data_batch[k][0]
            vec_map = load_vec_map(map_name)
            data_batch_for_guidance['vec_map'] = vec_map
        else:
            if k == 'scene_index' and k not in data_batch and 'scene_ids' in data_batch:
                k2 = 'scene_ids'
                data_batch_for_guidance[k] = data_batch[k2]
            else:
                k2 = k
                B, M = data_batch[k2].shape[:2]
                data_batch_for_guidance[k] = data_batch[k2].reshape(B*M, *data_batch[k2].shape[2:])

    return data_batch_for_guidance

def check_consistency(keywords, agent_centric_obs, agent_obs_gt, agent_obs_in_scene_format=False):
    torch.set_printoptions(precision=6)
    for k in keywords:
        if k in agent_centric_obs and k in agent_obs_gt:
            assert agent_centric_obs[k].shape == agent_obs_gt[k].shape, f'{k} {agent_centric_obs[k].shape} {agent_obs_gt[k].shape}'
            op1 = torch.round(agent_centric_obs[k].to(torch.float), decimals=20)
            op2 = torch.round(agent_obs_gt[k].to(torch.float), decimals=20)
            cond = torch.abs(op1 - op2) > 1e-2

            # if k == 'all_other_agents_future_availability':
            #     assert not torch.any(cond), f'{agent_centric_obs[k][torch.where(cond)]} {agent_obs_gt[k][torch.where(cond)]} {agent_centric_obs[k][1,5]} {agent_obs_gt[k][1,5]}'
            
            if k == 'maps':
                # M, C, H, W = op1.shape
                # op1 = agent_centric_obs[k].reshape(-1, H, W)
                # op2 = agent_obs_gt[k].reshape(-1, H, W)
                # cond = torch.abs(op1 - op2) > 1e-2
                # cond_viol = torch.mean(cond.to(float), dim=(1,2))

                # ind = torch.argsort(cond_viol)[-1]
                # print(ind.shape)
                # op1 = op1[ind]
                # op2 = op2[ind]
                # cond = torch.abs(op1 - op2) > 1e-2
                # assert not torch.any(cond), f'{k} {cond_viol[ind]} {ind} {torch.where(cond)} {op1.shape} {op1} {op2.shape} {op2}'

                op1 = agent_centric_obs[k][4, 0]
                op2 = agent_obs_gt[k][4, 0]
                cond = torch.abs(op1 - op2) > 1e-2
                assert not torch.any(cond), f'{k} {torch.mean(cond.to(float))} {torch.where(cond)} {op1.shape} {op1} {op2.shape} {op2}'

            assert torch.mean(cond.to(torch.float))<1e-3, f'{k} {torch.mean(cond.to(torch.float))} {torch.where(cond)} {agent_centric_obs[k].shape} {agent_centric_obs[k]} {agent_obs_gt[k].shape} {agent_obs_gt[k]}'
            # assert not torch.any(cond), f'{k} {torch.mean(cond.to(torch.float))} {torch.where(cond)} {agent_centric_obs[k].shape} {agent_centric_obs[k]} {agent_obs_gt[k].shape} {agent_obs_gt[k]}'
        else:
            print(k, 'is not checked')
    if 'extras' in agent_centric_obs and 'extras' in agent_obs_gt:
        
        op1 = agent_centric_obs['extras']['closest_lane_point'][1, 0, 0]
        op2 = agent_obs_gt['extras']['closest_lane_point'][1, 0, 0]
        cond = torch.abs(op1 - op2) > 1e-2
        assert torch.mean(cond.to(torch.float))<1e-3, f'closest_lane_point {torch.mean(cond.to(torch.float))} {torch.where(cond)} {op1.shape} {op1} {op2.shape} {op2}'

        
        
        assert agent_centric_obs['extras']['closest_lane_point'].shape == agent_obs_gt['extras']['closest_lane_point'].shape, f'closest_lane_point {agent_centric_obs["extras"]["closest_lane_point"].shape} {agent_obs_gt["extras"]["closest_lane_point"].shape}'
    
        cond = agent_centric_obs['extras']['closest_lane_point'] - agent_obs_gt['extras']['closest_lane_point'] > 1e-2
        assert torch.mean(cond.to(torch.float))<1e-3, f'closest_lane_point {torch.mean(cond.to(torch.float))} {torch.where(cond)} {agent_centric_obs["extras"]["closest_lane_point"].shape} {agent_centric_obs["extras"]["closest_lane_point"]} {agent_obs_gt["extras"]["closest_lane_point"].shape} {agent_obs_gt["extras"]["closest_lane_point"]}'
        # assert not torch.any(cond), f'closest_lane_point {torch.mean(cond.to(torch.float))} {torch.where(cond)} {agent_centric_obs["extras"]["closest_lane_point"]} {agent_obs_gt["extras"]["closest_lane_point"]}'

def check_action_consistency(action, action_gt):
    action = {"positions": action.positions, "yaws": action.yaws}
    action_gt = {"positions": action_gt.positions, "yaws": action_gt.yaws}
    for k in ["positions", "yaws"]:
        op1 = action[k]
        op2 = action_gt[k]
        cond = torch.abs(op1 - op2) > 1e-3
        assert torch.mean(cond.to(torch.float))<1e-3, f'{k} {torch.mean(cond.to(torch.float))} {torch.where(cond)} {op1.shape} {op1} {op2.shape} {op2}'



# ------------- visualization (modified from trajdata for visualizing parsed data) ----------------
def convert_lyft_type_to_nusc_type(agent_type):
    agent_type[agent_type == 0] = 0 # unknown
    agent_type[agent_type == 3] = 1 # vehicle
    agent_type[agent_type == 14] = 2 # pedestrian
    agent_type[agent_type == 10] = 3 # bicycle
    agent_type[agent_type == 11] = 4 # motorcycle
    return agent_type
def convert_nusc_type_to_lyft_type(agent_type):
    agent_type[agent_type < 0] = 0 # unknown
    agent_type[agent_type == 3] = 10 # bicycle
    agent_type[agent_type == 1] = 3 # vehicle
    agent_type[agent_type == 2] = 14 # pedestrian
    agent_type[agent_type == 4] = 11 # motorcycle
    return agent_type

def plot_vec_map_lanes(ax, batch, batch_idx, lane_segs=50):
    lanes = batch['extras']["closest_lane_point"][batch_idx]
    for i, lane_points in enumerate(lanes[:lane_segs]):
        lane_points = lane_points[
            torch.logical_not(torch.any(torch.isnan(lane_points), dim=1)), :
        ].numpy()

        ax.plot(
            lane_points[:, 0],
            lane_points[:, 1],
            "o-",
            markersize=3,
            label="Lane points"+str(i),
        )

def plot_agent_batch_dict(
    batch_original: dict,
    batch_idx: int,
    ax: Optional[Axes] = None,
    legend: bool = True,
    show: bool = True,
    close: bool = True,
) -> None:
    # In order to visualize during rollout
    if len(batch_original['history_positions'].shape) == 4:
        batch = {}
        for k in batch_original:
            if k == 'extras':
                batch['extras'] = {}
                for k2 in batch_original['extras']:
                    batch['extras'][k2] = batch_original['extras'][k2][0].detach().cpu()
            elif k in ['agent_name', 'map_names']:
                batch[k] = batch_original[k][0]
            else:
                batch[k] = batch_original[k][0].detach().cpu()
    else:
        batch = batch_original

    if ax is None:
        _, ax = plt.subplots()

    if 'agent_name' in batch:
        agent_name: str = batch['agent_name'][batch_idx]
    elif 'agent_names' in batch:
        agent_name: str = batch['agent_names'][batch_idx]
    else:
        raise ValueError("No agent name found in batch")
    
    agent_type = batch['type']
    agent_type = convert_lyft_type_to_nusc_type(agent_type)
    agent_type: AgentType = AgentType(agent_type[batch_idx].item())

    current_state = np.concatenate([batch['centroid'][batch_idx].numpy(), batch['yaw'][:, None][batch_idx].numpy()], axis=0)
    ax.set_title(
        f"{str(agent_type)}/{agent_name}\nat x={current_state[0]:.2f},y={current_state[1]:.2f},h={current_state[-1]:.2f}"
    )

    agent_from_world_tf: Tensor = batch['agent_from_world'][batch_idx].cpu()

    if batch['maps'] is not None:
        world_from_raster_tf: Tensor = torch.linalg.inv(
            batch['raster_from_world'][batch_idx].cpu()
        )

        agent_from_raster_tf: Tensor = agent_from_world_tf @ world_from_raster_tf

        draw_map(ax, batch['maps'][batch_idx], agent_from_raster_tf, alpha=1.0)

    agent_hist = StateTensor.from_array(torch.cat([batch['history_positions'][batch_idx].cpu(), batch['history_yaws'][batch_idx].cpu()], axis=-1), format='x,y,h')

    agent_fut = StateTensor.from_array(torch.cat([batch['target_positions'][batch_idx].cpu(), batch['target_yaws'][batch_idx].cpu()], axis=-1), format='x,y,h')

    agent_extent = batch['extent'][batch_idx].cpu()
    base_frame_from_agent_tf = torch.eye(3)

    palette = sns.color_palette("husl", 4)
    if agent_type == AgentType.VEHICLE:
        color = palette[0]
    elif agent_type == AgentType.PEDESTRIAN:
        color = palette[1]
    elif agent_type == AgentType.BICYCLE:
        color = palette[2]
    else:
        color = palette[3]

    draw_history(
        ax,
        agent_type,
        agent_hist[:-1],
        agent_extent,
        base_frame_from_agent_tf,
        facecolor=color,
        edgecolor=None,
        linewidth=0,
    )
    ax.plot(
        agent_hist.get_attr("x"),
        agent_hist.get_attr("y"),
        linestyle="--",
        color=color,
        label="Agent History",
    )
    draw_agent(
        ax,
        agent_type,
        agent_hist[-1],
        agent_extent,
        base_frame_from_agent_tf,
        facecolor=color,
        edgecolor="k",
        label="Agent Current",
    )
    ax.plot(
        agent_fut.get_attr("x"),
        agent_fut.get_attr("y"),
        linestyle="-",
        color=color,
        label="Agent Future",
    )

    if 'extras' in batch:
        plot_vec_map_lanes(ax, batch, batch_idx, lane_segs=50)

    if 'num_neigh' in batch:
        num_neigh = batch['num_neigh'][batch_idx]
    else:
        num_neigh = batch['all_other_agents_history_positions'][batch_idx].shape[0]
    if num_neigh > 0:
        neighbor_hist = StateTensor.from_array(torch.cat([batch['all_other_agents_history_positions'][batch_idx].cpu(), batch['all_other_agents_history_yaws'][batch_idx].cpu()], axis=-1), format='x,y,h')
        # neighbor_fut = StateTensor.from_array(torch.cat([batch['all_other_agents_future_positions'][batch_idx].cpu(), batch['all_other_agents_future_yaws'][batch_idx].cpu()], axis=-1), format='x,y,h')
        neighbor_extent = batch['all_other_agents_extents'][batch_idx].cpu()
        
        neighbor_type = batch['all_other_agents_types']
        
        
        neighbor_type = convert_lyft_type_to_nusc_type(neighbor_type)
        neighbor_type = neighbor_type[batch_idx].cpu()

        ax.plot([], [], c="olive", ls="--", label="Neighbor History")
        ax.plot([], [], c="darkgreen", label="Neighbor Future")

        for n in range(num_neigh):
            if torch.isnan(neighbor_hist[n, -1, :]).any():
                # this neighbor does not exist at the current timestep
                continue
            ax.plot(
                neighbor_hist.get_attr("x")[n, :],
                neighbor_hist.get_attr("y")[n, :],
                c="olive",
                ls="--",
            )
            draw_agent(
                ax,
                neighbor_type[n],
                neighbor_hist[n, -1],
                neighbor_extent[n, :],
                base_frame_from_agent_tf,
                facecolor="olive",
                edgecolor="k",
                alpha=0.7,
            )
            # ax.plot(
            #     neighbor_fut.get_attr("x")[n, :],
            #     neighbor_fut.get_attr("y")[n, :],
            #     c="darkgreen",
            # )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")

    # Doing this because the imshow above makes the map origin at the top.
    # TODO(pkarkus) we should just modify imshow not to change the origin instead.
    ax.invert_yaxis()

    if legend:
        ax.legend(loc="best", frameon=True)

    if show:
        plt.show()

    if close:
        plt.close()

    return ax