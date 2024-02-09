import numpy as np
import torch

from tbsim.envs.base import BatchedEnv
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers

from tbsim.utils.geometry_utils import batch_nd_transform_points_np
from tbsim.utils.guidance_metrics import guidance_metrics_from_config, constraint_metrics_from_config
from tbsim.utils.trajdata_utils import get_closest_lane_point_wrapper

from trajdata.utils.arr_utils import transform_xyh_np, transform_coords_np
import matplotlib.colors as mcolors

def guided_rollout(
    env,
    policy,
    policy_model,
    n_step_action=1,
    guidance_config=None,
    constraint_config=None,
    render=False,
    scene_indices=None,
    device=None,
    obs_to_torch=True,
    horizon=None,
    start_frames=None,
    eval_class='Diffuser',
    apply_guidance=True,
    apply_constraints=True,
):
    """
    Rollout an environment.
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy (RolloutWrapper): a policy that controls agents in the environment
        policy_model (LightningModule): the traffic model underlying the policy with set_guidance and set_constraints implemented.
        n_step_action (int): number of steps to take between querying models
        guidance_config: which guidance functions to use
        constriant_config: parameters of the constraints to use in each scene
        render (bool): if True, return a sequence of rendered frames
        scene_indices (tuple, list): (Optional) scenes indices to rollout with
        device: device to cast observation to
        obs_to_torch: whether to cast observation to torch
        horizon (int): (Optional) override horizon of the simulation
        start_frames (list) : (Optional) a list of starting frame index for each scene index.
        eval_class: (str) which class to use for evaluation
        apply_guidance: (bool) whether to apply guidance or not
        apply_constraints: (bool) whether to apply constraints or not

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
        renderings (list): A list of rendered frames in the form of np.ndarray, one for each episode
    """
    stats = {}
    info = {}
    renderings = []
    is_batched_env = isinstance(env, BatchedEnv)
    timers = Timers()

    # set up guidance and constraints, and associated metrics
    added_metrics = [] # save for removal later

    if len(guidance_config) > 0:
        # only set guidance for those support inner perturbation; filtration is currently considered as separate outside wrapper
        if apply_guidance and eval_class in ['SceneDiffuser', 'Diffuser', 'TrafficSim', 'BC', 'HierarchicalSampleNew']:
            # reset so that we can get an example batch to initialize guidance more efficiently
            env.reset(scene_indices=scene_indices, start_frame_index=start_frames)
            ex_obs = env.get_observation()
            if obs_to_torch:
                device = policy.device if device is None else device
                ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)
            policy_model.set_guidance(guidance_config, ex_obs['agents'])

        # TBD: extract stationary_mask and use it in metrics
        guidance_metrics = guidance_metrics_from_config(guidance_config)
        env._metrics.update(guidance_metrics)  
        added_metrics += guidance_metrics.keys()
    if len(constraint_config) > 0:
        # TODO : right now constraint metric only makes sense in open-loop setting
        if apply_constraints and eval_class in ['SceneDiffuser', 'Diffuser']:
            policy_model.set_constraints(constraint_config)
        constraint_metrics = constraint_metrics_from_config(constraint_config)
        env._metrics.update(constraint_metrics)  
        added_metrics += constraint_metrics.keys()

    # metrics are reset here too, so have to run again after adding new metrics, so have to run again after adding new metrics
    env.reset(scene_indices=scene_indices, start_frame_index=start_frames)

    done = env.is_done()
    counter = 0
    step_since_last_update = 0
    frames = []  
    while not done:
        timers.tic("step")
        with timers.timed("obs"):
            obs = env.get_observation()
        with timers.timed("to_torch"):
            if obs_to_torch:
                device = policy.device if device is None else device
                obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
            else:
                obs_torch = obs
        with timers.timed("network"):
            action = policy.get_action(obs_torch, step_index=counter)

        with timers.timed("env_step"):
            ims = env.step(
                action, num_steps_to_take=n_step_action, render=render
            )  # List of [num_scene, h, w, 3]
        if render:
            frames.extend(ims)
        counter += n_step_action
        step_since_last_update += n_step_action
        timers.toc("step")
        print(timers)
        print('counter', counter)

        done = env.is_done()

        if horizon is not None and counter >= horizon:
            break

    metrics = env.get_metrics()

    for k, v in metrics.items():
        if k not in stats:
            stats[k] = []
        if is_batched_env:  # concatenate by scene
            stats[k] = np.concatenate([stats[k], v], axis=0)
        else:
            stats[k].append(v)

    # remove all temporary added metrics
    for met_name in added_metrics:
        env._metrics.pop(met_name)
    # and undo guidance setting
    if policy_model is not None:
        policy_model.clear_guidance()

    env_info = env.get_info()
    for k, v in env_info.items():
        if k not in info:
            info[k] = []
        if is_batched_env:
            info[k].extend(v)
        else:
            info[k].append(v)

    if render:
        frames = np.stack(frames)
        if is_batched_env:
            # [step, scene] -> [scene, step]
            frames = frames.transpose((1, 0, 2, 3, 4))
        renderings.append(frames)

    env.reset_multi_episodes_metrics()

    return stats, info, renderings

################## HEURISTIC CONFIG UTILS #########################

from copy import deepcopy

def merge_guidance_configs(cfg1, cfg2):
    if cfg1 is None or len(cfg1) == 0:
        return cfg2
    if cfg2 is None or len(cfg2) == 0:
        return cfg1
    merge_cfg = deepcopy(cfg1)
    num_scenes = len(merge_cfg)
    for si in range(num_scenes):
        merge_cfg[si].extend(cfg2[si])
    return merge_cfg


def get_agents_future(sim_scene, fut_sec):
    '''
    Queries the sim scene for the future state traj (in global frame) of all agents.
    - sim_scene : to query
    - fut_sec : how far in the future (in sec) to get

    Returns:
    - agents_future: (N x T x 8) [pos, vel, acc, heading_angle]
    - fut_valid : (N x T) whether states are non-nan at each step
    '''
    agents_future, _, _ = sim_scene.cache.get_agents_future(sim_scene.init_scene_ts, sim_scene.agents, (fut_sec, fut_sec))

    # agents_future = np.stack(agents_future, axis=0)
    agents_future_shapes = np.array([af.shape for af in agents_future])
    row_max = np.max(agents_future_shapes[:, 0])
    # pad nan to match row length
    agents_future = np.stack([np.concatenate([af, np.nan*np.zeros((row_max-af.shape[0], af.shape[1]))]) for af in agents_future], axis=0)
    # TBD: fix this hack 
    from trajdata.caching.df_cache import STATE_FORMAT_STR
    fut_valid = np.sum(np.logical_not(np.isnan(agents_future)), axis=-1) == len(STATE_FORMAT_STR.split(','))
    return agents_future, fut_valid

def get_agents_curr(sim_scene):
    '''
    Queries the sim scene for current state of all agents

    returns:
    - curr_agent_state: (N x 8) [pos, vel, acc, heading_angle]
    '''
    curr_agent_state = sim_scene.cache.get_states(
            [sim_scene.agents[i].name for i in range(len(sim_scene.agents))],
            sim_scene.init_scene_ts
    )
    curr_agent_state = np.stack(curr_agent_state, axis=0)
    return curr_agent_state

def get_agent_from_world_tf(curr_state):
    pos = curr_state[:,:2]
    h = curr_state[:,-1:]
    hx, hy = np.cos(h), np.sin(h)

    last_row = np.zeros((pos.shape[0], 3))
    last_row[:,2] = 1.0
    world_from_agent_tf = np.stack([
                                np.concatenate([hx, -hy, pos[:,0:1]], axis=-1),
                                np.concatenate([hy,  hx, pos[:,1:2]], axis=-1),
                                last_row,
                                ], axis=-2)
    agent_from_world_tf = np.linalg.inv(world_from_agent_tf)

    return agent_from_world_tf

def heuristic_social_group(sim_scene, dt, group_dist_thresh, social_dist, cohesion, **kwargs):
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import csgraph_from_dense, connected_components
    import random

    curr_state = get_agents_curr(sim_scene)
    cur_pos = curr_state[:,:2]
    cur_vel = curr_state[:,2:4]

    # create graph with edges based on given distance threshold and direction of movement.
    #   create social groups from connected components
    not_moving = np.linalg.norm(cur_vel, axis=-1) < 0.9
    dir = cur_vel / (np.linalg.norm(cur_vel, axis=-1, keepdims=True) + 1e-6)
    cos_sim = np.sum(dir[:,np.newaxis] * dir[np.newaxis,:], axis=-1)
    move_sim = cos_sim >= 0
    move_sim[not_moving] = True # if they're not moving, don't care about direction
    move_sim[:,not_moving] = True
    # now distance
    dist = squareform(pdist(cur_pos))
    graph = np.logical_and(dist <= group_dist_thresh, move_sim)
    np.fill_diagonal(graph, 0)
    graph = graph.astype(int)
    graph = csgraph_from_dense(graph)

    n_comp, labels = connected_components(graph, directed=False)
    config_list = []
    for ci in range(n_comp):
        comp_mask = labels == ci
        comp_size = np.sum(comp_mask)
        # only want groups, not single agents
        if comp_size > 1:
            group_inds = np.nonzero(comp_mask)[0].tolist()
            # randomly sample leader
            leader = random.sample(group_inds, 1)[0]
            guide_config = {
                'name' : 'social_group',
                'params' : {
                            'leader_idx' : leader,
                            'social_dist' : social_dist,
                            'cohesion' : cohesion,    
                           },
                'agents' : group_inds
            }
            config_list.append(guide_config)

    if len(config_list) > 0:
        return config_list
    return None


def heuristic_global_target_pos_at_time(sim_scene, dt, target_time, urgency, pref_speed, target_tolerance, action_num, perturb_std=None, **kwargs):
    '''
    Sets a global target pos and time using GT.
    '''
    fut_sec = target_time * dt
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_pos = fut_traj[:,:,:2]
    agents = np.arange(fut_pos.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    if np.sum(valid_agts) == 0:
        return None
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]

    # take closest time to target we can get
    N, T = fut_valid.shape
    last_valid_t = np.amax(np.repeat(np.arange(T)[np.newaxis], N, axis=0) * fut_valid, axis=-1)
    target_pos = fut_pos[np.arange(N), last_valid_t]
    
    # set pref_speed based on avg GT speed
    if pref_speed is None:
        fut_vel = fut_traj[:,:,3:5]
        speed = np.linalg.norm(fut_vel, axis=-1)
        # print('last_valid_t', last_valid_t)
        # print('speed', speed)
        pref_speed = np.nansum(speed * fut_valid, axis=-1) / (last_valid_t + 1)
        # print('pref_speed_speed.shape', pref_speed_speed.shape)
        # print('pref_speed', pref_speed)
        pref_speed = pref_speed
    else:
        pref_speed = [pref_speed]*N

    # add noise if desired
    if perturb_std is not None and perturb_std > 0.0:
        target_pos = target_pos + np.random.randn(*(target_pos.shape))*perturb_std

    guide_config = {
        'name' : 'global_target_pos_at_time',
        'params' : {
                    'target_pos' : target_pos.tolist(),
                    'target_time' : last_valid_t.tolist(),
                    'urgency' : [urgency]*N,    
                    'pref_speed' : pref_speed,
                    'dt' : dt,

                    'target_tolerance': target_tolerance,
                    'action_num': action_num,
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_global_target_pos(sim_scene, dt, target_time, urgency, pref_speed, min_progress_dist, target_tolerance, action_num, perturb_std=None, **kwargs):
    '''
    Sets a global target pos using GT.
    '''
    guide_config = heuristic_global_target_pos_at_time(sim_scene, dt, target_time, urgency, pref_speed, target_tolerance, action_num, perturb_std, **kwargs)
    guide_config['name'] = 'global_target_pos'
    guide_config['params']['min_progress_dist'] = min_progress_dist
    guide_config['params'].pop('target_time', None)
    return guide_config

def heuristic_target_pos_at_time(sim_scene, dt, target_time, perturb_std=None, **kwargs):
    '''
    Sets a local target pos and time using GT.
    '''
    fut_sec = target_time * dt
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_pos = fut_traj[:,:,:2]
    agents = np.arange(fut_pos.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    if np.sum(valid_agts) == 0:
        return None
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]

    # take closest time to target we can get
    N, T = fut_valid.shape
    last_valid_t = np.amax(np.repeat(np.arange(T)[np.newaxis], N, axis=0) * fut_valid, axis=-1)
    target_pos = fut_pos[np.arange(N), last_valid_t]

    # add noise if desired
    if perturb_std is not None and perturb_std > 0.0:
        target_pos = target_pos + np.random.randn(*(target_pos.shape))*perturb_std

    # convert to local frame
    curr_state = get_agents_curr(sim_scene)[valid_agts]
    agt_from_world_tf = get_agent_from_world_tf(curr_state)
    target_pos = batch_nd_transform_points_np(target_pos, agt_from_world_tf)

    guide_config = {
        'name' : 'target_pos_at_time',
        'params' : {
                    'target_pos' : target_pos.tolist(),
                    'target_time' : last_valid_t.tolist(),
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_target_pos(sim_scene, dt, target_time, perturb_std=None, **kwargs):
    '''
    Sets a target pos using GT.
    '''
    guide_config = heuristic_target_pos_at_time(sim_scene, dt, target_time, perturb_std, **kwargs)
    guide_config['name'] = 'target_pos'
    guide_config['params'].pop('target_time', None)
    return guide_config

def heuristic_agent_collision(sim_scene, dt, num_disks, buffer_dist, decay_rate, excluded_agents, **kwargs):
    '''
    Applies collision loss to all agents.
    '''
    guide_config = {
        'name' : 'agent_collision',
        'params' : {
                    'num_disks' : num_disks,
                    'buffer_dist' : buffer_dist,
                    'decay_rate': decay_rate,
                    'excluded_agents': excluded_agents,
                    },
        'agents' : None, # all agents
    }
    return guide_config

def heuristic_map_collision(sim_scene, dt, num_points_lw, decay_rate, **kwargs):
    '''
    Applies collision loss to all agents.
    '''
    guide_config = {
        'name' : 'map_collision',
        'params' : {
                    'num_points_lw' : num_points_lw,
                    'decay_rate': decay_rate,
                    },
        'agents' : None, # all agents
    }
    return guide_config

# =============================================================================
def heuristic_global_stop_sign(sim_scene, dt, target_time, stop_box_dim, scale, horizon_length, num_time_steps_to_stop, action_num, low_speed_th, **kwargs):
    '''
    Sets a global stop sign using GT.
    scale: stlcg loss smoothing coefficient
    horizon_length: prediction horizon (in time steps)
    '''
    fut_sec = target_time * dt
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_pos = fut_traj[:,:,:2]
    agents = np.arange(fut_pos.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]
    
    # # TBD: hacky temporary for one particular drivesim scene
    # agents = np.array([0,1,2])
    # fut_valid = fut_valid[agents]

    N = len(agents)
    T = fut_valid.shape[1]
    # take closest time to target we can get
    last_valid_t = np.amax(np.repeat(np.arange(T)[np.newaxis], N, axis=0) * fut_valid, axis=-1)
    stop_sign_pos = fut_pos[np.arange(N), last_valid_t]

    guide_config = {
        'name' : 'global_stop_sign',
        'params' : {
                    'stop_sign_pos' : stop_sign_pos.tolist(),
                    # 'target_time' : last_valid_t.tolist(),

                    'stop_box_dim': [stop_box_dim] * N,
                    'scale': scale,

                    'horizon_length': horizon_length,
                    'time_step_to_start': 0,
                    'num_time_steps_to_stop': num_time_steps_to_stop,

                    'action_num': action_num,
                    'low_speed_th': low_speed_th,
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_stop_sign(sim_scene, dt, target_time, stop_box_dim, scale, horizon_length, num_time_steps_to_stop, action_num, low_speed_th, **kwargs):
    '''
    Sets a local stop sign using GT.
    '''
    fut_sec = target_time * dt
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_pos = fut_traj[:,:,:2]
    agents = np.arange(fut_pos.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]

    # take closest time to target we can get
    N, T = fut_valid.shape
    last_valid_t = np.amax(np.repeat(np.arange(T)[np.newaxis], N, axis=0) * fut_valid, axis=-1)
    stop_sign_pos = fut_pos[np.arange(N), last_valid_t]

    # convert to local frame
    curr_state = get_agents_curr(sim_scene)
    agt_from_world_tf = get_agent_from_world_tf(curr_state)
    stop_sign_pos = batch_nd_transform_points_np(stop_sign_pos, agt_from_world_tf)

    guide_config = {
        'name' : 'stop_sign',
        'params' : {
                    'stop_sign_pos' : stop_sign_pos.tolist(),
                    # 'target_time' : last_valid_t.tolist(),

                    'stop_box_dim': [stop_box_dim] * N,
                    'scale': scale,

                    'horizon_length': horizon_length,
                    'time_step_to_start': 0,
                    'num_time_steps_to_stop': num_time_steps_to_stop,

                    'action_num': action_num,
                    'low_speed_th': low_speed_th,
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_speed_limit(sim_scene, dt, speed_limit_quantile, low_speed_th, fut_sec, **kwargs):
    '''
    Sets a global stop sign using GT.
    '''
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_vel = fut_traj[:,:,3:5]
    agents = np.arange(fut_vel.shape[0])

    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    if np.sum(valid_agts) < fut_vel.shape[0]:
        fut_vel = fut_vel[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]
    # print('fut_valid', fut_valid.shape, fut_valid)
    # print('valid_agts', valid_agts.shape, valid_agts)
    # take closest time to target we can get
    speed = np.linalg.norm(fut_vel, axis=-1)

    speed_valid = speed*fut_valid
    # print('speed_valid.shape', speed_valid.shape, speed_valid)
    speed_limit = np.nanquantile(speed_valid[speed_valid>low_speed_th], speed_limit_quantile)
    if np.isnan(speed_limit):
        speed_limit = low_speed_th 
    # print('speed_limit', speed_limit)

    guide_config = {
        'name' : 'speed_limit',
        'params' : {
                    'speed_limit': speed_limit,
                   },
        'agents' : agents.tolist()
    }
    return guide_config


def heuristic_target_speed(sim_scene, dt, target_speed_multiplier, fut_sec, **kwargs):
    '''
    Sets a target speed using GT.

    -target_speed_multiplier: multiplier for the speed of the agent
    '''
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    fut_vel = fut_traj[:,:,3:5]
    agents = np.arange(fut_vel.shape[0])


    valid_agts = np.sum(fut_valid, axis=-1) > 0 # agents that show up at some point
    if np.sum(valid_agts) < fut_vel.shape[0]:
        fut_vel = fut_vel[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agents = agents[valid_agts]

    speed = np.linalg.norm(fut_vel, axis=-1)
    target_speed = speed * target_speed_multiplier

    guide_config = {
        'name' : 'target_speed',
        'params' : {
                    'target_speed': target_speed,
                    'fut_valid': fut_valid,
                    'dt': dt,
                   },
        'agents' : agents.tolist()
    }
    return guide_config

def heuristic_gptcollision(sim_scene, dt, collision_radius, **kwargs):
    '''
    Increase collision loss for a pair of agents.
    '''
    example_batch = kwargs['example_batch']
    agent_from_world = example_batch['agent_from_world'].cpu().numpy()

    fut_sec = 10
    min_current_speed = 2.0 # don't consider vehicles that are too slow
    angle_diff_max_th = 0.4*np.pi # don't consider vehicles that drive along different directions
    dist_diff_max_th = 30 # don't consider vehicles that are too far away
    dist_diff_min_th = 10 # don't consider vehicles that are too close
    sort_mode = 'distance' # 'distance': sort by distance, None: no sorting

    # (x,y,z,vx,vy,ax,ay,yaw)
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    # (M, T, 2)
    # print('fut_traj[:, 20, -1]', fut_traj[:, 20, -1])
    
    fut_pos = fut_traj[...,:2]
    fut_speed = np.linalg.norm(fut_traj[...,2:4], axis=-1)
    fut_yaw = fut_traj[...,-1]
    
    valid_presence = fut_valid[:,0] > 0 # agents that are present at the current timestep
    valid_speed = np.abs(fut_speed[:,0]) > min_current_speed # agents that have non-zero speed right now
    valid_agts = valid_presence * valid_speed
    # print('np.sum(valid_presence)', np.sum(valid_presence))
    # print('np.amax(fut_speed, axis=-1)', np.nanmax(np.abs(fut_speed), axis=-1))
    # print('np.sum(valid_speed)', np.sum(valid_speed))
    # print('0 np.sum(valid_agts)', np.sum(valid_agts), valid_agts)
    if np.sum(valid_agts) == 0:
        return None
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_speed = fut_speed[valid_agts]
        fut_yaw = fut_yaw[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agent_from_world = agent_from_world[valid_agts]
        map_new_ind_to_initial_ind = np.arange(valid_agts.shape[0])[valid_agts]
    else:
        map_new_ind_to_initial_ind = np.arange(valid_agts.shape[0])

    # Pick the pair with proper relative distance and angle
    yaw_diff = np.abs(np.expand_dims(fut_yaw, axis=1) - np.expand_dims(fut_yaw, axis=0))  # Shape: (M, M, T)
    pos_diff = np.linalg.norm(np.expand_dims(fut_pos, axis=1) - np.expand_dims(fut_pos, axis=0), axis=-1)  # Shape: (M, M, T)
    # print('yaw_diff[:,:,0]', yaw_diff[:,:,0].shape, yaw_diff[:,:,0])
    # print('yaw_diff[:,:,20]', yaw_diff[:,:,20].shape, yaw_diff[:,:,20])
    yaw_max_filter = (yaw_diff[:,:,0] < angle_diff_max_th) * (yaw_diff[:,:,20] < angle_diff_max_th)
    pos_max_filter = (pos_diff[:,:,0] < dist_diff_max_th) * (pos_diff[:,:,20] < dist_diff_max_th)
    pos_min_filter = (pos_diff[:,:,0] > dist_diff_min_th) * (pos_diff[:,:,20] > dist_diff_min_th)
    valid_agts = yaw_max_filter * pos_max_filter * pos_min_filter

    # print('np.sum(yaw_max_filter)', np.sum(yaw_max_filter))
    # print('np.sum(pos_max_filter)', np.sum(pos_max_filter))
    # print('np.sum(pos_min_filter)', np.sum(pos_min_filter))
    # print('1 np.sum(valid_agts)', np.sum(valid_agts), valid_agts)
    
    if np.sum(valid_agts) == 0:
        return None
    else:
        indices = np.argwhere(valid_agts)
        
        # sort indices by distance
        if sort_mode == 'distance':
            inds_sorted = np.argsort(pos_diff[:,:,20][valid_agts])
            indices = indices[inds_sorted]
        elif sort_mode is None:
            pass
        else:
            raise NotImplementedError
        
        # print('indices', indices.shape, indices)
        ind1, ind2 = indices[0]
        fut_pos_ind1_frame = transform_coords_np(fut_pos, agent_from_world[ind1])
        ind1_original = map_new_ind_to_initial_ind[ind1]
        ind2_original = map_new_ind_to_initial_ind[ind2]
        
        # ind2 is ahead
        if fut_pos_ind1_frame[ind2, 0, 0] > 0:
            target_ind = ind1_original
            ref_ind = ind2_original
        else:
            target_ind = ind2_original
            ref_ind = ind1_original
        print('target_ind', target_ind)
        print('ref_ind', ref_ind)

    guide_config = {
        'name' : 'gptcollision',
        'params' : {
                    'target_ind' : target_ind,
                    'ref_ind' : ref_ind,
                    'collision_radius' : 1.0,
                   },
        'agents' : None, # all agents
    }
    return guide_config

def heuristic_gptkeepdistance(sim_scene, dt, min_distance, max_distance, **kwargs):
    '''
    Applies keepdistance loss to a pair of agents.
    '''
    example_batch = kwargs['example_batch']
    agent_from_world = example_batch['agent_from_world'].cpu().numpy()

    fut_sec = 10
    min_current_speed = 2.0 # don't consider vehicles that are too slow
    angle_diff_max_th = 0.2*np.pi # don't consider vehicles that drive along different directions
    dist_diff_max_th = 30 # don't consider vehicles that are too far away
    dist_diff_min_th = 10 # don't consider vehicles that are too close
    sort_mode = None # 'distance': sort by distance, None: no sorting

    # (x,y,z,vx,vy,ax,ay,yaw)
    fut_traj, fut_valid = get_agents_future(sim_scene, fut_sec)
    # (M, T, 2)
    # print('fut_traj[:, 20, -1]', fut_traj[:, 20, -1])
    # print('fut_traj[:3, 20, :]', fut_traj[:3, 20, :])
    fut_pos = fut_traj[...,:2]
    fut_speed = np.linalg.norm(fut_traj[...,2:4], axis=-1)
    fut_yaw = fut_traj[...,-1]
    
    valid_presence = fut_valid[:,0] > 0 # agents that are present at the current timestep
    valid_speed = np.abs(fut_speed[:,0]) > min_current_speed # agents that have non-zero speed right now
    valid_agts = valid_presence * valid_speed
    # print('np.sum(valid_presence)', np.sum(valid_presence))
    # print('np.amax(fut_speed, axis=-1)', np.nanmax(np.abs(fut_speed), axis=-1))
    # print('np.sum(valid_speed)', np.sum(valid_speed))
    # print('0 np.sum(valid_agts)', np.sum(valid_agts), valid_agts)
    if np.sum(valid_agts) == 0:
        return None
    if np.sum(valid_agts) < fut_pos.shape[0]:
        fut_pos = fut_pos[valid_agts]
        fut_speed = fut_speed[valid_agts]
        fut_yaw = fut_yaw[valid_agts]
        fut_valid = fut_valid[valid_agts]
        agent_from_world = agent_from_world[valid_agts]
        map_new_ind_to_initial_ind = np.arange(valid_agts.shape[0])[valid_agts]
    else:
        map_new_ind_to_initial_ind = np.arange(valid_agts.shape[0])

    # Pick the pair with proper relative distance and angle
    yaw_diff = np.abs(np.expand_dims(fut_yaw, axis=1) - np.expand_dims(fut_yaw, axis=0))  # Shape: (M, M, T)
    pos_diff = np.linalg.norm(np.expand_dims(fut_pos, axis=1) - np.expand_dims(fut_pos, axis=0), axis=-1)  # Shape: (M, M, T)
    # print('yaw_diff[:,:,0]', yaw_diff[:,:,0].shape, yaw_diff[:,:,0])
    # print('yaw_diff[:,:,20]', yaw_diff[:,:,20].shape, yaw_diff[:,:,20])
    yaw_max_filter = (yaw_diff[:,:,0] < angle_diff_max_th) * (yaw_diff[:,:,20] < angle_diff_max_th)
    pos_max_filter = (pos_diff[:,:,0] < dist_diff_max_th) * (pos_diff[:,:,20] < dist_diff_max_th)
    pos_min_filter = (pos_diff[:,:,0] > dist_diff_min_th) * (pos_diff[:,:,20] > dist_diff_min_th)
    valid_agts = yaw_max_filter * pos_max_filter * pos_min_filter

    # print('np.sum(yaw_max_filter)', np.sum(yaw_max_filter))
    # print('np.sum(pos_max_filter)', np.sum(pos_max_filter))
    # print('np.sum(pos_min_filter)', np.sum(pos_min_filter))
    # print('1 np.sum(valid_agts)', np.sum(valid_agts), valid_agts)
    
    if np.sum(valid_agts) == 0:
        return None
    else:
        indices = np.argwhere(valid_agts)

        # sort indices by distance
        if sort_mode == 'distance':
            inds_sorted = np.argsort(pos_diff[:,:,20][valid_agts])
            indices = indices[inds_sorted]
        elif sort_mode is None:
            pass
        else:
            raise NotImplementedError

        # print('indices', indices.shape, indices)
        ind1, ind2 = indices[0]
        fut_pos_ind1_frame = transform_coords_np(fut_pos, agent_from_world[ind1])
        ind1_original = map_new_ind_to_initial_ind[ind1]
        ind2_original = map_new_ind_to_initial_ind[ind2]
        
        # ind2 is ahead
        if fut_pos_ind1_frame[ind2, 0, 0] > 0:
            target_ind = ind1_original
            ref_ind = ind2_original
        else:
            target_ind = ind2_original
            ref_ind = ind1_original
        print('target_ind', target_ind)
        print('ref_ind', ref_ind)

    guide_config = {
        'name' : 'gptkeepdistance',
        'params' : {
                    'target_ind' : target_ind,
                    'ref_ind' : ref_ind,
                    'min_distance' : min_distance,
                    'max_distance' : max_distance,
                },
        'agents' : None,
    }

    return guide_config







# =============================================================================

HEURISTIC_FUNC = {
    'global_target_pos_at_time' : heuristic_global_target_pos_at_time,
    'global_target_pos' : heuristic_global_target_pos,
    'target_pos_at_time' : heuristic_target_pos_at_time,
    'target_pos' : heuristic_target_pos,
    'agent_collision' : heuristic_agent_collision,
    'map_collision' : heuristic_map_collision,
    'social_group' : heuristic_social_group,

    'global_stop_sign' : heuristic_global_stop_sign,
    'stop_sign' : heuristic_stop_sign,
    'speed_limit': heuristic_speed_limit,

    'target_speed': heuristic_target_speed,

    'gptcollision' : heuristic_gptcollision,
    'gptkeepdistance' : heuristic_gptkeepdistance,
}

def compute_heuristic_guidance(heuristic_config, env, scene_indices, start_frames, example_batch=None):
    '''
    Creates guidance configs for each scene based on the given configuration.
    ''' 
    env.reset(scene_indices=scene_indices, start_frame_index=start_frames)
    heuristic_guidance_cfg = []
    for i, si in enumerate(scene_indices):
        scene_guidance = []
        cur_scene = env._current_scenes[i]
        dt = cur_scene.dataset.desired_dt
        for cur_heur in heuristic_config:
            assert set(('name', 'weight', 'params')).issubset(cur_heur.keys()), "All heuristics must have these 3 fields"
            assert cur_heur['name'] in HEURISTIC_FUNC, "Unrecognized heuristic!"
            dt = cur_heur['params'].pop('dt', dt) # some already include dt, don't want to duplicate
            cur_guidance = HEURISTIC_FUNC[cur_heur['name']](cur_scene, dt, **cur_heur['params'], example_batch=example_batch)
            if cur_guidance is not None:
                if not isinstance(cur_guidance, list):
                    cur_guidance = [cur_guidance]
                for guide_el in cur_guidance:
                    guide_el['weight'] = cur_heur['weight']
                    scene_guidance.append(guide_el)
        heuristic_guidance_cfg.append(scene_guidance)

    return heuristic_guidance_cfg


################## HIGH-QUALITY (OFFLINE) VISUALIZATION #########################

import os

from scipy.signal import savgol_filter
import subprocess
import matplotlib as mpl
mpl.use('Agg') # NOTE: very important to avoid matplotlib memory leak when looping and plotting
import matplotlib.pyplot as plt

from l5kit.geometry import transform_points
from trajdata.simulation.sim_df_cache import SimulationDataFrameCache
from trajdata import UnifiedDataset

from tbsim.utils.vis_utils import COLORS
import tbsim.utils.geometry_utils as GeoUtils

import matplotlib.collections as mcoll
import matplotlib.patches as patches

def get_agt_color(agt_idx):
    # agt_colors = ["grey", "purple", "blue", "green", "orange", "red"]
    agt_colors = ["orchid", "grey", "royalblue", "limegreen", "orange", "salmon"]
    return agt_colors[agt_idx % len(agt_colors)]

def get_agt_cmap(agt_idx):
    agt_colors = ["Purples_r", "Greys_r", "Blues_r", "Greens_r", "Oranges_r", "Reds_r"]
    return agt_colors[agt_idx % len(agt_colors)]

def get_group_color(agt_idx):
    # agt_colors = ["grey", "purple", "blue", "green", "orange", "red"]
    group_colors = ["red", "green", "blue", "orange"]
    return group_colors[agt_idx % len(group_colors)]

def colorline(
        ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def get_trajdata_renderer(desired_data, data_dirs, future_sec=2.0, history_sec=1.0,
                            raster_size=500, px_per_m=5, 
                            rebuild_maps=False,
                            cache_location='~/.unified_data_cache'):
    kwargs = dict(
        cache_location=cache_location,
        desired_data=desired_data,
        data_dirs=data_dirs,
        future_sec=(0.1, future_sec),
        history_sec=(history_sec, history_sec),
        incl_raster_map=True, # so that maps will be cached with correct resolution
        raster_map_params={
                "px_per_m": px_per_m,
                "map_size_px": raster_size,
                "return_rgb": True,
        },
        incl_vector_map=True,
        num_workers=os.cpu_count(),
        desired_dt=0.1,
        rebuild_cache=rebuild_maps,
        rebuild_maps=rebuild_maps,
        # A dictionary that contains functions that generate our custom data.
        # Can be any function and has access to the batch element.
        extras={
            # "closest_lane_point": get_closest_lane_point_wrapper(),
        },
    )

    dataset = UnifiedDataset(**kwargs)
    renderer = UnifiedRenderer(dataset, raster_size=raster_size, resolution=px_per_m)

    return renderer

class UnifiedRenderer(object):
    def __init__(self, dataset, raster_size=500, resolution=2):
        self.dataset = dataset
        self.raster_size = raster_size
        self.resolution = resolution
        num_total_scenes = dataset.num_scenes()
        scene_info = dict()
        for i in range(num_total_scenes):
            si = dataset.get_scene(i)
            scene_info[si.name] = si
        self.scene_info = scene_info

    def do_render_map(self, scene_name):
        """ whether the map needs to be rendered """
        return SimulationDataFrameCache.are_maps_cached(self.dataset.cache_path,
                                                              self.scene_info[scene_name].env_name)

    def render(self, ras_pos, ras_yaw, scene_name):
        scene_info = self.scene_info[scene_name]
        cache = SimulationDataFrameCache(
            self.dataset.cache_path,
            scene_info,
            0,
            self.dataset.augmentations,
        )
        render_map = self.do_render_map(scene_name)

        state_im = np.ones((self.raster_size, self.raster_size, 3))
        if render_map:
            patch_data, _, _ = cache.load_map_patch(
                ras_pos[0],
                ras_pos[1],
                self.raster_size,
                self.resolution,
                (0, 0),
                ras_yaw,
                return_rgb=True
            )
            # drivable area
            state_im[patch_data[0] > 0] = np.array([200, 211, 213]) / 255.
            # road/lane dividers
            state_im[patch_data[1] > 0] = np.array([164, 184, 196]) / 255.
            # crosswalks and sidewalks
            state_im[patch_data[2] > 0] = np.array([96, 117, 138]) / 255.

        raster_from_agent = np.array([
            [self.resolution, 0, 0.5 * self.raster_size],
            [0, self.resolution, 0.5 * self.raster_size],
            [0, 0, 1]
        ])

        world_from_agent: np.ndarray = np.array(
            [
                [np.cos(ras_yaw), np.sin(ras_yaw), ras_pos[0]],
                [-np.sin(ras_yaw), np.cos(ras_yaw), ras_pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        agent_from_world = np.linalg.inv(world_from_agent)

        raster_from_world = raster_from_agent @ agent_from_world

        del cache

        return state_im, raster_from_world

def draw_trajectories(ax, trajectories, raster_from_world, linewidth, alpha=1.0, use_agt_color=False, agt_color=None, z=None):
    raster_trajs = transform_points(trajectories, raster_from_world)
    if isinstance(alpha, float):
        alpha = [alpha] * len(raster_trajs)
    for i, traj in enumerate(raster_trajs):
        if z is not None:
            z_i = z[i]
        else:
            z_i = None
        if use_agt_color:
            cmap = get_agt_cmap(i)
        elif agt_color is not None:
            if isinstance(agt_color, str):
                cmap = agt_color
            elif isinstance(agt_color, list):
                cmap = agt_color[i]
        else:
            cmap = 'viridis'
        # ["grey", "orchid", "royalblue", "limegreen", "orange", "salmon"]
        # if i in [6, 12, 13, 3, 16]:
        colorline(
            ax,
            traj[..., 0],
            traj[..., 1],
            z=z_i,
            cmap=cmap,
            linewidth=linewidth,
            alpha=alpha[i]
        )

def draw_action_traj(ax, action_traj, raster_from_world, world_from_agent, linewidth,alpha=0.9):
    world_trajs = GeoUtils.batch_nd_transform_points_np(action_traj[np.newaxis],world_from_agent[np.newaxis])
    raster_trajs = GeoUtils.batch_nd_transform_points_np(world_trajs, raster_from_world[np.newaxis])
    raster_trajs = TensorUtils.join_dimensions(raster_trajs,0,2)
    colorline(
        ax,
        raster_trajs[..., 0],
        raster_trajs[..., 1],
        cmap="viridis",
        linewidth=linewidth,
        alpha=alpha,
    )

def draw_action_samples(ax, action_samples, raster_from_world, world_from_agent, linewidth,alpha=0.5, cmap="RdPu"):
    world_trajs = GeoUtils.batch_nd_transform_points_np(action_samples, world_from_agent[np.newaxis,np.newaxis])
    raster_trajs = GeoUtils.batch_nd_transform_points_np(world_trajs, raster_from_world[np.newaxis,np.newaxis])
    for traj_i in range(raster_trajs.shape[0]):
        colorline(
            ax,
            raster_trajs[traj_i, :, 0],
            raster_trajs[traj_i, :, 1],
            cmap=cmap,
            linewidth=linewidth,
            alpha=alpha,
        )

def draw_agent_boxes_plt(ax, pos, yaw, extent, raster_from_agent, 
                         outline_colors=None,
                         outline_widths=None,
                         fill_colors=None,
                         mark_agents=None,
                         draw_agent_index=True):
    if fill_colors is not None:
        assert len(fill_colors) == pos.shape[0]
    if outline_colors is not None:
        assert len(outline_colors) == pos.shape[0]
    if outline_widths is not None:
        assert len(outline_widths) == pos.shape[0]
    boxes = GeoUtils.get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2))
    if mark_agents is not None:
        # tuple of agt_idx, marker style, color
        mark_agents = {
            v[0] : (v[1], v[2]) for v in mark_agents
        }
    else:
        mark_agents = dict()

    for bi, b in enumerate(boxes_raster):
        cur_fill_color = get_agt_color(bi) if fill_colors is None else fill_colors[bi]
        cur_outline_color = "grey" if outline_colors is None else outline_colors[bi]
        cur_outline_width = 0.5 if outline_widths is None else outline_widths[bi]

        rect = patches.Polygon(b, fill=True, color=cur_fill_color, zorder=3)
        rect_border = patches.Polygon(b, fill=False, color=cur_outline_color, zorder=3, linewidth=cur_outline_width)
        ax.add_patch(rect)
        ax.add_patch(rect_border)

        if bi in mark_agents:
            rect = patches.Polygon(b, fill=True, color=mark_agents[bi][1], zorder=3)
            rect_border = patches.Polygon(b, fill=False, color=cur_outline_color, zorder=3, linewidth=cur_outline_width)
            ax.add_patch(rect)
            ax.add_patch(rect_border)
            # mark_pos = np.mean(b, axis=0)
            # ax.scatter(mark_pos[0], mark_pos[1], marker=mark_agents[bi][0], color=mark_agents[bi][1], s=120.0, zorder=4, alpha=0.7)
        
        if draw_agent_index:
            # draw agent index for the valid ones
            if not (np.isnan(np.mean(b[:,0])).any() or np.isnan(np.mean(b[:,1])).any()):
                ax.text(np.mean(b[:,0]), np.mean(b[:,1]), str(bi), fontsize=5, color='black', ha='center', va='center', clip_on=True, zorder=5)

def draw_constraint(ax, loc, rel_time, max_time, raster_from_world, world_from_agent, marker_color='r', marker_size=32.0, bounding_box=None):
    if world_from_agent is not None:
        world_loc = GeoUtils.batch_nd_transform_points_np(loc[np.newaxis], world_from_agent)
    else:
        world_loc = loc[np.newaxis]

    raster_loc = GeoUtils.batch_nd_transform_points_np(world_loc, raster_from_world)[0]
    if rel_time is not None:
        cmap = mpl.cm.get_cmap('viridis')
        ax.plot(raster_loc[0:1], raster_loc[1:2], 'x', color=cmap(float(rel_time)/max_time))

    if bounding_box is not None:
        # for stop_sign box
        x = raster_loc[0:1]
        y = raster_loc[1:2]
        bx, by = bounding_box
        
        stop_box_x_min = x - bx/2
        stop_box_x_max = x + bx/2
        stop_box_y_min = y - by/2
        stop_box_y_max = y + by/2

        ax.plot(
            [stop_box_x_min, stop_box_x_min, stop_box_x_max, stop_box_x_max, stop_box_x_min],
            [stop_box_y_min, stop_box_y_max, stop_box_y_max, stop_box_y_min, stop_box_y_min],
            color=marker_color,
            linewidth=0.5,
        )
    else:
        # ax.scatter(raster_loc[0:1], raster_loc[1:2], color=marker_color, s=16.0, edgecolors='red', zorder=3)
        ax.scatter(raster_loc[0:1], raster_loc[1:2], color=marker_color, s=marker_size, zorder=3, alpha=1.0)



def draw_scene_data(ax, scene_name, scene_data, starting_frame, rasterizer, 
                    guidance_config=None,
                    constraint_config=None,
                    draw_agents=True,
                    draw_trajectory=True,
                    draw_action=True,
                    draw_diffusion_step=None,
                    n_step_action=5,
                    draw_action_sample=False,
                    traj_len=200,
                    traj_alpha=1.0,
                    use_agt_color=False,
                    marker_size=32.0,
                    ras_pos=None,
                    linewidth=3.0,
                    draw_agent_index=True,
                    draw_mode='action'):
    t = starting_frame
    if ras_pos is None:
        # ras_pos = scene_data["centroid"][0, t]
        ras_pos = np.mean(scene_data["centroid"][:,0], axis=0)

    state_im, raster_from_world = rasterizer.render(
        ras_pos=ras_pos,
        # ras_yaw=scene_data["yaw"][0, t],
        ras_yaw=0,
        # ras_yaw=np.pi,
        scene_name=scene_name
    )
    extent_scale = 1.0

    # agent drawing colors (may be modified by guidance config)
    fill_colors = np.array([get_agt_color(aidx) for aidx in range(scene_data["centroid"].shape[0])])
    outline_colors = np.array([COLORS["agent_contour"] for _ in range(scene_data["centroid"].shape[0])])
    outline_widths = np.array([0.5 for _ in range(scene_data["centroid"].shape[0])])
    mark_agents = []
    

    ax.imshow(state_im)
    # print('starting_frame', starting_frame)
    # print('scene_data["action_sample_positions"].shape', scene_data["action_sample_positions"].shape)
    # print('scene_data["centroid"].shape', scene_data["centroid"].shape)
    # print('scene_data["attn_weights"].shape', scene_data["attn_weights"].shape)

    if draw_mode == 'entire_traj_attn':
        chosen_ind = 3
        cmap = plt.cm.Reds

        print("scene_data['attn_weights'].shape", scene_data['attn_weights'].shape)
        if 'attn_weights' in scene_data:
            attn_weights = scene_data['attn_weights']
            
            # (M1, T_sim, N, T, M2) -> (M1, N, T, M2) -> (N, T, M1, M2) -> (T, M1, M2) -> (M1, M2)
            # attn_weights = attn_weights[:, t, ...].transpose(1, 2, 0, 3).mean(0)[31]

            # (M1, T_sim, T, M2) -> (M2)
            # attn_weights_chosen_ind = attn_weights[chosen_ind, t, 31, :]
            # (M1, T_sim, T, M2) -> (T_sim, M2) -> (M2, T_sim)
            attn_weights_chosen_ind = attn_weights[chosen_ind, t:t+traj_len, 31, :].transpose(1, 0)
        else: # equally attend to every agent if attn_weights is not available
            # attn_weights_chosen_ind = [1.0 for _ in range(scene_data["centroid"].shape[0])]
            attn_weights_chosen_ind = np.ones(scene_data["centroid"].shape[0], traj_len)
        
        # m = np.mean(attn_weights_chosen_ind, axis=0)
        # s = np.std(attn_weights_chosen_ind, axis=0)
        # attn_weights_chosen_ind = np.clip(attn_weights_chosen_ind, m-2*s, m+2*s)
        # print('attn_weights_chosen_ind', attn_weights_chosen_ind.shape)
        attn_value_box = attn_weights_chosen_ind[:, 0]
        agt_box_color_value = mcolors.Normalize(np.min(attn_value_box), np.max(attn_value_box))(attn_value_box)
        # print('attn_value_box', attn_value_box.shape, attn_value_box)

        attn_value = mcolors.Normalize(np.min(attn_weights_chosen_ind), np.max(attn_weights_chosen_ind))(attn_weights_chosen_ind)
        # print('attn_value', attn_value.shape, attn_value)


        attn_value_box = np.clip(attn_value_box, 0.05, 0.95)
        attn_value = np.clip(attn_value, 0.05, 0.95)
        

        traj_alpha = [0.8 for _ in range(scene_data["centroid"].shape[0])]
        traj_line_width = linewidth
        traj_use_agt_color = False
        traj_z = attn_value
        traj_agt_color = 'Reds'

        fill_colors = np.array([cmap(agt_box_color_value[aidx]) for aidx in range(scene_data["centroid"].shape[0])])

        traj_alpha[chosen_ind] = 0.0

    else:
        traj_alpha = 0.9 # 0.6 for GPT pair rules plotting
        traj_line_width = 1.2 * linewidth # 0.7 * linewidth for GPT pair rules plotting
        traj_use_agt_color = use_agt_color
        traj_agt_color = None
        traj_z = None


    if draw_action_sample and "action_sample_positions" in scene_data:
        NA = scene_data["action_sample_positions"].shape[0]
        for aidx in range(NA):
            draw_action_samples(
                ax,
                action_samples=scene_data["action_sample_positions"][aidx, t],
                raster_from_world=raster_from_world,
                # actions are always wrt to the frame that they were planned in
                # world_from_agent = scene_data["world_from_agent"][aidx,t],
                world_from_agent = scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                linewidth=linewidth*0.5,
                alpha=0.3,
                cmap=get_agt_cmap(aidx),
            )

    if draw_action and "action_traj_positions" in scene_data:
        NA = scene_data["action_traj_positions"].shape[0]
        for aidx in range(NA):
            draw_action_traj(
                ax,
                action_traj=scene_data["action_traj_positions"][aidx, t],
                raster_from_world=raster_from_world,
                # actions are always wrt to the frame that they were planned in
                world_from_agent = scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                linewidth=linewidth*0.75
            )

    if draw_trajectory and "centroid" in scene_data:
        draw_trajectories(
            ax,
            trajectories=scene_data["centroid"][:, t:t+traj_len],
            raster_from_world=raster_from_world,
            linewidth=traj_line_width,
            use_agt_color=traj_use_agt_color,
            agt_color=traj_agt_color,
            alpha=traj_alpha,
            z=traj_z,
        )

    if draw_diffusion_step is not None:
        NA = scene_data["diffusion_steps_traj"].shape[0]
        for aidx in range(NA):
            draw_action_traj(
                ax,
                action_traj=scene_data["diffusion_steps_traj"][aidx, t, :, draw_diffusion_step, :2], # positions
                raster_from_world=raster_from_world,
                # actions are always wrt to the frame that they were planned in
                world_from_agent = scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                linewidth=linewidth*0.6
            )

    # overwrite the previous agent and trajectory for chosen_ind
    if draw_mode == 'entire_traj_attn':
        mark_agents.append((chosen_ind, "*", 'blue'))
        draw_trajectories(
            ax,
            trajectories=scene_data["centroid"][[chosen_ind], t:t+traj_len],
            raster_from_world=raster_from_world,
            linewidth=1.5*linewidth,
            use_agt_color=False,
            agt_color='Blues',
            z=[0.9 for _ in range(traj_len)],
            alpha=0.7,
        )


    if guidance_config is not None:
        social_group_cnt = 0
        for cur_guide in guidance_config:
            if cur_guide['name'] == 'target_pos_at_time':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # only plot if waypoint is coming in future
                    rel_time = cur_guide['params']['target_time'][aidx] - t
                    if rel_time >= 0:
                        draw_constraint(ax, 
                                        np.array(cur_guide['params']['target_pos'][aidx]),
                                        # conver to "global" timestamp
                                        rel_time,
                                        n_step_action,
                                        raster_from_world,
                                        # constraints are always wrt to the local planning frame of agent
                                        scene_data["world_from_agent"][saidx,::n_step_action][int(t/n_step_action)],
                                        marker_color=get_agt_color(saidx),
                                        marker_size=marker_size,)
            elif cur_guide['name'] == 'target_pos':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # only plot if waypoint is coming in future
                    draw_constraint(ax, 
                                    np.array(cur_guide['params']['target_pos'][aidx]),
                                    None,
                                    n_step_action,
                                    raster_from_world,
                                    # constraints are always wrt to the local planning frame of agent
                                    scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                                    marker_color=get_agt_color(saidx),
                                    marker_size=marker_size,
                                    bounding_box=np.array(cur_guide['params']['stop_box_dim'][saidx]))
            elif cur_guide['name'] == 'global_target_pos_at_time':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # only plot if waypoint is coming in future
                    rel_time = cur_guide['params']['target_time'][aidx] - t
                    if rel_time >= 0:
                        draw_constraint(ax, 
                                        np.array(cur_guide['params']['target_pos'][aidx]),
                                        # conver to "global" timestamp
                                        rel_time,
                                        n_step_action,
                                        raster_from_world,
                                        # global constraints are already in world frame
                                        None,
                                        marker_color=get_agt_color(saidx),
                                        marker_size=marker_size,)
            elif cur_guide['name'] == 'global_target_pos':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    # if aidx in [6, 12, 13, 3, 16]:
                    draw_constraint(ax, 
                                    np.array(cur_guide['params']['target_pos'][aidx]),
                                    None,
                                    n_step_action,
                                    raster_from_world,
                                    None,
                                    marker_color=get_agt_color(saidx),
                                    marker_size=marker_size)
            elif cur_guide['name'] == 'social_group':
                cur_group_color = get_group_color(social_group_cnt)
                outline_colors[cur_guide['agents']] = cur_group_color
                outline_widths[cur_guide['agents']] = 1.0
                # mark to denote leader
                mark_agents.append((cur_guide['params']['leader_idx'], "*", cur_group_color))

                social_group_cnt += 1
            elif cur_guide['name'] == 'stop_sign':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    draw_constraint(ax,
                                    np.array(cur_guide['params']['stop_sign_pos'][aidx]),
                                    None,
                                    n_step_action,
                                    raster_from_world,
                                    # constraints are always wrt to the local planning frame of agent
                                    # TBD: make this consistent with loss and metric
                                    scene_data["world_from_agent"][saidx,::n_step_action][int(t/n_step_action)],
                                    marker_color=get_agt_color(saidx),
                                    marker_size=marker_size,
                                    bounding_box=np.array(cur_guide['params']['stop_box_dim'][aidx]))
            elif cur_guide['name'] == 'global_stop_sign':
                for aidx, saidx in enumerate(cur_guide['agents']):
                    draw_constraint(ax,
                                    np.array(cur_guide['params']['stop_sign_pos'][aidx]),
                                    None,
                                    n_step_action,
                                    raster_from_world,
                                    None,
                                    marker_color=get_agt_color(saidx),
                                    marker_size=marker_size,
                                    bounding_box=np.array(cur_guide['params']['stop_box_dim'][aidx]))
            elif cur_guide['name'] in ['gptcollision', 'gptkeepdistance']:
                target_ind = cur_guide['params']['target_ind']
                ref_ind = cur_guide['params']['ref_ind']

                # mark to denote the chosen pair
                mark_agents.append((target_ind, "*", 'red'))
                mark_agents.append((ref_ind, "*", 'blue'))

                # red_cmap = mcolors.ListedColormap(['red'])
                # blue_cmap = mcolors.ListedColormap(['blue'])
                red_cmap = 'Reds_r'
                blue_cmap = 'Blues_r'

                draw_trajectories(
                    ax,
                    trajectories=scene_data["centroid"][[target_ind], t:t+traj_len],
                    raster_from_world=raster_from_world,
                    linewidth=1.5*linewidth,
                    use_agt_color=False,
                    agt_color=red_cmap,
                    alpha=1.0,
                )

                draw_trajectories(
                    ax,
                    trajectories=scene_data["centroid"][[ref_ind], t:t+traj_len],
                    raster_from_world=raster_from_world,
                    linewidth=1.5*linewidth,
                    use_agt_color=False,
                    agt_color=blue_cmap,
                    alpha=1.0,
                )



    if constraint_config is not None:
        # TODO this assumes in the local frame of last planned
        #       will need to change for global
        # or really should be only visualizing configs that we're active at that re-plan step.
        # viz target location
        NA = len(constraint_config['agents'])
        for aidx in range(NA):
            # only plot if waypoint is coming in future
            rel_time = constraint_config['times'][aidx] - t
            if rel_time >= 0:
                draw_constraint(ax, 
                                np.array(constraint_config['locs'][aidx]),
                                # conver to "global" timestamp
                                rel_time,
                                n_step_action,
                                raster_from_world,
                                # constraints are always wrt to the local planning frame of agent
                                scene_data["world_from_agent"][aidx,::n_step_action][int(t/n_step_action)],
                                marker_color=get_agt_color(aidx),
                                marker_size=marker_size,)
    
    if draw_agents:
        draw_agent_boxes_plt(
            ax,
            pos=scene_data["centroid"][:, t],
            yaw=scene_data["yaw"][:, [t]],
            extent=scene_data["extent"][:, t, :2] * extent_scale,
            raster_from_agent=raster_from_world,
            outline_colors=outline_colors.tolist(),
            outline_widths=outline_widths.tolist(),
            fill_colors=fill_colors.tolist(), #COLORS["agent_fill"]
            mark_agents=mark_agents,
            draw_agent_index=draw_agent_index,
        )

    # set range to plot to focus on the important part of the scene
    # print('state_im.shape', state_im.shape)

    # 68->907 gptcollision, 50step
    # ax.set_xlim([130, state_im.shape[1]-130])
    # ax.set_ylim([130, state_im.shape[0]-130])
    # mark collision position
    # CTG
    # ax.scatter(210, 180, marker='*', color='blue', s=120.0, zorder=4, alpha=0.7)
    # ax.scatter(210, 180, marker='*', color='red', s=160.0, zorder=4, alpha=0.7)

    # CTG++
    # ax.scatter(194, 182, marker='*', color='blue', s=120.0, zorder=4, alpha=0.7)
    # ax.scatter(194, 182, marker='*', color='red', s=160.0, zorder=4, alpha=0.7)
    

    # 88->39 gptcollision, 55step
    # ax.set_xlim([120, state_im.shape[1]-130])
    # ax.set_ylim([80, state_im.shape[0]-170])

    # 44->626 gptkeepdistance, 75step
    # ax.set_xlim([170, state_im.shape[1]-0])
    # ax.set_ylim([70, state_im.shape[0]-100])
    # mark offroad position
    # ax.scatter(280, 148, marker='^', color='red', s=120.0, zorder=4, alpha=0.7)
    # ax.scatter(320, 123, marker='^', color='blue', s=100.0, zorder=4, alpha=0.7)

    # ax.set_xlim([70, state_im.shape[1]-70])
    # ax.set_ylim([70, state_im.shape[0]-70])

    ax.set_xlim([0, state_im.shape[1]])
    ax.set_ylim([0, state_im.shape[0]])

    if not rasterizer.do_render_map(scene_name):
        ax.grid(True)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    else:
        # ax.grid(False)
        ax.grid(True)
    # ax.invert_xaxis()

    # TBD: hard-coded for now
    ax.grid(False)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    del state_im

def preprocess(scene_data, filter_yaw=False):
    data = dict()
    for k in scene_data.keys():
        data[k] = scene_data[k][:].copy()

    if filter_yaw:
        data["yaw"] = savgol_filter(data["yaw"], 11, 3)
    return data

def create_video(img_path_form, out_path, fps):
    '''
    Creates a video from a format for frame e.g. 'data_out/frame%04d.png'.
    Saves in out_path.
    '''
    subprocess.run(['ffmpeg', '-y', '-r', str(fps), '-i', img_path_form, '-vf' , "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_path])


def scene_to_video(rasterizer, scene_data, scene_name, output_dir,
                    guidance_config=None,
                    constraint_config=None,
                    filter_yaw=False,
                    fps=10,
                    n_step_action=5,
                    first_frame_only=False,
                    sim_num=0,
                    save_every_n_frames=1,
                    draw_mode='action'):
    scene_data = preprocess(scene_data, filter_yaw)
    frames = [0] if first_frame_only else range(0, scene_data["centroid"].shape[1], save_every_n_frames)
    for i, frame_i in enumerate(frames):
        fig, ax = plt.subplots()
        # draw_mode = 'entire_traj'
        # draw_mode = 'entire_traj_attn'
        if draw_mode == 'action':
            draw_agents = True
            draw_action = True
            draw_trajectory = False
            traj_len = 200 # effective only when draw_trajectory is True
            draw_action_sample = True

            linewidth = 2.0
            use_agt_color = False
            marker_size = 32
            traj_alpha = 1.0
        elif draw_mode in ['entire_traj', 'entire_traj_attn']:
            draw_agents = True
            draw_action = False
            draw_trajectory = True
            draw_action_sample = False
            traj_len = 200

            linewidth = 2.0
            use_agt_color = True
            marker_size = 32
            traj_alpha = 0.6
            if draw_mode == 'entire_traj_attn':
                traj_alpha = 1.0
        elif draw_mode == 'map':
            draw_agents = False
            draw_action = False
            draw_trajectory = False
            draw_action_sample = False
            traj_len = 200

            linewidth = 2.0
            use_agt_color = True
            marker_size = 800
            traj_alpha = 1.0
        else:
            raise NotImplementedError

        draw_scene_data(
            ax,
            scene_name,
            scene_data,
            frame_i,
            rasterizer,
            guidance_config=guidance_config,
            constraint_config=constraint_config,
            draw_agents=draw_agents,
            draw_trajectory=draw_trajectory,
            draw_action=draw_action,
            draw_action_sample=draw_action_sample,
            n_step_action=n_step_action,
            traj_len=traj_len,
            traj_alpha=traj_alpha,
            use_agt_color=use_agt_color,
            marker_size=marker_size,
            linewidth=linewidth,
            ras_pos=np.mean(scene_data["centroid"][:, 0], axis=0),
            draw_mode=draw_mode,
        )

        if first_frame_only:
            ffn = os.path.join(output_dir, "{sname}_{simnum:04d}_{framei:03d}.png").format(sname=scene_name, simnum=sim_num, framei=frame_i)
        else:
            video_dir = os.path.join(output_dir, scene_name + '_%04d' % (sim_num))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            ffn = os.path.join(video_dir, "{:03d}.png").format(i)
        plt.savefig(ffn, dpi=200, bbox_inches="tight", pad_inches=0)
        print("Figure written to {}".format(ffn))
        fig.clf()
        plt.close(fig)

    if not first_frame_only:
        create_video(os.path.join(video_dir, "%03d.png"), video_dir + ".mp4", fps=fps)

def draw_diffusion_prof(ax, scene_data, frame_i, diffusion_step,
                        val_inds=[4,5],
                        val_names=['acc', 'yawvel'],
                        first_frame_only=False):
    assert len(val_inds) == len(val_names)
    t = frame_i
    NT = scene_data["diffusion_steps_traj"].shape[2]
    NA = scene_data["diffusion_steps_traj"].shape[0]
    for aidx in range(NA):
        for cidx, vinfo in enumerate(zip(val_inds, val_names)):
            vidx, vname = vinfo
            ax[aidx,cidx].plot(np.arange(NT), scene_data["diffusion_steps_traj"][aidx, t, :, diffusion_step, vidx],
                                c=plt.rcParams['axes.prop_cycle'].by_key()['color'][aidx % 9])
            ax[aidx,cidx].set_ylabel(vname + " agent %d" % (aidx))
            # if vname == 'acc':
            #     ax[aidx,cidx].set_ylim(-1.0, 1.0)
            # if vname == 'yawvel':
            #     ax[aidx,cidx].set_ylim(-1.0, 1.0)
            # if vname == 'vel':
            #     ax[aidx,cidx].set_ylim(0, 2.5)

def scene_diffusion_video(rasterizer, scene_data, scene_name, output_dir,
                             n_step_action=5,
                             viz_traj=True,
                             viz_prof=False):
    scene_data = preprocess(scene_data, False)
    assert "diffusion_steps_traj" in scene_data
    print(scene_data["diffusion_steps_traj"].shape)
    num_diff_steps = scene_data["diffusion_steps_traj"].shape[-2]
    NA = scene_data["diffusion_steps_traj"].shape[0]
    # first acceleration profiles
    if viz_prof:
        for frame_i in range(0, 1): #range(0, scene_data["diffusion_steps_traj"].shape[1], n_step_action):
            video_dir = os.path.join(output_dir, scene_name + "_diffusion_ctrl_%03d" % (frame_i))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            for diff_step in range(num_diff_steps):
                num_col = 3
                fig, ax = plt.subplots(NA, num_col)
                draw_diffusion_prof(
                    ax,
                    scene_data,
                    frame_i,
                    diff_step,
                    val_inds=[2, 4, 5], # which indices of the state to plot
                    val_names=['vel', 'acc', 'yawvel']
                )
                plt.subplots_adjust(right=1.3)
                # fig.set_figheight(4)
                # fig.set_figwidth(8.05)
                ffn = os.path.join(video_dir, "{:05d}.png").format(diff_step)
                plt.savefig(ffn, dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                print("Figure written to {}".format(ffn))

            per_frame_len_sec = 4.0
            create_video(os.path.join(video_dir, "%05d.png"), video_dir + ".mp4", fps=(num_diff_steps/per_frame_len_sec))

    if viz_traj:
        # then state trajectories (resulting from accel)
        for frame_i in range(0, 1): #range(0, scene_data["diffusion_steps_traj"].shape[1], n_step_action):
            video_dir = os.path.join(output_dir, scene_name + "_diffusion_%03d" % (frame_i))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            for diff_step in range(num_diff_steps):
                fig, ax = plt.subplots()
                draw_scene_data(
                    ax,
                    scene_name,
                    scene_data,
                    frame_i,
                    rasterizer,
                    draw_trajectory=False,
                    draw_action=False,
                    draw_action_sample=False,
                    draw_diffusion_step=diff_step,
                    n_step_action=n_step_action,
                    traj_len=20,
                    linewidth=2.0,
                    ras_pos=scene_data["centroid"][0, 0]
                )
                ffn = os.path.join(video_dir, "{:05d}.png").format(diff_step)
                plt.savefig(ffn, dpi=212, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)
                print("Figure written to {}".format(ffn))

            per_frame_len_sec = 4.0
            create_video(os.path.join(video_dir, "%05d.png"), video_dir + ".mp4", fps=(num_diff_steps/per_frame_len_sec))

def visualize_guided_rollout(output_dir, rasterizer, si, scene_data,
                            guidance_config=None,
                            constraint_config=None,
                            filter_yaw=False,
                            fps=10,
                            n_step_action=5,
                            viz_diffusion_steps=False,
                            first_frame_only=False,
                            sim_num=0,
                            save_every_n_frames=1,
                            draw_mode='action'):
    '''
    guidance/constraint configs are for the given scene ONLY.
    '''
    if viz_diffusion_steps:
        print('Visualizing diffusion for %s...' % (si))
        scene_diffusion_video(rasterizer, scene_data, si, output_dir,
                                n_step_action=n_step_action,
                                viz_prof=False,
                                viz_traj=True)
    print('Visualizing rollout for %s...' % (si))
    scene_to_video(rasterizer, scene_data, si, output_dir,
                    guidance_config=guidance_config,
                    constraint_config=constraint_config,
                    filter_yaw=filter_yaw,
                    fps=fps,
                    n_step_action=n_step_action,
                    first_frame_only=first_frame_only,
                    sim_num=sim_num,
                    save_every_n_frames=save_every_n_frames,
                    draw_mode=draw_mode)
