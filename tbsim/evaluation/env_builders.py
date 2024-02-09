from copy import deepcopy
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from trajdata import AgentType, UnifiedDataset

from tbsim.l5kit.vectorizer import build_vectorizer


from tbsim.configs.eval_config import EvaluationConfig
from tbsim.configs.base import ExperimentConfig
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
from tbsim.envs.env_l5kit import EnvL5KitSimulation
from tbsim.envs.env_trajdata import EnvUnifiedSimulation, EnvSplitUnifiedSimulation
from tbsim.utils.config_utils import translate_l5kit_cfg, translate_trajdata_cfg, translate_pass_trajdata_cfg
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP, get_closest_lane_point_wrapper, get_full_fut_traj, get_full_fut_valid
import tbsim.envs.env_metrics as EnvMetrics
from tbsim.evaluation.metric_composers import CVAEMetrics, OccupancyMetrics

from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit


class EnvironmentBuilder(object):
    """Builds an simulation environment for evaluation."""
    def __init__(self, eval_config: EvaluationConfig, exp_config: ExperimentConfig, device):
        self.eval_cfg = eval_config
        self.exp_cfg = exp_config
        self.device = device

    def _get_analytical_metrics(self):
        metrics = dict(
            all_off_road_rate=EnvMetrics.OffRoadRate(),
            all_disk_off_road_rate=EnvMetrics.DiskOffRoadRate(),
            # all_sem_layer_rate=EnvMetrics.SemLayerRate(),
            all_collision_rate=EnvMetrics.CollisionRate(),
            all_disk_collision_rate=EnvMetrics.DiskCollisionRate(),
            # agents_collision_rate=EnvMetrics.CollisionRate(),
            all_coverage=EnvMetrics.OccupancyCoverage(
                gridinfo={"offset": np.zeros(2), "step": 2.0*np.ones(2)},
            ),
            all_diversity=EnvMetrics.OccupancyDiversity(
                gridinfo={"offset": np.zeros(2), "step": 4.0*np.ones(2)},
            ),
            all_failure=EnvMetrics.CriticalFailure(num_offroad_frames=2),
            all_comfort=EnvMetrics.Comfort(sim_dt=self.exp_cfg.algo.step_time, stat_dt=0.5),
        )
        return metrics

    def _get_learned_metrics(self):
        perturbations = dict()
        if self.eval_cfg.perturb.enabled:
            for sigma in self.eval_cfg.perturb.OU.sigma:
                perturbations["OU_sigma_{}".format(sigma)] = OrnsteinUhlenbeckPerturbation(
                    theta=self.eval_cfg.perturb.OU.theta*np.ones(3),
                    sigma=sigma*np.array(self.eval_cfg.perturb.OU.scale))

        cvae_metrics = CVAEMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        learned_occu_metric = OccupancyMetrics(
            eval_config=self.eval_cfg,
            device=self.device,
            ckpt_root_dir=self.eval_cfg.ckpt_root_dir,
        )

        metrics = dict(
            all_cvae_metrics=cvae_metrics.get_metrics(self.eval_cfg,perturbations=perturbations,rolling = self.eval_cfg.cvae.rolling,
            rolling_horizon = self.eval_cfg.cvae.rolling_horizon,env=self.eval_cfg.env,),
            all_occu_likelihood=learned_occu_metric.get_metrics(self.eval_cfg,perturbations=perturbations,rolling = self.eval_cfg.occupancy.rolling,
            rolling_horizon = self.eval_cfg.occupancy.rolling_horizon,env=self.eval_cfg.env,)
        )
        return metrics

    def get_env(self):
        raise NotImplementedError


class EnvL5Builder(EnvironmentBuilder):
    def get_env(self):
        exp_cfg = self.exp_cfg.clone()
        os.environ["L5KIT_DATA_FOLDER"] = self.eval_cfg.dataset_path
        exp_cfg.unlock()

        dm = LocalDataManager(None)
        l5_config = translate_l5kit_cfg(exp_cfg)

        rasterizer = build_rasterizer(l5_config, dm)
        vectorizer = build_vectorizer(l5_config, dm)
        eval_zarr = ChunkedDataset(dm.require(exp_cfg.train.dataset_valid_key)).open()
        env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)
        l5_config = deepcopy(l5_config)
        l5_config["raster_params"]["raster_size"] = (500, 500)
        l5_config["raster_params"]["pixel_size"] = (0.3, 0.3)
        # l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
        render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.distance_th_far = 1e+5  # keep controlling everything
        exp_cfg.env.simulation.disable_new_agents = True
        exp_cfg.env.generate_agent_obs = True
        exp_cfg.env.simulation.distance_th_close = 30  # control everything within this bound
        exp_cfg.env.rasterizer.filter_agents_threshold = 0.8  # control everything that's above this confidence threshold

        exp_cfg.lock()

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if self.eval_cfg.metrics.compute_learned_metrics:
            metrics.update(self._get_learned_metrics())

        env = EnvL5KitSimulation(
            exp_cfg.env,
            dataset=env_dataset,
            num_scenes=self.eval_cfg.num_scenes_per_batch,
            prediction_only=False,
            renderer=render_rasterizer,
            metrics=metrics,
            seed=self.eval_cfg.seed,
            skimp_rollout=self.eval_cfg.skimp_rollout,
        )

        return env


class EnvNuscBuilder(EnvironmentBuilder):
    def get_env(self,split_ego=False,parse_obs=True):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.train.dataset_path = self.eval_cfg.dataset_path
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        data_cfg = translate_trajdata_cfg(exp_cfg)

        # This needs to be done for every environment that wants
        # to have access to future trajectories during evaluation.
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance

        kwargs = dict(
            desired_data=["val"],
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                "nusc_trainval": data_cfg.dataset_path,
                "nusc_mini": data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center
            },
            incl_vector_map=True,
            num_workers=os.cpu_count(),
            desired_dt=data_cfg.step_time,
            standardize_data=data_cfg.standardize_data,
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
            },
        )
        
        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        # if self.eval_cfg.metrics.compute_learned_metrics:
        #     metrics.update(self._get_learned_metrics())
        if split_ego:
            env = EnvSplitUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
                split_ego=split_ego,
                parse_obs = parse_obs,
            )
        else:
            env = EnvUnifiedSimulation(
                exp_cfg.env,
                dataset=env_dataset,
                seed=self.eval_cfg.seed,
                num_scenes=self.eval_cfg.num_scenes_per_batch,
                prediction_only=False,
                metrics=metrics,
            )

        return env

class EnvUnifiedBuilder(EnvironmentBuilder):
    def get_env(self):
        exp_cfg = self.exp_cfg.clone()
        exp_cfg.unlock()
        exp_cfg.env.simulation.num_simulation_steps = self.eval_cfg.num_simulation_steps
        exp_cfg.env.simulation.start_frame_index = exp_cfg.algo.history_num_frames + 1
        exp_cfg.lock()

        # the config used at training time
        # print('exp_cfg', exp_cfg)
        data_cfg = translate_pass_trajdata_cfg(exp_cfg)

        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        if data_cfg.trajdata_predict_types is not None:
            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]
        # TBD: hack to accomodate drivesim rollout data when not enough future data is available
        if 'drivesim' in self.eval_cfg.trajdata_data_dirs.keys():
            future_sec_min = 0.1
        else:
            future_sec_min = future_sec
        kwargs = dict(
            cache_location=Path(self.eval_cfg.trajdata_cache_location).expanduser(),
            desired_data=self.eval_cfg.trajdata_source_test,
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec_min, future_sec),#(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs=self.eval_cfg.trajdata_data_dirs,
            only_types=agent_only_types,
            only_predict=agent_predict_types,
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=data_cfg.trajdata_incl_map,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            incl_vector_map=True,
            vector_map_params = {
                "incl_road_lanes": True,                
                "incl_road_areas": False,                 
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,                 
                # Collation can be quite slow if vector maps are included,
                "no_collate": True,
            },
            # centric=data_cfg.trajdata_centric,
            # hardcode to agent-centric and then use a wrapper to support scene-centric
            # TBD: support scene-centric simulation
            centric='agent',
            scene_description_contains=data_cfg.trajdata_scene_desc_contains,
            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            num_workers=os.cpu_count(),
            rebuild_cache=self.eval_cfg.trajdata_rebuild_cache,
            rebuild_maps=self.eval_cfg.trajdata_rebuild_cache,
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
                "closest_lane_point": get_closest_lane_point_wrapper(self.eval_cfg.evaluation_vec_map_params),
                "full_fut_traj": get_full_fut_traj,
                "full_fut_valid": get_full_fut_valid,
            },
        )

        env_dataset = UnifiedDataset(**kwargs)

        metrics = dict()
        if self.eval_cfg.metrics.compute_analytical_metrics:
            metrics.update(self._get_analytical_metrics())
        if self.eval_cfg.metrics.compute_learned_metrics:
            metrics.update(self._get_learned_metrics())

        # if we don't have a map, can't compute map-based metrics
        if not data_cfg.trajdata_incl_map:
            metrics.pop("all_off_road_rate", None)
            metrics.pop("all_sem_layer_rate", None)
            metrics.pop("all_coverage", None)
            metrics.pop("all_diversity", None)
            metrics.pop("all_failure", None)

        env = EnvUnifiedSimulation(
            exp_cfg.env,
            dataset=env_dataset,
            seed=self.eval_cfg.seed,
            num_scenes=self.eval_cfg.num_scenes_per_batch,
            prediction_only=False,
            metrics=metrics,
            save_action_samples=self.eval_cfg.save_action_samples,
        )

        return env
