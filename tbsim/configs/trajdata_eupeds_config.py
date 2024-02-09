import math
import numpy as np

from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig
from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class EupedsTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(EupedsTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/.unified_data_cache"
        # leaves out the ETH-Univ dataset for training
        self.trajdata_source_train = ["eupeds_eth-train_loo"]
        self.trajdata_source_valid = ["eupeds_eth-val_loo"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "eupeds_eth" : "./datasets/eth_ucy", 
            "eupeds_hotel" : "./datasets/eth_ucy",
            "eupeds_univ" : "./datasets/eth_ucy",
            "eupeds_zara1" : "./datasets/eth_ucy",
            "eupeds_zara2" : "./datasets/eth_ucy"
        }

        # for debug
        self.trajdata_rebuild_cache = False

        self.rollout.enabled = False
        self.rollout.save_video = True
        self.rollout.every_n_steps = 5000
        self.rollout.warm_start_n_steps = 0

        # training config
        # assuming dt=0.4, history_frames=8, future_frames=12 (benchmark setting)
        self.training.batch_size = 400
        self.training.num_steps = 72000
        self.training.num_data_workers = 8

        self.save.every_n_steps = 1000
        self.save.best_k = 10

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 70
        self.validation.num_steps_per_epoch = 20

        self.on_ngc = False
        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = False  # enable tensorboard logging
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "tbsim"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100


class EupedsEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(EupedsEnvConfig, self).__init__()

        # no maps to include
        self.data_generation_params.trajdata_incl_map = False
        self.data_generation_params.trajdata_only_types = ["pedestrian"]
        self.data_generation_params.trajdata_max_agents_distance = np.inf

        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # number of semantic layers that will be used (based on which trajdata dataset is being used)
        self.rasterizer.num_sem_layers = 0
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1. / 10.
        # where the agent is on the map, (0.0, 0.0) is the center and image width is 2.0, i.e. (1.0, 0.0) is the right edge
        self.rasterizer.ego_center = (0.0, 0.0)