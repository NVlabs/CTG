import math
import numpy as np

from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class OrcaTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(OrcaTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/.unified_data_cache"
        #
        # with maps
        #
        # self.trajdata_source_train = ["orca_maps-train"]
        # self.trajdata_source_valid = ["orca_maps-val"]
        #
        # no maps
        #
        # self.trajdata_source_train = ["orca_no_maps-train"]
        # self.trajdata_source_valid = ["orca_no_maps-val"]
        #
        # mixed
        #
        self.trajdata_source_train = ["orca_maps-train", "orca_no_maps-train"]
        self.trajdata_source_valid = ["orca_maps-val", "orca_no_maps-val"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "orca_maps" : "../orca-sim/datagen_out/datagen_trajdata_v3",
            "orca_no_maps" : "../orca-sim/datagen_out/datagen_trajdata_v3",
        }

        # for debug
        self.trajdata_rebuild_cache = False

        self.rollout.enabled = False
        self.rollout.save_video = True
        self.rollout.every_n_steps = 5000
        self.rollout.warm_start_n_steps = 0

        # training config
        self.training.batch_size = 400
        self.training.num_steps = 100000
        self.training.num_data_workers = 8

        self.save.every_n_steps = 3000 # 1000
        self.save.best_k = 5

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 4
        self.validation.every_n_steps = 200 #570 # 210
        self.validation.num_steps_per_epoch = 100 # 25

        self.on_ngc = False
        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = False  # enable tensorboard logging
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "tbsim"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100


class OrcaEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(OrcaEnvConfig, self).__init__()

        #
        # with map
        #
        self.data_generation_params.trajdata_incl_map = True
        self.data_generation_params.trajdata_max_agents_distance = np.inf # 0.001
        self.rasterizer.num_sem_layers = 2

        #
        # no map
        #
        # self.data_generation_params.trajdata_incl_map = False
        # self.data_generation_params.trajdata_max_agents_distance = np.inf
        # self.rasterizer.num_sem_layers = 0

        self.data_generation_params.trajdata_only_types = ["pedestrian"]

        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # number of semantic layers that will be used (based on which trajdata dataset is being used)
        # how to group layers together to viz RGB image
        self.rasterizer.rgb_idx_groups = ([1], [0], [1])
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1.0 / 12.0
        # where the agent is on the map, (0.0, 0.0) is the center and image width is 2.0, i.e. (1.0, 0.0) is the right edge
        self.rasterizer.ego_center = (-0.5, 0.0)
        # if incl_map = True, but no map is available, will fill dummy map with this value
        self.rasterizer.no_map_fill_value = 0.5 # -1.0