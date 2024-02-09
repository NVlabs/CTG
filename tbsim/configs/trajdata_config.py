import math
import numpy as np
from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig

#
# Base configurations for using unified data loader (trajdata)
#

class TrajdataTrainConfig(TrainConfig):
    def __init__(self):
        super(TrajdataTrainConfig, self).__init__()

        # essentially passes through args to unified dataset
        self.datamodule_class = "PassUnifiedDataModule"

        self.trajdata_cache_location = "~/.unified_data_cache"
        # list of desired_data for training set
        self.trajdata_source_train = ["nusc_trainval-train"]
        # list of desired_data for validation set
        self.trajdata_source_valid = ["nusc_trainval-val"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "nusc_trainval" : "../behavior-generation-dataset/nuscenes",
            "nusc_test" : "../behavior-generation-dataset/nuscenes",
            "nusc_mini" : "../behavior-generation-dataset/nuscenes",
        }

        # whether to rebuild the cache or not
        self.trajdata_rebuild_cache = False

        # vec map params (only used by algorithm which uses vec map)
        self.training_vec_map_params = {
            'S_seg': 15,
            'S_point': 80,
            'map_max_dist': 80,
            'max_heading_error': 0.25*np.pi,
            'ahead_threshold': -40,
            'dist_weight': 1.0,
            'heading_weight': 0.1,
        }


class TrajdataEnvConfig(EnvConfig):
    def __init__(self):
        super(TrajdataEnvConfig, self).__init__()

        # NOTE: this should NOT be changed in sub-classes
        self.name = "trajdata"

        #
        # general data options
        #
        self.data_generation_params.trajdata_centric = "agent" # or "scene"
        # which types of agents to include from ['unknown', 'vehicle', 'pedestrian', 'bicycle', 'motorcycle']
        self.data_generation_params.trajdata_only_types = ["vehicle", "pedestrian"]
        self.data_generation_params.trajdata_predict_types = None
        # list of scene description filters
        self.data_generation_params.trajdata_scene_desc_contains = None
        # whether or not to include the map in the data
        self.data_generation_params.trajdata_incl_map = True
        # max distance to be considered neighbors
        self.data_generation_params.trajdata_max_agents_distance = np.inf
        # standardize position and heading for the predicted agnet
        self.data_generation_params.trajdata_standardize_data = True

        #
        # map params -- default for nuscenes
        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # whether or not to rasterize the agent histories
        self.rasterizer.include_hist = True
        # number of semantic layers that will be used (based on which trajdata dataset is being used)
        self.rasterizer.num_sem_layers = 7
        # which layers constitute the drivable area
        #   None uses the default drivable layers for the given data source
        #   empty list assumes the entire map is drivable (even regions with 0 in all layers)
        #   non-empty list only uses the specified layer indices as drivable region
        self.rasterizer.drivable_layers = None
        # how to group layers together to viz RGB image
        self.rasterizer.rgb_idx_groups = ([0, 1, 2], [3, 4], [5, 6])
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 0.5
        # where the agent is on the map, (0.0, 0.0) is the center
        self.rasterizer.ego_center = (-0.5, 0.0)
        # if incl_map = True, but no map is available, will fill dummy map with this value
        self.rasterizer.no_map_fill_value = -1.0
