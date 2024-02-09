import math
import numpy as np
from tbsim.configs.base import AlgoConfig


class BehaviorCloningConfig(AlgoConfig):
    def __init__(self):
        super(BehaviorCloningConfig, self).__init__()
        self.eval_class = "BC"

        self.name = "bc"
        self.model_architecture = "resnet18"
        self.map_feature_dim = 256
        self.history_num_frames = 30
        self.history_num_frames_ego = 30
        self.history_num_frames_agents = 30
        self.future_num_frames = 52
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.1

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


class SpatialPlannerConfig(BehaviorCloningConfig):
    def __init__(self):
        super(SpatialPlannerConfig, self).__init__()
        self.eval_class = None

        self.name = "spatial_planner"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.loss_weights.pixel_res_loss = 1.0
        self.loss_weights.pixel_yaw_loss = 1.0


class AgentPredictorConfig(BehaviorCloningConfig):
    def __init__(self):
        super(AgentPredictorConfig, self).__init__()
        self.eval_class = "HierAgentAware"

        self.name = "agent_predictor"
        self.agent_feature_dim = 128
        self.global_feature_dim = 128
        self.context_size = (30, 30)
        self.goal_conditional = True
        self.goal_feature_dim = 32
        self.decoder.layer_dims = (128, 128, 128)

        self.use_rotated_roi = False
        self.use_transformer = False
        self.roi_layer_key = "layer2"
        self.use_GAN = False
        self.history_conditioning = False

        self.loss_weights.lane_reg_loss = 0.5
        self.loss_weights.GAN_loss = 0.5

        self.optim_params.GAN.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.GAN.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.GAN.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.GAN.regularization.L2 = 0.00  # L2 regularization strength

        self.disable_control_on_stationary = 'current_speed' # ['any_speed', 'current_speed'. 'on_lane', False] vehicles (always stationary) will be forced to 0 during rollout (including denoising iteration)
        self.moving_speed_th = 5e-1 # speed threshold for determining if stationary



class BehaviorCloningGCConfig(BehaviorCloningConfig):
    def __init__(self):
        super(BehaviorCloningGCConfig, self).__init__()
        self.eval_class = None
        self.name = "bc_gc"
        self.goal_feature_dim = 32
        self.decoder.layer_dims = (128, 128)


class EBMMetricConfig(BehaviorCloningConfig):
    def __init__(self):
        super(EBMMetricConfig, self).__init__()
        self.eval_class = None
        self.name = "ebm"
        self.negative_source = "permute"
        self.map_feature_dim = 64
        self.traj_feature_dim = 32
        self.embedding_dim = 32
        self.embed_layer_dims = (128, 64)
        self.loss_weights.infoNCE_loss = 1.0


class OccupancyMetricConfig(BehaviorCloningConfig):
    def __init__(self):
        super(OccupancyMetricConfig, self).__init__()
        self.eval_class = "metric"
        self.name = "occupancy"
        self.loss_weights.pixel_bce_loss = 0.0
        self.loss_weights.pixel_ce_loss = 1.0
        self.agent_future_cond.enabled = True
        self.agent_future_cond.every_n_frame = 5


class VAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(VAEConfig, self).__init__()
        self.eval_class = "TrafficSim"
        self.name = "vae"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32

        self.vae.latent_dim = 4
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)

        self.loss_weights.kl_loss = 1e-4


class DiscreteVAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(DiscreteVAEConfig, self).__init__()
        self.eval_class = "TPP"

        self.name = "discrete_vae"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32

        self.ego_conditioning = False
        self.EC_feat_dim = 64
        self.vae.latent_dim = 10
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.Gaussian_var = True
        self.vae.recon_loss_type = "NLL"
        self.vae.logpi_clamp = -6.0

        self.loss_weights.kl_loss = 10
        self.loss_weights.EC_coll_loss = 10
        self.loss_weights.deviation_loss = 0.5
        self.eval.mode = "mean"

        self.agent_future_cond.enabled = False
        self.agent_future_cond.feature_dim = 32
        self.agent_future_cond.transformer = True

        self.min_std = 0.1


class TreeVAEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(TreeVAEConfig, self).__init__()
        self.eval_class = None

        self.name = "tree_vae"
        self.map_feature_dim = 256
        self.goal_conditional = True
        self.goal_feature_dim = 32
        self.stage = 2
        self.num_frames_per_stage = 10

        self.vae.latent_dim = 4
        self.vae.condition_dim = 128
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 100
        self.vae.encoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.rnn_hidden_size = 100
        self.vae.decoder.mlp_layer_dims = (128, 128)
        self.vae.decoder.Gaussian_var = False
        self.vae.recon_loss_type = "MSE"
        self.vae.logpi_clamp = -6.0
        self.ego_conditioning = True
        self.EC_feat_dim = 64
        self.loss_weights.EC_collision_loss = 10
        self.loss_weights.deviation_loss = 0.5
        self.loss_weights.kl_loss = 10
        self.loss_weights.collision_loss = 5.0
        self.eval.mode = "sum"
        self.scene_centric = True
        self.shuffle = True
        self.vectorize_lane = False
        self.min_std = 0.1
        self.perturb.enabled=True
        self.perturb.N_pert = 3
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = 2.0
        self.perturb.OU.scale = [1.0,1.0,0.2]


class BehaviorCloningECConfig(BehaviorCloningConfig):
    def __init__(self):
        super(BehaviorCloningECConfig, self).__init__()
        self.eval_class = None

        self.name = "bc_ec"
        self.map_feature_dim = 256
        self.goal_conditional = True
        self.goal_feature_dim = 32

        self.EC.feature_dim = 64
        self.EC.RNN_hidden_size = 32
        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.yaw_reg_loss = 0.05
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 4
        self.loss_weights.EC_collision_loss = 5
        self.loss_weights.deviation_loss = 0.2


class GANConfig(BehaviorCloningConfig):
    def __init__(self):
        super(GANConfig, self).__init__()
        self.eval_class = "GAN"

        self.name = "gan"

        self.map_feature_dim = 256
        self.optim_params.disc.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.policy.learning_rate.initial = 1e-4  # generator learning rate

        self.decoder.layer_dims = (128, 128)

        self.traj_encoder.rnn_hidden_size = 100
        self.traj_encoder.feature_dim = 32
        self.traj_encoder.mlp_layer_dims = (128, 128)

        self.gan.latent_dim = 4
        self.gan.loss_type = "lsgan"
        self.gan.disc_layer_dims = (128, 128)
        self.gan.num_eval_samples = 10

        self.loss_weights.prediction_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.0
        self.loss_weights.gan_gen_loss = 1.0
        self.loss_weights.gan_disc_loss = 1.0

        self.optim_params.disc.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.disc.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.disc.regularization.L2 = 0.00  # L2 regularization strength


class TransformerPredConfig(AlgoConfig):
    def __init__(self):
        super(TransformerPredConfig, self).__init__()

        self.name = "TransformerPred"
        self.model_architecture = "Factorized"
        self.history_num_frames = 10
        # this will also create raster history (we need to remove the raster from train/eval dataset - only visualization)
        self.history_num_frames_ego = 10
        self.history_num_frames_agents = 10
        self.future_num_frames = 20
        self.step_time = 0.1
        self.N_t = 2
        self.N_a = 1
        self.d_model = 128
        self.XY_pe_dim = 16
        self.temporal_pe_dim = 16
        self.map_emb_dim = 32
        self.d_ff = 256
        self.head = 4
        self.dropout = 0.01
        self.XY_step_size = [2.0, 2.0]

        self.disable_other_agents = False
        self.disable_map = False
        self.goal_conditioned = True

        # self.disable_lane_boundaries = True
        self.global_head_dropout = 0.0
        self.training_num_N = 10000
        self.N_layer_enc = 2
        self.N_layer_tgt_enc = 1
        self.N_layer_tgt_dec = 1
        self.vmax = 30
        self.vmin = -10
        self.calc_likelihood = False
        self.calc_collision = False

        self.map_enc_mode = "all"
        self.temporal_bias = 0.5
        self.weights.lane_regulation_weight = 1.5
        self.weights.weights_scaling = [1.0, 1.0, 1.0]
        self.weights.ego_weight = 1.0
        self.weights.all_other_weight = 0.5
        self.weights.collision_weight = 1
        self.weights.reg_weight = 10
        self.weights.goal_reaching_weight = 2

        # map encoding parameters
        self.CNN.map_channels = (self.history_num_frames+1)*2+3
        self.CNN.lane_channel = (self.history_num_frames+1)*2
        self.CNN.hidden_channels = [20, 20, 20, 10]
        self.CNN.ROI_outdim = 10
        self.CNN.output_size = self.map_emb_dim
        self.CNN.patch_size = [15, 35, 25, 25]
        self.CNN.kernel_size = [5, 5, 5, 3]
        self.CNN.strides = [1, 1, 1, 1]
        self.CNN.input_size = [224, 224]
        self.CNN.veh_ROI_outdim = 4
        self.CNN.veh_patch_scale = 1.5

        # Multi-modal prediction
        self.M = 1

        self.Discriminator.N_layer_enc = 1

        self.vectorize_map = False
        self.vectorize_lane = True
        self.points_per_lane = 5

        self.try_to_use_cuda = True

        # self.model_params.future_num_frames = 0
        # self.model_params.step_time = 0.2
        self.render_ego_history = False
        # self.model_params.history_num_frames_ego = 0
        # self.model_params.history_num_frames = 0
        # self.model_params.history_num_frames_agents = 0

        self.optim_params.policy.learning_rate.initial = 3e-4  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength


class TransformerGANConfig(TransformerPredConfig):
    def __init__(self):
        super(TransformerGANConfig, self).__init__()
        self.name = "TransformerGAN"
        self.calc_likelihood = True
        self.f_steps = 5
        self.weights.GAN_weight = 0.2
        self.GAN_static = True

        self.optim_params_discriminator.learning_rate.initial = (
            1e-3  # policy learning rate
        )
        self.optim_params_discriminator.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params_discriminator.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params_discriminator.regularization.L2 = (
            0.00  # L2 regularization strength
        )

class DiffuserConfig(AlgoConfig):
    def __init__(self):
        super(DiffuserConfig, self).__init__()
        self.eval_class = "Diffuser"

        self.name = "diffuser"

        # data coordinate
        self.coordinate = 'agent_centric' # ['agent_centric', 'scene_centric']
        self.scene_agent_max_neighbor_dist = 30 # used only when data_centric == 'scene' and coordinate == 'agent'
        
        ## model
        self.map_encoder_model_arch = "resnet18"
        self.diffuser_model_arch = "TemporalMapUnet"

        self.disable_control_on_stationary = 'current_speed' # ['any_speed', 'current_speed'. 'on_lane', False] vehicles (always stationary) will be forced to 0 during rollout (including denoising iteration)
        self.moving_speed_th = 5e-1 # speed threshold for determining if stationary

        # whether ego/neighbor histories are rasterized and given to
        #   the model or they need to be processed by an MLP
        self.rasterized_history = True
        # whether the map will be passed in as conditioning
        #       i.e. do we need a map encoder
        #    this corresponds only to the TRUE map
        self.rasterized_map = True
        # whether to use a "global" map feature (passed in as conditioning to diffuser)
        #   and/or a feature grid (sampled at each trajectory position and concated to trajectory input to diffuser)
        # this is called "map" but actually corresponds to the rasterized feature in general -- may be history
        self.use_map_feat_global = True
        self.use_map_feat_grid = False

        self.base_dim = 32
        self.horizon = 52 # param to control the number of time steps to use for future prediction
        self.n_diffusion_steps = 100
        self.action_weight = 1
        self.diffusor_loss_weights = None
        self.loss_discount = 1 # apply same loss over whole trajectory, don't lessen loss further in future

        # True can only be used when diffuser_input_mode in ['state', 'action'] 
        self.predict_epsilon = False
        self.dim_mults = (2, 4, 8) # (1, 4, 8) # defines the channel size at layers of diffuser convs

        self.clip_denoised = False
        self.loss_type = 'l2'

        #
        # Exponential moving average
        self.use_ema = True
        self.ema_step = 1 #10
        self.ema_decay = 0.995
        self.ema_start_step = 4000 # 2000 -- smaller batch size for real-world data

        # ['concat']
        self.diffuser_building_block = 'concat'

        self.action_loss_only = False

        # ['state', 'action', 'state_and_action', 'state_and_action_no_dyn']
        self.diffuser_input_mode = 'state_and_action'
        # only used when diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']
        self.use_reconstructed_state = False

        # likelihood of not using conditioning as input, even if available
        #       if 1.0, doesn't include cond encoder in arch
        #       if 0.0, conditioning is always used as usual
        # [0, 1] OG paper ablates 0.1-0.5, but 0.1-0.3 is their best
        self.conditioning_drop_map_p = 0.0
        self.conditioning_drop_neighbor_p = 0.0

        # value to fill in when condition is "dropped". Should not just be 0 -- the model
        #   should "know" data is missing, not just think there are e.g. no obstacles in the map.
        #   NOTE: this should be the same as the value given to trajdata to fill in
        #           missing map data
        self.conditioning_drop_fill = 0.5 # -1, 1, and 0 all show up in map or neighbor history



        # the final conditioning feature size after all cond inputs
        #       have been processed together (e.g. map feat, hist feat, state feat...)
        self.cond_feat_dim = 256
        self.curr_state_feat_dim = 64 # only used with rasterized history
        self.map_feature_dim = 256 # TODO seems a bit big, could be contributing to overfit
        self.map_grid_feature_dim = 32
        # TODO bit smaller?
        self.history_feature_dim = 128 # if separate from map
        
        self.history_num_frames = 30 # param to control the number of time steps to use for history
        self.history_num_frames_ego = self.history_num_frames
        self.history_num_frames_agents = self.history_num_frames

        self.future_num_frames = self.horizon
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.loss_weights.diffusion_loss = 1.0

        self.optim_params.policy.learning_rate.initial = 2e-4  # policy learning rate

        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        # self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

        # how many samples to take for validation (during training)
        self.diffuser.num_eval_samples = 10

        # nusc_train_val, hist=10, fut=20
        # self.nusc_norm_info = {
        #     'diffuser': [( 0.970673, 0.000232, 1.080191, 0.000183, -0.016233, 0.000202,  ), ( 2.452948, 0.140067, 2.397183, 0.022983, 0.872437, 0.025260,  )],
        #     'agent_hist': [( -0.453062, 0.000294, 1.089531, 5.839054, 2.054347,  ), ( 1.201422, 0.048892, 2.376539, 1.791524, 0.379787,  )],
        #     'neighbor_hist': [( -0.203397, 1.064796, 0.637057, 1.171413, 0.297763,  ), ( 8.696633, 8.639190, 1.797830, 2.566643, 0.866921,  )],
        # }

        # nusc_train_val, hist=30, fut=52
        self.nusc_norm_info = {
            'diffuser': [( 2.135494, 0.003704, 0.970226, 0.000573, -0.002965, 0.000309,  ), ( 5.544400, 0.524067, 2.206522, 0.049049, 0.729327, 0.023765,  )],
            'agent_hist': [( -1.198923, 0.000128, 0.953161, 4.698113, 2.051664,  ), ( 3.180241, 0.159182, 2.129779, 2.116855, 0.388149,  )],
            'neighbor_hist': [( -0.237441, 1.118636, 0.489575, 0.868664, 0.222984,  ), ( 7.587311, 7.444489, 1.680952, 2.578202, 0.832563,  )],
        }

        # nusc_train_val, hist=40, fut=140
        # self.nusc_norm_info = {
        #     'diffuser': [( 7.897785, -0.026593, 1.400290, 0.000936, -0.002819, 0.000162,  ), ( 15.210303, 2.772619, 2.415771, 0.115524, 0.646058, 0.022165,  )],
        #     'agent_hist': [( -1.819856, 0.005488, 1.533596, 4.942008, 2.056190,  ), ( 3.914807, 0.172819, 2.716692, 1.410469, 0.368531,  )],
        #     'neighbor_hist': [( 1.356429, 2.987132, 0.586324, 1.993453, 0.588986,  ), ( 33.843525, 25.669230, 1.529238, 2.703502, 1.085830,  )],
        # }

        # nuplan, hist=30, fut=52, vehicle
        self.nuplan_norm_info = {
            'diffuser': [( 4.796822, 0.005868, 2.192235, 0.001452, 0.031407, 0.000549,  ), ( 9.048500, 0.502242, 3.416214, 0.081688, 5.372146, 0.074966,  )],
            'agent_hist': [( -2.740177, 0.002556, 2.155241, 3.783593, 1.812939,  ), ( 5.212748, 0.139311, 3.422815, 1.406989, 0.298052,  )],
            'neighbor_hist': [( -0.405011, 1.321956, 0.663815, 0.581748, 0.162458,  ), ( 8.044137, 6.039638, 2.541506, 1.919543, 0.685904,  )],
        }

        # nuplan, hist=30, fut=52, pedestrian
        # self.nuplan_norm_info = {
        #     'diffuser': [( 1.967022, 0.015665, 0.756763, 0.001987, -0.006722, 0.000462,  ), ( 1.855162, 0.414431, 0.547232, 0.339879, 1.421376, 0.419087,  )],
        #     'agent_hist': [( -1.209108, -0.004591, 0.842570, 0.689749, 0.631157,  ), ( 1.169023, 0.189204, 0.490094, 0.198055, 0.158181,  )],
        #     'neighbor_hist': [( 0.225483, 0.344136, 0.112020, 0.107992, 0.054426,  ), ( 7.460583, 5.544016, 0.415600, 0.241769, 0.232373,  )],
        # }

class STRIVEConfig(BehaviorCloningConfig):
    def __init__(self):
        super(STRIVEConfig, self).__init__()
        self.eval_class = "STRIVE"
        self.name = "strive"
        self.map_feature_dim = 256
        self.goal_conditional = False
        self.goal_feature_dim = 32

        # TODO add learnable prior?

        self.vae.latent_dim = 64
        self.vae.num_eval_samples = 10
        self.vae.encoder.rnn_hidden_size = 128
        self.vae.encoder.mlp_layer_dims = (350, 256, 256, 256)
        # self.vae.decoder.rnn_hidden_size = 128
        self.vae.decoder.mlp_layer_dims = (324, 256, 256, 256)

        self.vae.condition_dim = 256
        self.map_feature_dim = 256
        self.history_feature_dim = 128 # if separate from map

        # TODO: tune this
        #       weight schedule?
        self.loss_weights.kl_loss = 1e-4
        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.yaw_reg_loss = 0.0 # 0.1

        self.history_num_frames = 30
        self.history_num_frames_ego = 30
        self.history_num_frames_agents = 30
        self.future_num_frames = 52
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = (324, 256, 256, 128)
        self.decoder.state_as_input = True
        self.decoder.normalization = True # layernorm

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.optim_params.policy.learning_rate.initial = 2e-4  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

        # nusc_train_val, hist=10, fut=20
        self.nusc_norm_info = {
            'agent_hist': [( -0.453062, 0.000294, 1.089531, 5.839054, 2.054347,  ), ( 1.201422, 0.048892, 2.376539, 1.791524, 0.379787,  )],
            'neighbor_hist': [( -0.203397, 1.064796, 0.637057, 1.171413, 0.297763,  ), ( 8.696633, 8.639190, 1.797830, 2.566643, 0.866921,  )],
        }

        # nusc_train_val, hist=30, fut=52
        # self.nusc_norm_info = {
        #     'agent_hist': [( -1.198923, 0.000128, 0.953161, 4.698113, 2.051664,  ), ( 3.180241, 0.159182, 2.129779, 2.116855, 0.388149,  )],
        #     'neighbor_hist': [( -0.237441, 1.118636, 0.489575, 0.868664, 0.222984,  ), ( 7.587311, 7.444489, 1.680952, 2.578202, 0.832563,  )],
        # }

class SceneDiffuserConfig(AlgoConfig):
    def __init__(self):
        super(SceneDiffuserConfig, self).__init__()
        self.eval_class = "SceneDiffuser"

        self.name = "scene_diffuser"

        # data coordinate
        self.coordinate = 'agent_centric' # ['agent_centric', 'scene_centric']
        self.scene_agent_max_neighbor_dist = np.inf # used only when data_centric == 'scene' and self.coordinate == 'agent' and self.neigh_hist_embed_method is not None
        
        ## model
        self.diffuser_model_arch = "SceneTransformer" # ['TemporalMapUnet', 'SceneTransformer']
        
        self.agent_hist_embed_method = 'concat' # ['mlp', 'transformer', 'concat']
        self.neigh_hist_embed_method = 'interaction_edge' # ['mlp', None, 'interaction_edge', 'interaction_edge_and_input']
        self.map_embed_method = 'transformer' # ['cnn', 'cnn_local_patch', 'transformer']
        self.map_encoder_model_arch = "resnet18" # used if rasterized map is used and self.map_embed_method == 'cnn'
        self.interaction_edge_speed_repr = 'rel_vel_new_new' # ['abs_speed', 'rel_vel', 'rel_vel_per_step', 'rel_vel_new', 'rel_vel_new_new']. this field applies only when self.neigh_hist_embed_method = 'interaction_edge'.
        self.single_cond_feat = False # this field is used only when agent_hist_embed_method == 'mlp' or map_embed_method == 'cnn'
        self.normalize_rel_states = True # this field controls if normalizing neighbor states in edges
        self.mask_social_interaction = False # this field is used only when self.diffuser_model_arch=='SceneTransformer' and self.neigh_hist_embed_method != 'interaction_edge'
        self.mask_edge = False # this field applies only when self.neigh_hist_embed_method = 'interaction_edge'. this field controls if edge values will be set to all 0s.
        self.neighbor_inds = [0,1,2,3,4,5,9,10] # [0,1,2,3,4,5] # inds of info used in edge corresponding to (x,y,cos,sin,speed,l,w) or (x,y,cos,sin,velx,vely,l,w,rel_d,rel_d_lw, rel_t_to_collision)
        self.edge_attr_separation = [[0,1],[2,3],[4,5],[6,7]] # [] # (x,y), (cos,sin), rel_d
        self.social_attn_radius = 30.0 # 30.0 # this field applies only when self.neigh_hist_embed_method = 'interaction_edge'. this field controls the radius of the social attention.
        
        self.use_last_hist_step = False # True # this field applies only when self.neigh_hist_embed_method = 'interaction_edge'. this field controls if the last history step of neighbor feature will be used for all neighbor future.
        self.use_noisy_fut_edge = False # this field applies only when mask_edge = False. It controls if the future edge will be noisy or not.
        self.use_const_speed_edge = True # False # this field applies only when self.neigh_hist_embed_method = 'interaction_edge'. this field controls if the neighbor future will be predicted using constant speed motion model.

        self.all_interactive_social = False # this field is used only when self.neigh_hist_embed_method == 'interaction_edge'. If True, all social attention layer will involve interactions. If False, only the first social attention layer will involve interactions.
        self.mask_time = False # whether to mask the future time steps for temporal attention layers
        self.layer_num_per_edge_decoder = 4 # 2 # this field applies only when self.neigh_hist_embed_method = 'interaction_edge'. number of layers per edge decoder
        self.attn_combination = 'gate' # sum # ['sum', 'gate']

        self.disable_control_on_stationary = 'current_speed' # ['any_speed', 'current_speed'. 'on_lane', False] vehicles (always stationary) will be forced to 0 during rollout (including denoising iteration)
        self.moving_speed_th = 5e-1 # speed threshold for determining if stationary

        # whether ego/neighbor histories are rasterized and given to
        #   the model or they need to be processed by an MLP
        self.rasterized_history = False # not used by SceneDiffuser
        # whether the map will be passed in as conditioning
        #       i.e. do we need a map encoder
        #    this corresponds only to the TRUE map
        self.rasterized_map = False
        # whether to use a "global" map feature (passed in as conditioning to diffuser)
        #   and/or a feature grid (sampled at each trajectory position and concated to trajectory input to diffuser)
        # this is called "map" but actually corresponds to the rasterized feature in general -- may be history
        self.use_map_feat_global = False
        self.use_map_feat_grid = False
        if self.map_embed_method == 'cnn':
            self.rasterized_map = True
            self.use_map_feat_global = True
        elif self.map_embed_method == 'cnn_local_patch':
            self.rasterized_map = True     
            self.use_map_feat_grid = True

        self.horizon = 52 # param to control the number of time steps to use for future prediction
        self.n_diffusion_steps = 100
        self.action_weight = 1
        self.diffusor_loss_weights = None
        self.loss_discount = 1 # apply same loss over whole trajectory, don't lessen loss further in future

        # True can only be used when diffuser_input_mode in ['state', 'action'] 
        self.predict_epsilon = False

        self.clip_denoised = False
        self.loss_type = 'l2_with_agent_collision_and_map_collision' # 'l2'

        #
        # Exponential moving average
        self.use_ema = True
        self.ema_step = 1 #10
        self.ema_decay = 0.995
        self.ema_start_step = 4000 # 2000 -- smaller batch size for real-world data

        self.action_loss_only = False

        # ['state', 'action', 'state_and_action', 'state_and_action_no_dyn']
        self.diffuser_input_mode = 'state_and_action'
        # only used when diffuser_input_mode in ['state_and_action', 'state_and_action_no_dyn']
        self.use_reconstructed_state = False

        # during training, likelihood of not using conditioning as input, even if available
        #       if 1.0, doesn't include cond encoder in arch
        #       if 0.0, conditioning is always used as usual
        # [0, 1] OG paper ablates 0.1-0.5, but 0.1-0.3 is their best
        self.conditioning_drop_map_p = 0.0 # 0.05 # not used by SceneDiffuser
        self.conditioning_drop_neighbor_p = 0.0 # 0.05 # not used by SceneDiffuser

        # value to fill in when condition is "dropped". Should not just be 0 -- the model
        #   should "know" data is missing, not just think there are e.g. no obstacles in the map.
        #   NOTE: this should be the same as the value given to trajdata to fill in
        #           missing map data
        # -1, 1, and 0 all show up in map or neighbor history
        self.conditioning_drop_fill = 0.5  # not used by SceneDiffuser



        self.map_feature_dim = 128
        self.map_grid_feature_dim = 32
        self.history_feature_dim = 64 # 32 # if separate from map, this is also used for d_edge
        
        self.history_num_frames = 30 # param to control the number of time steps to use for history
        self.history_num_frames_ego = self.history_num_frames
        self.history_num_frames_agents = self.history_num_frames

        self.future_num_frames = self.horizon
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.loss_weights.diffusion_loss = 1.0
        self.loss_weights.collision_loss = 0.0 # 0.05
        self.loss_weights.offroad_loss = 0.0 # 0.1
        self.loss_weights.history_reconstruction_loss = 0.0

        self.loss_decay_rates = {'collision_decay_rate': 0.9, 'offroad_decay_rate': 0.9}

        self.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate

        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        # self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength

        # how many samples to take for validation (during training)
        self.diffuser.num_eval_samples = 10
        
        # norm_info (x, y, vel, yaw, acc, yawvel)
        # nusc_train_val, hist=10, fut=20
        # self.nusc_norm_info = {
        #     'diffuser': [( 0.970673, 0.000232, 1.080191, 0.000183, -0.016233, 0.000202,  ), ( 2.452948, 0.140067, 2.397183, 0.022983, 0.872437, 0.025260,  )],
        #     'agent_hist': [( -0.453062, 0.000294, 1.089531, 5.839054, 2.054347,  ), ( 1.201422, 0.048892, 2.376539, 1.791524, 0.379787,  )],
        #     'neighbor_hist': [( -0.203397, 1.064796, 0.637057, 1.171413, 0.297763,  ), ( 8.696633, 8.639190, 1.797830, 2.566643, 0.866921,  )],
        # }

        # nusc_train_val, hist=30, fut=52, 3std
        # self.nusc_norm_info = {
        #     'diffuser': [( 4.109492, 0.905238, 1.811630, 0.076673, 0.034290, 0.000544,  ), ( 31.178465, 15.988221, 8.819587, 1.692861, 187.082550, 0.207716,  )],
        #     'agent_hist': [( -4.773609, 1.024253, 1.245693, 3.760632, 1.583485,  ), ( 32.658169, 16.891933, 2.280290, 2.042531, 0.598463,  )],
        # }

        # nusc_train_val, hist=30, fut=52, 3std, dist=50
        # self.nusc_norm_info = {
        #     'diffuser': [( 4.118960, 0.736761, 1.546495, 0.081439, 0.028270, 0.000352,  ), ( 21.324146, 15.658240, 5.361055, 1.652314, 112.881981, 0.190125,  )],
        #     'agent_hist': [( -1.563321, 0.700066, 1.007865, 3.555584, 1.602182,  ), ( 22.261192, 16.344381, 2.042918, 1.662851, 0.512880,  )],
        # }

        # nusc_train_val, hist=30, fut=52 (from diffuser, based on agent-centric, for debugging purposes)
        # self.nusc_norm_info = {
        #     'diffuser': [( 2.135494, 0.003704, 0.970226, 0.000573, -0.002965, 0.000309,  ), ( 5.544400, 0.524067, 2.206522, 0.049049, 0.729327, 0.023765,  )],
        #     'agent_hist': [( -1.198923, 0.000128, 0.953161, 4.698113, 2.051664,  ), ( 3.180241, 0.159182, 2.129779, 2.116855, 0.388149,  )],
        #     'neighbor_hist': [( -0.237441, 1.118636, 0.489575, 0.868664, 0.222984,  ), ( 7.587311, 7.444489, 1.680952, 2.578202, 0.832563,  )],
        # }

        # nusc_train_val, hist=30, fut=52 (based on agent-centric, angle wrap, agent_num=None, dist=30, std=3)
        # self.nusc_norm_info = {
        #     'diffuser': [( 2.258406, -0.004694, 0.993113, 0.000398, -0.004583, 0.000215,  ), ( 5.133034, 0.488767, 2.017744, 0.046498, 0.630069, 0.021352,  )],
        #     'agent_hist': [( -1.279382, 0.003333, 0.999354, 4.285832, 1.923620,  ), ( 2.972631, 0.157705, 1.994173, 1.708093, 0.387984,  )],
        #     'neighbor_hist': [( -0.272701, 1.225347, 0.415054, 0.817593, 0.205175,  ), ( 7.493609, 5.755337, 1.216254, 2.324026, 0.830190,  )],
        # }

        # nusc_train_val, hist=10, fut=20 (based on agent-centric, angle wrap, agent_num=20, dist=50, std=3)
        # self.nusc_norm_info = {
        #     'diffuser': [( 0.963494, -0.001977, 1.084534, 0.000136, -0.016077, 0.000149,  ), ( 2.265510, 0.127576, 2.341157, 0.021109, 0.734371, 0.021819,  )],
        #     'agent_hist_diff': [( -0.401560, 0.001091, 1.304674, -0.000048, -0.046115, 0.000115,  ), ( 1.033356, 0.039461, 3.487783, 0.009608, 41.335037, 0.051111,  )],
        #     'agent_hist': [(  -0.455360, 0.001215, 1.158188, 4.910670, 2.078251,  ), ( 1.145886, 0.044043, 2.435899, 1.316310, 0.352800,  )],
        #     'neighbor_hist': [( 0.264443, 1.500840, 0.637679, 2.389215, 0.917770,  ), ( 18.809500, 15.665614, 1.608743, 2.388980, 0.926426,  )],
        #     'neighbor_fut': [( 0.397639, 1.519584, 0.632787, 1.713392, 0.520402,  ), ( 16.716795, 13.857098, 1.592105, 2.612304, 1.028956,  )]
        # }

        # nusc_train_val, hist=30, fut=52 (based on agent-centric, angle wrap, agent_num=20, dist=50, std=3)
        self.nusc_norm_info = {
            'diffuser': [( 2.308888, -0.004272, 1.030073, 0.000432, -0.004650, 0.000229,  ), ( 5.253208, 0.494499, 2.126455, 0.046915, 0.632508, 0.021437,  )],
            'agent_hist_diff': [( -1.268214, 0.003279, 1.426153, -0.000207, -0.009448, -0.000218,  ), ( 2.970397, 0.152488, 3.195604, 0.027117, 29.687944, 0.054703,  )],
            'agent_hist': [( -1.306818, 0.003479, 1.057708, 4.849725, 2.082991,  ), ( 3.057464, 0.159666, 2.134414, 1.338733, 0.358624,  )],
            'neighbor_hist': [( 0.299714, 1.586693, 0.580586, 1.351683, 0.341308,  ), ( 13.234150, 12.151883, 1.452567, 2.618240, 0.925493,  )],
            'neighbor_fut': [( 0.574813, 1.412446, 0.436985, 0.834127, 0.214287,  ), ( 10.500033, 8.795879, 1.412703, 2.606573, 0.894968,  )]
        }

        # nusc_train_val, hist=30, fut=140
        # self.nusc_norm_info = {
        #     'diffuser': [( 13.874539, 0.773850, 1.895871, 0.062099, 0.023509, -0.000007,  ), ( 34.712517, 17.002407, 4.440071, 1.657297, 75.816147, 0.105079,  )],
        #     'agent_hist': [( -2.107278, 1.210411, 1.394245, 4.797410, 2.089976,  ), ( 34.273563, 17.649628, 2.515871, 1.360063, 0.360934,  )],
        # }

        # nuplan, vehicle, hist=30, fut=52
        self.nuplan_norm_info = {
            'diffuser': [( 4.317393, 0.005299, 2.007901, 0.001517, 0.027003, 0.000730,  ), ( 8.128303, 0.436614, 3.379836, 0.067413, 4.997261, 0.073624,  )],
            'agent_hist_diff': [( -2.366227, 0.002367, 2.254006, -0.000930, -0.009220, 0.000224,  ), ( 4.653337, 0.123767, 4.684658, 0.047780, 54.973598, 0.105584,  )],
            'agent_hist': [( -2.428304, 0.002483, 2.006769, 4.890794, 2.027937,  ), ( 4.738251, 0.130210, 3.472377, 0.534438, 0.129952,  )],
            'neighbor_hist': [( 1.744586, 2.933562, 1.482359, 1.375110, 0.569008,  ), ( 17.750463, 11.847486, 2.611224, 2.349849, 0.865068,  )],
            'neighbor_fut': [( 4.236001, 2.346138, 1.454897, 0.958200, 0.352373,  ), ( 15.373465, 10.320680, 2.627709, 2.209175, 0.796853,  )],
        }

        # nuplan, pedestrian, hist=30, fut=52
        # self.nuplan_norm_info = {
        #     'diffuser': [( 1.967022, 0.015665, 0.756763, 0.001987, -0.006722, 0.000462,  ), ( 1.855162, 0.414431, 0.547232, 0.339879, 1.421376, 0.419087,  )],
        #     'agent_hist': [( -1.209108, -0.004591, 0.842570, 0.689749, 0.631157,  ), ( 1.169023, 0.189204, 0.490094, 0.198055, 0.158181,  )],
        #     'neighbor_hist': [( 0.225483, 0.344136, 0.112020, 0.107992, 0.054426,  ), ( 7.460583, 5.544016, 0.415600, 0.241769, 0.232373,  )],
        # }

        # nuplan, scene, vehicle, hist=30, fut=52, 3std
        # self.nuplan_norm_info = {
        #     'diffuser': [( 9.681753, 4.267665, 1.637228, 0.054537, 0.030122, 0.002374,  ), ( 24.157074, 14.541869, 8.229482, 1.227725, 185.403122, 0.216928,  )],
        #     'agent_hist': [( 3.671831, 5.140579, 2.221056, 1.485775, 0.591026,  ), ( 28.527243, 16.585091, 3.130496, 2.409720, 0.900555,  )],
        # }

        # nuplan, scene, vehicle, hist=10, fut=20, 3std
        # self.nuplan_norm_info = {
        #     'diffuser': [( 8.629487, 5.702340, 2.851898, 0.049617, 0.049150, 0.009598,  ), ( 31.293472, 16.655573, 22.489195, 1.454194, 494.963470, 0.558987,  )],
        #     'agent_hist': [( 4.966475, 6.206179, 2.524934, 3.378294, 1.602678,  ), ( 33.760521, 17.976578, 3.399076, 1.637678, 0.459216,  )],
        # }