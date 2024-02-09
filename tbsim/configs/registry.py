"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.l5kit_config import (
    L5KitTrainConfig,
    L5KitMixedEnvConfig,
    L5KitMixedSemanticMapEnvConfig,
)

from tbsim.configs.nusc_config import (
    NuscTrainConfig,
    NuscEnvConfig
)

from tbsim.configs.trajdata_eupeds_config import (
    EupedsTrainConfig,
    EupedsEnvConfig
)
from tbsim.configs.trajdata_nusc_config import (
    NuscTrajdataTrainConfig,
    NuscTrajdataEnvConfig
)
from tbsim.configs.trajdata_nusc_ped_config import (
    NuscTrajdataPedTrainConfig,
    NuscTrajdataPedEnvConfig
)
from tbsim.configs.trajdata_nusc_all_config import (
    NuscTrajdataAllTrainConfig,
    NuscTrajdataAllEnvConfig
)
from tbsim.configs.trajdata_l5kit_config import (
    L5KitTrajdataTrainConfig,
    L5KitTrajdataEnvConfig
)
from tbsim.configs.trajdata_nuplan_config import (
    NuplanTrajdataTrainConfig,
    NuplanTrajdataEnvConfig
)

from tbsim.configs.trajdata_nuplan_ped_config import (
    NuplanTrajdataPedTrainConfig,
    NuplanTrajdataPedEnvConfig
)

from tbsim.configs.trajdata_nuplan_all_config import (
    NuplanTrajdataAllTrainConfig,
    NuplanTrajdataAllEnvConfig
)

from tbsim.configs.orca_config import (
    OrcaTrainConfig,
    OrcaEnvConfig
)

from tbsim.configs.trajdata_drivesim_config import (
    DriveSimTrajdataTrainConfig,
    DriveSimTrajdataEnvConfig
)

# --- scene-centric ---
from tbsim.configs.trajdata_nusc_scene_config import (
    NuscTrajdataSceneTrainConfig,
    NuscTrajdataSceneEnvConfig
)
from tbsim.configs.trajdata_nuplan_scene_config import (
    NuplanTrajdataSceneTrainConfig,
    NuplanTrajdataSceneEnvConfig
)

from tbsim.configs.algo_config import (
    BehaviorCloningConfig,
    BehaviorCloningECConfig,
    SpatialPlannerConfig,
    BehaviorCloningGCConfig,
    TransformerPredConfig,
    TransformerGANConfig,
    AgentPredictorConfig,
    VAEConfig,
    EBMMetricConfig,
    GANConfig,
    DiscreteVAEConfig,
    TreeVAEConfig,
    OccupancyMetricConfig,
    DiffuserConfig,
    STRIVEConfig,
    SceneDiffuserConfig,
)


EXP_CONFIG_REGISTRY = dict()

EXP_CONFIG_REGISTRY["l5_bc"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="l5_bc",
)

EXP_CONFIG_REGISTRY["l5_gan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=GANConfig(),
    registered_name="l5_gan",
)

EXP_CONFIG_REGISTRY["l5_bc_gc"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=BehaviorCloningGCConfig(),
    registered_name="l5_bc_gc",
)

EXP_CONFIG_REGISTRY["l5_spatial_planner"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="l5_spatial_planner",
)

EXP_CONFIG_REGISTRY["l5_agent_predictor"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="l5_agent_predictor"
)

EXP_CONFIG_REGISTRY["l5_vae"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="l5_vae",
)

EXP_CONFIG_REGISTRY["l5_bc_ec"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=BehaviorCloningECConfig(),
    registered_name="l5_bc_ec",
)

EXP_CONFIG_REGISTRY["l5_discrete_vae"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=DiscreteVAEConfig(),
    registered_name="l5_discrete_vae",
)

EXP_CONFIG_REGISTRY["l5_tree_vae"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=TreeVAEConfig(),
    registered_name="l5_tree_vae",
)

EXP_CONFIG_REGISTRY["l5_transformer"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=TransformerPredConfig(),
    registered_name="l5_transformer",
)

EXP_CONFIG_REGISTRY["l5_transformer_gan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=TransformerGANConfig(),
    registered_name="l5_transformer_gan",
)

EXP_CONFIG_REGISTRY["l5_ebm"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=EBMMetricConfig(),
    registered_name="l5_ebm",
)

EXP_CONFIG_REGISTRY["l5_occupancy"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=OccupancyMetricConfig(),
    registered_name="l5_occupancy"
)

EXP_CONFIG_REGISTRY["l5_diff"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="l5_diff"
)

EXP_CONFIG_REGISTRY["nusc_bc"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="nusc_bc"
)

EXP_CONFIG_REGISTRY["nusc_bc_gc"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=BehaviorCloningGCConfig(),
    registered_name="nusc_bc_gc"
)

EXP_CONFIG_REGISTRY["nusc_spatial_planner"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="nusc_spatial_planner"
)

EXP_CONFIG_REGISTRY["nusc_vae"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="nusc_vae"
)

EXP_CONFIG_REGISTRY["nusc_discrete_vae"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=DiscreteVAEConfig(),
    registered_name="nusc_discrete_vae"
)

EXP_CONFIG_REGISTRY["nusc_tree_vae"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=TreeVAEConfig(),
    registered_name="nusc_tree_vae"
)

EXP_CONFIG_REGISTRY["nusc_diff_stack"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="nusc_diff_stack"
)


EXP_CONFIG_REGISTRY["nusc_agent_predictor"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="nusc_agent_predictor"
)

EXP_CONFIG_REGISTRY["nusc_gan"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=GANConfig(),
    registered_name="nusc_gan"
)

EXP_CONFIG_REGISTRY["nusc_occupancy"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=OccupancyMetricConfig(),
    registered_name="nusc_occupancy"
)

EXP_CONFIG_REGISTRY["nusc_diff"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="nusc_diff"
)

EXP_CONFIG_REGISTRY["eupeds_bc"] = ExperimentConfig(
    train_config=EupedsTrainConfig(),
    env_config=EupedsEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="eupeds_bc"
)

EXP_CONFIG_REGISTRY["eupeds_vae"] = ExperimentConfig(
    train_config=EupedsTrainConfig(),
    env_config=EupedsEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="eupeds_vae"
)

EXP_CONFIG_REGISTRY["orca_bc"] = ExperimentConfig(
    train_config=OrcaTrainConfig(),
    env_config=OrcaEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="orca_bc"
)

EXP_CONFIG_REGISTRY["orca_diff"] = ExperimentConfig(
    train_config=OrcaTrainConfig(),
    env_config=OrcaEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="orca_diff"
)

EXP_CONFIG_REGISTRY["trajdata_nusc_bc"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="trajdata_nusc_bc"
)

EXP_CONFIG_REGISTRY["trajdata_nusc_vae"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="trajdata_nusc_vae"
)

EXP_CONFIG_REGISTRY["trajdata_nusc_spatial_planner"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="trajdata_nusc_spatial_planner"
)

EXP_CONFIG_REGISTRY["trajdata_nusc_agent_predictor"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="nusc_agent_predictor"
)

EXP_CONFIG_REGISTRY["trajdata_nusc_diff"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nusc_diff"
)

EXP_CONFIG_REGISTRY["trajdata_nusc_strive"] = ExperimentConfig(
    train_config=NuscTrajdataTrainConfig(),
    env_config=NuscTrajdataEnvConfig(),
    algo_config=STRIVEConfig(),
    registered_name="trajdata_nusc_strive"
)

EXP_CONFIG_REGISTRY["trajdata_l5_bc"] = ExperimentConfig(
    train_config=L5KitTrajdataTrainConfig(),
    env_config=L5KitTrajdataEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="trajdata_l5_bc"
)

EXP_CONFIG_REGISTRY["trajdata_l5_vae"] = ExperimentConfig(
    train_config=L5KitTrajdataTrainConfig(),
    env_config=L5KitTrajdataEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="trajdata_l5_vae"
)

EXP_CONFIG_REGISTRY["trajdata_l5_spatial_planner"] = ExperimentConfig(
    train_config=L5KitTrajdataTrainConfig(),
    env_config=L5KitTrajdataEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="trajdata_l5_spatial_planner",
)

EXP_CONFIG_REGISTRY["trajdata_l5_agent_predictor"] = ExperimentConfig(
    train_config=L5KitTrajdataTrainConfig(),
    env_config=L5KitTrajdataEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="trajdata_l5_agent_predictor"
)

EXP_CONFIG_REGISTRY["trajdata_l5_diff"] = ExperimentConfig(
    train_config=L5KitTrajdataTrainConfig(),
    env_config=L5KitTrajdataEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_l5_diff"
)

EXP_CONFIG_REGISTRY["nusc_ped_diff"] = ExperimentConfig(
    train_config=NuscTrajdataPedTrainConfig(),
    env_config=NuscTrajdataPedEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="nusc_ped_diff"
)

EXP_CONFIG_REGISTRY["nusc_all_diff"] = ExperimentConfig(
    train_config=NuscTrajdataAllTrainConfig(),
    env_config=NuscTrajdataAllEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nusc_all_diff"
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_bc"] = ExperimentConfig(
    train_config=NuplanTrajdataTrainConfig(),
    env_config=NuplanTrajdataEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="trajdata_nuplan_bc"
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_spatial_planner"] = ExperimentConfig(
    train_config=NuplanTrajdataTrainConfig(),
    env_config=NuplanTrajdataEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="trajdata_nuplan_spatial_planner",
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_agent_predictor"] = ExperimentConfig(
    train_config=NuplanTrajdataTrainConfig(),
    env_config=NuplanTrajdataEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="trajdata_nuplan_agent_predictor"
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_diff"] = ExperimentConfig(
    train_config=NuplanTrajdataTrainConfig(),
    env_config=NuplanTrajdataEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nuplan_diff"
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_ped_diff"] = ExperimentConfig(
    train_config=NuplanTrajdataPedTrainConfig(),
    env_config=NuplanTrajdataPedEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nuplan_ped_diff"
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_all_diff"] = ExperimentConfig(
    train_config=NuplanTrajdataAllTrainConfig(),
    env_config=NuplanTrajdataAllEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nuplan_all_diff"
)

# --- scene-centric ---
EXP_CONFIG_REGISTRY["trajdata_nusc_scene_diff"] = ExperimentConfig(
    train_config=NuscTrajdataSceneTrainConfig(),
    env_config=NuscTrajdataSceneEnvConfig(),
    algo_config=SceneDiffuserConfig(),
    registered_name="trajdata_nusc_scene_diff"
)

EXP_CONFIG_REGISTRY["trajdata_nuplan_scene_diff"] = ExperimentConfig(
    train_config=NuplanTrajdataSceneTrainConfig(),
    env_config=NuplanTrajdataSceneEnvConfig(),
    algo_config=SceneDiffuserConfig(),
    registered_name="trajdata_nuplan_scene_diff"
)

EXP_CONFIG_REGISTRY["trajdata_drivesim_diff"] = ExperimentConfig(
    train_config=DriveSimTrajdataTrainConfig(),
    env_config=DriveSimTrajdataEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_drivesim_diff"
)

def get_registered_experiment_config(registered_name):
    registered_name = backward_compatible_translate(registered_name)

    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()


def backward_compatible_translate(registered_name):
    """Try to translate registered name to maintain backward compatibility."""
    translation = {
        "l5_mixed_plan": "l5_bc",
        "l5_mixed_gc": "l5_bc_gc",
        "l5_ma_rasterized_plan": "l5_agent_predictor",
        "l5_gan_plan": "l5_gan",
        "l5_mixed_ec_plan": "l5_bc_ec",
        "l5_mixed_vae_plan": "l5_vae",
        "l5_mixed_discrete_vae_plan": "l5_discrete_vae",
        "l5_mixed_tree_vae_plan": "l5_tree_vae",
        "nusc_rasterized_plan": "nusc_bc",
        "nusc_mixed_gc": "nusc_bc_gc",
        "nusc_ma_rasterized_plan": "nusc_agent_predictor",
        "nusc_gan_plan": "nusc_gan",
        "nusc_vae_plan": "nusc_vae",
        "nusc_mixed_tree_vae_plan": "nusc_tree_vae",
    }
    if registered_name in translation:
        registered_name = translation[registered_name]
    return registered_name
