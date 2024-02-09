import json
import argparse
from multiprocessing import current_process
import math
import numpy as np
import os
from copy import copy
from pprint import pprint
import torch
import h5py
from trajdata.simulation.sim_stats import calc_stats
import tbsim.utils.tensor_utils as TensorUtils
import pathlib
import glob
import csv
from tbsim.utils.trajdata_utils import load_vec_map

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

agent_keys = ["lon_accel", "lat_accel", "jerk"]
rel_agents_keys = ["rel_lon_accel", "rel_lat_accel", "rel_jerk"]

SIM_DT = 0.1
STAT_DT = 0.5
HIST_POS_MAX = 50
HIST_NUM_BINS = 20
HIST_VEL_MAX = 4
HIST_ACCEL_MAX = 2
HIST_JERK_MAX = 1
# 'ade', 'fde', 
SAVE_STAT_NAMES = ['mean_real_dev', 'all_failure_failure_any',  'all_failure_failure_collision',  'all_failure_failure_offroad', 'all_collision_rate_coll_any', 'all_off_road_rate_rate', 'all_disk_off_road_rate_rate',\
                    'guide_speed_limit', 'guide_target_speed', 'guide_acc_limit', 'guide_agent_collision', 'guide_map_collision_disk', \
                    'guide_global_target_pos', 'guide_social_group', \
                    'guide_global_stop_sign', \
                    'all_disk_collision_rate_coll_any', 'guide_agent_collision_disk', \
                    'all_coverage_onroad', 'all_coverage_success', 'all_coverage_total', \
                    'velocity_dist', 'lon_accel_dist', 'lat_accel_dist', 'jerk_dist', ]
# stats that are variations on the same root, will be collected and saved
COLLECT_SAVE_STAT_NAMES = ['all_sem_layer_rate_', 'all_comfort_', 'emd_']

def parse_single_result(results_dir, eval_out_dir, save_hist_data=False, gt_hist_path=None):
    rjson = json.load(open(os.path.join(results_dir, "stats.json"), "r"))
    cfg = json.load(open(os.path.join(results_dir, "config.json"), "r"))
    print('eval_out_dir', eval_out_dir)
    # eval_out_dir = os.path.join(results_dir, 'eval_out')
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    agg_results = dict()
    scene_results = dict()
    guide_mets = []
    for k in rjson:
        if k == "scene_index":
            scene_results["names"] = rjson[k]
        else:
            if k.split('_')[0] == 'guide':
                # guide metrics must be handled specially (many nans from scene masking)
                met_name = '_'.join(k.split('_')[:-1])
                if met_name in scene_results:
                    # scene_results[met_name] = np.stack([scene_results[met_name], rjson[k]], axis=-1)
                    scene_results[met_name].append(rjson[k])
                else:
                    scene_results[met_name] = [rjson[k]]
                    guide_mets.append(met_name)
            else:
                per_scene = rjson[k]
                scene_results[k] = rjson[k]
                agg_results[k] = per_scene
                if np.sum(np.isnan(rjson[k])) > 0:
                    print("WARNING: metric %s has some nan values! Ignoring them..." % (k))
                rnum = np.nanmean(rjson[k])
                agg_results[k] = rnum
                print("{0} = {1:.4f}".format(k, np.nanmean(rjson[k])))

    num_scenes = len(scene_results["names"])

    for guide_met in guide_mets:
        met_arr = scene_results[guide_met]
        # # TBD: a hack dealing with the case when only 1 scene is evaluated
        # met_arr = np.array(met_arr)
        # if len(met_arr.shape) == 1:
        #     met_arr = met_arr[:, None]
        # valid_inds = np.argmin(np.isnan(met_arr), axis=-1)
        # met_arr = met_arr[np.arange(num_scenes), valid_inds]
        # scene_results[guide_met] = met_arr.tolist()
        # agg_results[guide_met] = np.nanmean(met_arr)
        
        # collect stats across all "scene" versions of guide metrics
        out_met_arr = []
        sc_cnt = [np.argmin(np.isnan(met_arr[si])) for si in range(len(met_arr))]
        while len(out_met_arr) < num_scenes:
            # s0 always valid
            out_met_arr.append(met_arr[0][sc_cnt[0]])
            # find how many scenes were valid in this batch (next non-nan index)
            if sc_cnt[0] + 1 >= len(met_arr[0]):
                # we're at the end
                continue
            nan_mask = np.isnan(met_arr[0][sc_cnt[0]+1:])
            # print('nan_mask', nan_mask)
            if np.sum(nan_mask) == len(nan_mask):
                # all valid
                num_valid = len(nan_mask)
            else:
                num_valid = np.argmin(nan_mask)
            sc_cnt[0] += num_valid + 1
            # collect other valid in batch
            # skip it when there's only 1 scene in the batch
            if len(met_arr) > 1:
                for si in range(num_valid):
                    cur_s = si + 1
                    out_met_arr.append(met_arr[cur_s][sc_cnt[cur_s]])
                    sc_cnt[cur_s] += num_valid + 1 - si

        assert len(out_met_arr) == num_scenes

        scene_results[guide_met] = out_met_arr
        agg_results[guide_met] = np.mean(out_met_arr)
        print("{0} = {1:.4f}".format(guide_met, agg_results[guide_met]))

    print("num_scenes: {}".format(num_scenes))


    # --- estimating wasserstein distances from GT on stats ---
    if args.estimate_dist:
        # histogram of trajectory stats like vel, accel, jerk
        hjson = compute_and_save_hist(os.path.join(results_dir, "data.hdf5"), eval_out_dir, save_hist_data, normalization_by=args.normalization_by, disable_control_on_stationary=args.disable_control_on_stationary)

        if gt_hist_path is None:
            # hist_stats_fn = os.path.join(eval_out_dir, "hist_stats.json")
            # hjson = json.load(open(hist_stats_fn, "r"))
            if cfg["env"] == "l5kit":
                gt_hjson = json.load(open("gt_files/l5kit/GroundTruth/hist_stats.json", "r"))
            elif cfg["env"] == "nusc":
                gt_hjson = json.load(open("gt_files/nusc/GroundTruth/hist_stats.json", "r"))
            elif cfg["env"] == "trajdata":
                # TBD: this is a bit hacky
                if 'lyft' in cfg["trajdata_source_test"][0]:
                    # gt_hjson = json.load(open("gt_files/trajdata_l5/GroundTruth/hist_stats.json", "r"))
                    gt_hjson = json.load(open("gt_files/l5kit/GroundTruth/hist_stats.json", "r"))
                elif 'nusc' in cfg["trajdata_source_test"][0]:
                    gt_hjson = json.load(open("gt_files/trajdata_nusc_new/GroundTruth/hist_stats_new_"+args.normalization_by+"_"+args.disable_control_on_stationary+".json", "r"))
                elif 'nuplan' in cfg["trajdata_source_test"][0]:
                    gt_hjson = json.load(open("gt_files/trajdata_nuplan/GroundTruth/hist_stats_"+args.normalization_by+"_"+args.disable_control_on_stationary+".json", "r"))
                else:
                    raise
            else:
                raise
        else:
            gt_hjson = json.load(open(gt_hist_path, "r"))

        k_d_v_list = []
        int_k_d_v_list = []
        for k in gt_hjson["stats"]:
            k_d = "{}_dist".format(k)
            k_d_v = calc_hist_distance(
                hist1=np.array(gt_hjson["stats"][k]),
                hist2=np.array(hjson["stats"][k]),
                bin_edges=np.array(gt_hjson["ticks"][k][1:])
            )
            agg_results[k_d] = k_d_v
            if k in agent_keys:
                k_d_v_list.append(k_d_v)
                print(k_d, k_d_v)
            elif k in rel_agents_keys:
                int_k_d_v_list.append(k_d_v)
                print(k_d, k_d_v)
        agg_results['mean_real_dev'] = np.mean(k_d_v_list)
        agg_results['mean_real_dev_int'] = np.mean(int_k_d_v_list)
        print('mean_real_dev', np.mean(k_d_v_list))
        print('mean_real_dev_int', np.mean(int_k_d_v_list))
    # ---------------------------------------------------------

    # stats to save
    all_stat_names = copy(SAVE_STAT_NAMES)
    if len(COLLECT_SAVE_STAT_NAMES) > 0:
        for collect_name in COLLECT_SAVE_STAT_NAMES:
            add_stat_names = [k for k in agg_results.keys() if collect_name in k]
            all_stat_names += add_stat_names

    # save csv of per_scene
    with open(os.path.join(eval_out_dir, 'results_per_scene.csv'), 'w') as f:
        csvwrite = csv.writer(f)
        csvwrite.writerow(['scene'] + all_stat_names)
        for sidx, scene in enumerate(scene_results["names"]):
            currow = [scene] + [np.round(scene_results[k][sidx], 4) if k in scene_results else np.nan for k in all_stat_names]
            csvwrite.writerow(currow)

    # save agg csv
    with open(os.path.join(eval_out_dir, 'results_agg.csv'), 'w') as f:
        csvwrite = csv.writer(f)
        csvwrite.writerow(all_stat_names)
        currow = [agg_results[k] if k in agg_results else np.nan for k in all_stat_names]
        csvwrite.writerow(currow)

    return agg_results

def compute_and_save_hist(h5_path, out_path, save_hist_data, normalization_by, disable_control_on_stationary):
    """Compute histogram statistics for a run"""
    h5f = h5py.File(h5_path, "r")
    bins = {
        "velocity": torch.linspace(0, HIST_VEL_MAX, HIST_NUM_BINS+1),
        "lon_accel": torch.linspace(-HIST_ACCEL_MAX, HIST_ACCEL_MAX, HIST_NUM_BINS+1),
        "lat_accel": torch.linspace(-HIST_ACCEL_MAX, HIST_ACCEL_MAX, HIST_NUM_BINS+1),
        "jerk": torch.linspace(-HIST_JERK_MAX, HIST_JERK_MAX, HIST_NUM_BINS+1),

        "rel_pos_norm": torch.linspace(0, HIST_POS_MAX, HIST_NUM_BINS+1),
        "rel_lon_pos_abs": torch.linspace(0, HIST_POS_MAX, HIST_NUM_BINS+1),
        "rel_lat_pos_abs": torch.linspace(0, HIST_POS_MAX, HIST_NUM_BINS+1),
        
        "rel_yaw_abs": torch.linspace(0, np.pi, HIST_NUM_BINS+1),
        "rel_yaw_vel_abs": torch.linspace(0, np.pi, HIST_NUM_BINS+1),

        "rel_vel_norm": torch.linspace(0, HIST_VEL_MAX, HIST_NUM_BINS+1),
        "rel_lon_vel": torch.linspace(-HIST_VEL_MAX, HIST_VEL_MAX, HIST_NUM_BINS+1),
        "rel_lat_vel": torch.linspace(-HIST_VEL_MAX, HIST_VEL_MAX, HIST_NUM_BINS+1),
        
        "rel_accel_norm": torch.linspace(0, HIST_ACCEL_MAX, HIST_NUM_BINS+1),
        "rel_lon_accel": torch.linspace(-HIST_ACCEL_MAX, HIST_ACCEL_MAX, HIST_NUM_BINS+1),
        "rel_lat_accel": torch.linspace(-HIST_ACCEL_MAX, HIST_ACCEL_MAX, HIST_NUM_BINS+1),

        "rel_jerk": torch.linspace(-HIST_JERK_MAX, HIST_JERK_MAX, HIST_NUM_BINS+1),
    }

    sim_stats = dict()
    ticks = None

    dt_ratio = int(math.ceil(STAT_DT / SIM_DT))
    for i, scene_index in enumerate(h5f.keys()):
        scene_data = h5f[scene_index]
        if "map_names" in scene_data:
            # since one scene shares the same map
            map_name = scene_data["map_names"][0][0][0]
            map_name = map_name.decode('utf-8')
            vec_map = load_vec_map(map_name)
        else:
            vec_map = None
        sim_pos = scene_data["centroid"]
        sim_yaw = scene_data["yaw"][:][:, :, None]
        sim_pos = sim_pos[:,::dt_ratio]
        sim_yaw = sim_yaw[:,::dt_ratio]

        num_agents = sim_pos.shape[0]
        sim = calc_stats(positions=torch.Tensor(sim_pos), heading=torch.Tensor(sim_yaw), dt=STAT_DT, bins=bins, disable_control_on_stationary=disable_control_on_stationary, vec_map=vec_map)

        for k in sim:
            if normalization_by == 'scene':
                # NOTE: this normalizes each scene by the num agents
                #          so it's comparing the per-scene distribution, not across the WHOLE dataset
                if k not in sim_stats:
                    if torch.sum(sim[k].hist) == 0.0:
                        sim_stats[k] = sim[k].hist
                    else:
                        sim_stats[k] = (sim[k].hist / torch.sum(sim[k].hist))
                else:
                    if torch.sum(sim[k].hist) == 0.0:
                        sim_stats[k] += sim[k].hist
                    else:
                        sim_stats[k] += (sim[k].hist / torch.sum(sim[k].hist))
            elif normalization_by == 'dataset':
                if k not in sim_stats:
                    sim_stats[k] = sim[k].hist
                else:
                    sim_stats[k] += sim[k].hist
            else:
                raise ValueError("Unknown normalization_by: {}".format(normalization_by))

        if ticks is None or k not in ticks:
            if ticks is None:
                ticks = dict()
            for k in sim:
                ticks[k] = sim[k].bin_edges

    for k in sim_stats:
        if normalization_by == 'scene':
            sim_stats[k] = TensorUtils.to_numpy(sim_stats[k] / len(h5f.keys())).tolist()
        elif normalization_by == 'dataset':
            # normalize by total count to proper distrib
            sim_stats[k] = TensorUtils.to_numpy(sim_stats[k] / torch.sum(sim_stats[k])).tolist()
        else:
            raise ValueError("Unknown normalization_by: {}".format(normalization_by))
    for k in ticks:
        ticks[k] = TensorUtils.to_numpy(ticks[k]).tolist()
    hjson = {"stats": sim_stats, "ticks": ticks}
    if save_hist_data:
        results_path = out_path
        output_file = os.path.join(results_path, "hist_stats.json")
        json.dump(hjson, open(output_file, "w+"), indent=4)
        print("results dumped to {}".format(output_file))
    return hjson



def parse(args):
    all_results = []
    if args.results_set is not None:
        all_results = sorted(glob.glob(os.path.join(args.results_set, '*')))
    elif args.results_dir is not None:
        all_results = [args.results_dir]
    else:
        raise
    print(all_results)

    result_names = [cur_res.split('/')[-1] for cur_res in all_results]
    print(result_names)

    agg_res_list = []
    for result_path in all_results:
        if os.path.isdir(result_path):
            if args.eval_out_dir is None:
                eval_out_dir = os.path.join(result_path, 'eval_out')
            else:
                eval_out_dir = os.path.join(args.eval_out_dir, result_path.split('/')[-1])

            agg_res = parse_single_result(result_path, eval_out_dir, args.save_hist_data, gt_hist_path=args.gt_hist)
            agg_res_list.append(agg_res)
        
    if args.results_set is not None:
        all_stat_names = copy(SAVE_STAT_NAMES)
        if len(COLLECT_SAVE_STAT_NAMES) > 0:
            for collect_name in COLLECT_SAVE_STAT_NAMES:
                add_stat_names = [k for k in agg_res_list[0].keys() if collect_name in k]
                all_stat_names += add_stat_names

        # save agg csv
        if args.eval_out_dir is None:
            eval_out_dir = args.results_set
        else:
            eval_out_dir = args.eval_out_dir
        with open(os.path.join(eval_out_dir, 'results_agg_set.csv'), 'w') as f:
            csvwrite = csv.writer(f)
            csvwrite.writerow(['eval_name'] + all_stat_names)
            for res_name, res_agg in zip(result_names, agg_res_list):
                currow = [np.round(res_agg[k], 4) if k in res_agg else np.nan for k in all_stat_names]
                csvwrite.writerow([res_name] + currow)


'''
calculate wasserstein distance between two histograms
'''
def calc_hist_distance(hist1, hist2, bin_edges, normalize_hist=True):
    from pyemd import emd
    bins = np.array(bin_edges)
    bins_dist = np.abs(bins[:, None] - bins[None, :])

    
    if normalize_hist == True:
        hist1_norm = hist1 / np.sum(hist1)
        hist2_norm = hist2 / np.sum(hist2)
        hist_dist = emd(hist1_norm, hist2_norm, bins_dist)
        # print('hist1_norm: {}'.format(hist1_norm))
        # print('hist2_norm: {}'.format(hist2_norm))
    else:
        hist_dist = emd(hist1, hist2, bins_dist)
        # print('hist1: {}'.format(hist1))
        # print('hist2: {}'.format(hist2))
    
    return hist_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="A directory of results files (including config.json and stats.json)"
    )

    parser.add_argument(
        "--results_set",
        type=str,
        default=None,
        help="A directory of directories where each contained directory is a results_dir to evaluate"
    )
    
    parser.add_argument(
        "--eval_out_dir",
        type=str,
        default=None,
        help="A directory to save evaluation results"
    )

    parser.add_argument(
        "--estimate_dist",
        default=False,
        action="store_true",
        help="Estimate wasserstein distance on stats between the evaluation results and the GT results."
    )

    parser.add_argument(
        "--save_hist_data",
        default=False,
        action="store_true",
        help="Estimate wasserstein distance on stats between the evaluation results and the GT results."
    )

    parser.add_argument(
        "--gt_hist",
        type=str,
        default=None,
        help="Path to histogram stats for GT data if wanting to compute EMD metrics"
    )

    parser.add_argument(
        "--normalization_by",
        type=str,
        default='scene',
        help="'scene', 'dataset'"
    )

    parser.add_argument(
        "--disable_control_on_stationary",
        type=str,
        default='current_speed',
        help="'current_speed', 'any_speed', 'on_lane', None"
    )

    args = parser.parse_args()

    parse(args)
