"""Command line tool to create a table of evaluations metrics."""
import datetime
import logging
import math
import os
import pdb
import pickle
import sys
import argparse
import numpy as np
import torch
import trajnettools

import trajnetbaselines
from .scene_funcs.scene_funcs import scene_funcs


class Evaluator(object):
    def __init__(self, scenes, file_name, sample_rate):  # nonlinear_scene_index
        self.scenes = scenes
        self.file_name = file_name
        self.sample_rate = sample_rate
        self.average_l2 = {'N': len(scenes)}
        self.final_l2 = {'N': len(scenes)}
        self.cross_track = {'N': len(scenes)}
        self.scene_violations = {'N': len(scenes)}
        self.variance_error = {'N': len(scenes)}
        self.avg_test_time = {'N': len(scenes)}
        self.scene_funcs = scene_funcs()
        self.n_obs = None
        self.n_pred = None

        files = os.listdir("./center_lines/")
        self.center_line = {}
        for i in files:
            with open("./center_lines/" + i, "rb") as fp:  # Unpickling
                self.center_line[i] = torch.from_numpy(pickle.load(fp))

    def aggregate(self, predictor, store_image=0):
        """
        store_image: if 1, means we want to store images of scene and predicted trajectories.
        """
        store_image_stride = 15  # sets how many images are going to be stored. every store_image_stride images, one image will be stored.
        average = 0.0
        nonlinear = 0.0
        final = 0.0
        cross_track = 0.0
        scene_violation = 0.0
        allPredictions = []
        tot_test_time = 0
        l2_error = []
        flag = 1
        cnt = 0
        for scene_i, paths in enumerate(
                self.scenes):  # paths is a list of the pedestrian of interest and other neighbors
            pixel_scale = torch.tensor([float(self.scene_funcs.pixel_scale_dict[self.file_name[scene_i]])])
            store_image_tmp = store_image * (int(scene_i % store_image_stride == 0))

            prediction, test_time, scene_violation_smpl, my_flag = predictor(paths, n_obs=self.n_obs,
                                                                             file_name=self.file_name[scene_i],
                                                                             sample_rate=self.sample_rate[scene_i],
                                                                             pixel_scale=pixel_scale,
                                                                             scene_funcs=self.scene_funcs,
                                                                             store_image=store_image_tmp,
                                                                             center_line=self.center_line)

            allPredictions.append(prediction)
            average_l2 = trajnettools.metrics.average_l2(paths[0], prediction, self.n_pred)
            final_l2 = trajnettools.metrics.final_l2(paths[0], prediction)
            cross_track_l2 = trajnettools.metrics.cross_track(paths[0], prediction, self.n_pred)

            # aggregate

            if (my_flag):
                scene_violation -= scene_violation_smpl
                cnt += 1
            average += average_l2
            scene_violation += scene_violation_smpl
            l2_error.append(average_l2)
            final += final_l2
            cross_track += cross_track_l2
            tot_test_time += test_time

        average /= len(self.scenes) - cnt

        print(cnt, ", cnt normalized", cnt / len(self.scenes))
        final /= len(self.scenes)
        cross_track /= len(self.scenes)
        scene_violation /= len(self.scenes)
        l2_error_pow2 = [(i - average) ** 2 for i in l2_error]
        self.average_l2['model'] = average
        self.final_l2['model'] = final
        self.cross_track['model'] = cross_track
        self.scene_violations['model'] = scene_violation
        self.variance_error['model'] = math.sqrt(sum(l2_error_pow2) / len(l2_error_pow2))
        self.avg_test_time['model'] = tot_test_time / len(self.scenes)
        return

    def result(self):
        return self.average_l2, self.final_l2, self.variance_error, self.avg_test_time, self.scene_violations, self.cross_track  # self.average_l2_nonlinear,


def eval(input_file, predictor):
    print('dataset', input_file)

    sample = 0.05 if 'syi.ndjson' in input_file else None
    reader = trajnettools.Reader(input_file, scene_type='paths')
    file_name = []
    scenes = []
    sample_rate = []
    scene_instance = scene_funcs(device='cpu').to('cpu')
    for files, idx, s, rate in reader.scenes(sample=sample):
        scenes.append(s)
        file_name.append(files)
        sample_rate.append(rate)
    del scene_instance

    # non-linear scenes from high Kalman Average L2
    n_obs = predictor.n_obs
    n_pred = predictor.n_pred

    evaluator = Evaluator(scenes, file_name=file_name, sample_rate=sample_rate)  # nonlinear_scene_index
    evaluator.n_obs = n_obs  # setting n_obs and n_pred values
    evaluator.n_pred = n_pred

    evaluator.aggregate(predictor, store_image=0)
    return evaluator.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-add', default='output/rrb', type=str,
                        help='address to the model to be evaluated')
    parser.add_argument('--data-add', default='../trajnetdataset/output_interaction_completeDataset_unseenscene/val/',
                        help='address to the test data')
    parser.add_argument('--image_add', default='output/',
                        help='address to the folder to store qualitative images')
    args = parser.parse_args()
    # baseAdd = 'output/'
    # baseAdd = 'output/finalModels/mlp_MLP_traj_scene_vehs_singlemodal_speed_gauss/'
    # baseAdd = 'output/finalModels/mlp_MLP_traj_scene_vehs_multimodal_speed_gauss/'
    # baseAdd = 'output/finalModels/mlp_MLP_traj_scene_vehs_multimodal_speed_gauss_rho/'
    # baseAdd = 'output/finalModels/'
    # baseAdd = 'output/finalModels/RRB_mixed/'
    # baseAdd = 'output/finalModels/ablation/'
    # baseAdd = 'output/finalModels/mlp_MLP_traj_scene_vehs_singlemodalCenter_speed_gauss/'
    model = args.model_add+'.pkl'
    predictor = trajnetbaselines.Predictor.load(model)

    # baseAddress = '../trajnetdataset/output_interaction_crossscene_10pred_sampled/train/'
    # baseAddress = '../trajnetdataset/output_interaction_completeDataset_unseenscene/val/'
    # baseAddress = '../trajnetdataset/output_interaction_completeDataset_unseenscene/val_for_monitoring_training_visual/'
    # baseAddress = '../trajnetdataset/output_interaction_completeDataset/val_for_monitoring_training///'
    # baseAddress = '../trajnetdataset/output_interaction_completeDataset/val/'
    # baseAddress = '../trajnetdataset/output_interaction_crosstrack_10pred_sampled/val/'
    # baseAddress = '../trajnetdataset/output_3scenes_interaction/val/'
    list_data = os.listdir(args.data_add)
    test_files = [i for i in list_data]
    datasets = [args.data_add + i for i in test_files]
    results = {dataset
                   .replace('data/', '')
                   .replace('.ndjson', ''): eval(dataset, predictor)
               for i, dataset in enumerate(datasets)}

    ADE = np.zeros([3])
    FDE = np.zeros([3])
    Num = np.zeros([3])
    Scene_viol = np.zeros([3])
    cross_track = np.zeros([3])
    for i in datasets:
        N = results[i[:-7]][0]['N']
        if ('Roundabout' in i):
            ADE[0] += results[i[:-7]][0]['model'] * N
            FDE[0] += results[i[:-7]][1]['model'] * N
            Num[0] += N
            Scene_viol[0] += results[i[:-7]][-2]['model'] * N
            cross_track[0] += results[i[:-7]][-1]['model'] * N
        elif ('Intersection' in i):
            ADE[1] += results[i[:-7]][0]['model'] * N
            FDE[1] += results[i[:-7]][1]['model'] * N
            Num[1] += N
            Scene_viol[1] += results[i[:-7]][-2]['model'] * N
            cross_track[1] += results[i[:-7]][-1]['model'] * N
        elif ('Merging' in i):
            ADE[2] += results[i[:-7]][0]['model'] * N
            FDE[2] += results[i[:-7]][1]['model'] * N
            Num[2] += N
            Scene_viol[2] += results[i[:-7]][-2]['model'] * N
            cross_track[2] += results[i[:-7]][-1]['model'] * N

    ADE = ADE / Num
    FDE = FDE / Num
    Scene_viol = Scene_viol / Num
    cross_track = cross_track / Num
    print('Roundabout', 'Intersection', 'Merging')
    print(ADE)
    print(FDE)
    print(Scene_viol)
    print(cross_track)
    print('average values:')
    print(np.sum(ADE) / 3)
    print(np.sum(FDE) / 3)
    print(np.sum(Scene_viol) / 3)
    print(np.sum(cross_track) / 3)

if __name__ == '__main__':
    main()
