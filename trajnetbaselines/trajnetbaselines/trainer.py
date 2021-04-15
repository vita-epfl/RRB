import argparse
import datetime
import logging
import os
import pdb
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import trajnettools
from torch.utils.tensorboard import SummaryWriter

from . import __version__ as VERSION
from . import augmentation
from .utils import *
from .loss import gaussian_2d
from .mlp import EDN, EDN_M, RRB, RRB_M
from .predictor import Predictor
from .scene_funcs.scene_funcs import scene_funcs, scene_preprocess, nearest_point_on_road

scene_loss = 0
use_mpc = 0
sampled = 0


class Trainer(object):
    def __init__(self, timestamp, model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None, n_obs=9, n_pred=12, scale=1, batch_size=32, scene_mode='only_traj'):
        self.model = model if model is not None else LSTM()
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 15))

        self.runs_dir = "runs/{}".format(timestamp)
        self.writer = SummaryWriter(self.runs_dir)
        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.scale = 1
        self.batch_size = batch_size
        self.scene_funcs = scene_funcs(device=device).to(device)
        self.scene_mode = scene_mode
        self.resampling_dim = (38, 74)
        self.model.resampling_dim = self.resampling_dim
        files = os.listdir("./center_lines/")
        self.center_line = {}
        self.gaussian_loss = gaussian_2d().to(self.device)
        for i in files:
            with open("./center_lines/" + i, "rb") as fp:  # Unpickling
                self.center_line[i] = torch.from_numpy(pickle.load(fp)).to(device=self.device)

    def loop(self, train_scenes, val_scenes, out, epochs=35, start_epoch=0):  # for loop of training in different epochs
        train_scenes = self.rearrange_scenes(train_scenes, shuffle=1)
        val_scenes = self.rearrange_scenes(val_scenes, shuffle=0)
        for epoch in range(start_epoch, start_epoch + epochs):
            self.train(train_scenes, epoch, out)  # train the model
            self.val(val_scenes, epoch)
            if (epoch % 2 == 0 and wholedata):
                Predictor(self.model, self.n_obs, self.n_pred, scale=self.scale).save(out + '.epoch{}'.format(epoch))
        Predictor(self.model, self.n_obs, self.n_pred, scale=self.scale).save(out)  # saves the final model

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def rearrange_scenes(self, scenes, shuffle):
        """Dividing train_scenes into different sublists, each containing tracks of the same scene, then shuffling
        each seperately """
        scenes_lists = [[] for x in range(int(len(scenes) / self.batch_size))]
        previous_scene_name = scenes[0][0]
        idx = 0
        list_id = 0
        while (idx < len(scenes)):
            count = 0
            flag = 1
            temp_list = []
            while count < self.batch_size and idx < len(scenes):
                if scenes[idx][0] == previous_scene_name:
                    temp_list.append(scenes[idx])
                    count += 1
                    idx += 1
                else:
                    flag = 0
                    previous_scene_name = scenes[idx][0]
                    idx += 1
                    break
            if flag and count == self.batch_size:  # if we made a complete list
                scenes_lists[list_id] = temp_list
                list_id += 1
        # check if we have empty lists at the end
        num_left_lists = int(len(scenes) / self.batch_size) - list_id
        if num_left_lists > 0:
            for i in range(num_left_lists):
                del (scenes_lists[list_id])

        # Shuffling tracks inside each list and also between lists
        if shuffle:
            for i in range(len(scenes_lists)):
                random.shuffle(scenes_lists[i])
            random.shuffle(scenes_lists)
        return scenes_lists

    def select_closest_neighbors(self, scene, n_near_by_agents):
        temp = torch.tensor(scene, device=self.device)
        dist = (temp[self.n_obs - 1, 0] - temp[self.n_obs - 1]).pow(2).sum(dim=1)
        k = min(dist.size(0), n_near_by_agents)
        dist[dist != dist] = 1000000
        closest_indexes = torch.topk(dist, k=k, dim=0,
                                     largest=False).indices  # find 4 smallest values, the first is the veh of interest, so let's keep others.
        closest_nghbrs = torch.gather(temp, 1, closest_indexes.unsqueeze(0).unsqueeze(2).repeat(temp.size(0), 1, 2))
        return closest_nghbrs

    def train(self, scenes, epoch, out):
        start_time = time.time()
        print('epoch', epoch)
        self.lr_scheduler.step()
        epoch_loss = 0.0
        epoch_loss_prob = 0.0
        epoch_loss_traj_nn = 0.0
        epoch_loss_traj_kd = 0.0
        epoch_loss_speed = 0.0
        self.model.train()  # Sets the module in training mode. (Just changes self.training to true)(it will return "normal" variable instead of position)
        n_scenes = len(scenes) * self.batch_size
        n_near_by_agents = 15  # maximum num of near-by_agents to be considered
        scene = torch.zeros([self.batch_size, self.n_pred + self.n_obs, n_near_by_agents, 2],
                            device=self.device)  # size =  (n_batches, n_pred+n_obs, near-by_agents,2(x,y))
        self.model_process = 0

        nb_iterations = len(scenes)
        for iteration in range(
                nb_iterations):  # in the case of non-devidable n_scenes by batch_size, ignore left scenes in the end
            scene_start = time.time()
            file_name = []
            sample_rate = []
            scene.fill_(
                0)  # to remove numbers of previous iteration (maybe num nearby agents is less then the previous time and previous ones stay)
            start_idx = iteration * self.batch_size
            for i in range(self.batch_size):  # store batch of scenes in scene which is a numpay array
                neighbors = self.select_closest_neighbors(scenes[iteration][i][2], n_near_by_agents)
                scene[i, :, :neighbors.size(1), :] = neighbors
                file_name.append(scenes[iteration][i][0])  # save file_name of each scene
                sample_rate.append(scenes[iteration][i][3])  # save file_name of each scene

            preprocess_time = time.time() - scene_start
            loss_dict = self.train_batch(scene, file_name, sample_rate, epoch=epoch, draw_batch=(iteration == 0),
                                         iterations=iteration)
            epoch_loss += loss_dict['loss']
            epoch_loss_prob += loss_dict['loss_prob']
            epoch_loss_traj_nn += loss_dict['loss_traj_nn']
            epoch_loss_traj_kd += loss_dict['loss_traj_kd']
            epoch_loss_speed += loss_dict['loss_speed']
            total_time = time.time() - scene_start
            if iteration % 100 == 0 and use_mpc:
                Predictor(self.model, self.n_obs, self.n_pred, scale=self.scale).save(
                    out + '.epoch{}'.format(epoch))  # saves the model
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': iteration * self.batch_size, 'n_batches': n_scenes,
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.lr(),
                    'loss': round(loss_dict['loss'] / self.batch_size, 3),
                })
            if iteration % 100 == 0:
                print({
                    'type': 'train',
                    # 'epoch': epoch, 'batch': scene_i, 'n_batches': n_scenes,
                    'epoch': epoch, 'batch': iteration * self.batch_size, 'n_batches': n_scenes,
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.lr(),
                    'loss': round(loss_dict['loss'] / self.batch_size, 3),
                })
        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1, 'n_batches': n_scenes,
            'lr': self.lr(),
            'loss': round(epoch_loss / n_scenes, 3),
            'loss_class': round(epoch_loss_prob / n_scenes, 3),
            'loss_traj_nn': round(epoch_loss_traj_nn / n_scenes, 3),
            'loss_traj_kd': round(epoch_loss_traj_kd / n_scenes, 3),
            'loss_speed': round(epoch_loss_speed / n_scenes, 3),
            'tot-time': round(time.time() - start_time, 1),
            'model-proc-time': round(self.model_process, 1),
        })
        self.writer.add_scalar("train/loss", round(epoch_loss / n_scenes, 3), epoch)
        self.writer.add_scalar("train/loss_class", round(epoch_loss_prob / n_scenes, 3), epoch)
        self.writer.add_scalar("train/loss_traj_kd", round(epoch_loss_traj_kd / n_scenes, 3), epoch)
        self.writer.add_scalar("train/loss_traj_nn", round(epoch_loss_traj_nn / n_scenes, 3), epoch)
        self.writer.add_scalar("train/loss_speed", round(epoch_loss_speed / n_scenes, 3), epoch)
        self.writer.add_scalar("train/learning_rate", self.lr(), epoch)
        self.writer.add_scalar("train/time", round(time.time() - start_time, 1), epoch)

    def val(self, val_scenes, epoch):
        val_loss_minl2_nn_hi = 0.0  # highway
        val_loss_minl2_nn_ro = 0.0  # roundabout
        val_loss_minl2_nn_in = 0.0  # intersection
        n_ro = 1
        n_hi = 1
        n_in = 1
        road_viol = 0.0
        val_loss_l2_kd = 0.0
        val_loss_minl2_kd_in = 0.0
        val_loss_minl2_kd_hi = 0.0
        val_loss_minl2_kd_ro = 0.0
        eval_start = time.time()
        self.model.train()  # Sets the module in training mode. # Sven's comment: so that it does not return positions but still normals
        n_scenes = len(val_scenes) * self.batch_size
        n_near_by_agents = 15  # maximum num of near-by_agents to be considered
        scene = torch.zeros([self.batch_size, self.n_pred + self.n_obs, n_near_by_agents, 2],
                            device=self.device)  # size =  (n_batches, n_pred+n_obs, near-by_agents,2(x,y))

        nb_iterations = len(val_scenes)
        draw_batch_rnd = int(np.random.rand() * nb_iterations)
        for iteration in range(
                nb_iterations):  # in the case of non-devidable n_scenes by batch_size, ignore left scenes in the end
            scene.fill_(0)
            file_name = []
            sample_rate = []
            start_idx = iteration * self.batch_size
            for i in range(self.batch_size):  # store batch of scenes in scene which is a numpay array
                neighbors = self.select_closest_neighbors(val_scenes[iteration][i][2], n_near_by_agents)
                scene[i, :, :neighbors.size(1), :] = neighbors
                file_name.append(val_scenes[iteration][i][0])  # save file_name of each scene
                sample_rate.append(val_scenes[iteration][i][3])  # save file_name of each scene

            loss_dict = self.val_batch(scene, file_name, sample_rate, epoch, draw_batch=(iteration == draw_batch_rnd))
            val_loss_l2_kd += loss_dict['loss_l2_kd']
            road_viol += loss_dict['road_viol']
            if ('Roundabout' in file_name[0]):
                val_loss_minl2_kd_ro += loss_dict['loss_minl2_kd']
                val_loss_minl2_nn_ro += loss_dict['loss_minl2_nn']
                n_ro += self.batch_size
            elif ('Intersection' in file_name[0]):
                val_loss_minl2_kd_in += loss_dict['loss_minl2_kd']
                val_loss_minl2_nn_in += loss_dict['loss_minl2_nn']
                n_in += self.batch_size
            elif ('Merging' in file_name[0]):
                val_loss_minl2_kd_hi += loss_dict['loss_minl2_kd']
                val_loss_minl2_nn_hi += loss_dict['loss_minl2_nn']
                n_hi += self.batch_size
            else:
                print('uknown scene')
        val_loss_minl2_kd = (val_loss_minl2_kd_ro / n_ro + val_loss_minl2_kd_in / n_in) / 2
        val_loss_minl2_nn = (val_loss_minl2_nn_ro / n_ro + val_loss_minl2_nn_in / n_in) / 2
        eval_time = time.time() - eval_start
        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1, 'n_batches': n_scenes,
            'lr': self.lr(),
            'val_loss_minl2_kd': round(val_loss_minl2_kd, 3),
            'val_loss_l2_kd': round(val_loss_l2_kd / n_scenes, 3),
            'val_loss_minl2_nn': round(val_loss_minl2_nn, 3),
            'road_viol': round(road_viol / n_scenes, 3),
            'time': round(eval_time, 1),
        })
        self.writer.add_scalar("val/val_loss_l2_kd", val_loss_l2_kd / n_scenes, epoch + 1)
        self.writer.add_scalar("val/val_loss_minl2_kd", val_loss_minl2_kd / n_scenes, epoch + 1)
        self.writer.add_scalar("val/val_loss_minl2_nn", val_loss_minl2_nn / n_scenes, epoch + 1)
        self.writer.add_scalar("val/road_viol", road_viol / n_scenes, epoch + 1)
        self.writer.add_scalar("val/learning_rate", self.lr(), epoch + 1)
        self.writer.add_scalar("val/time", eval_time, epoch + 1)

    def train_batch(self, xy, file_name, sample_rate, epoch, draw_batch,
                    iterations):  # xy size = (n_batches, n_pred+n_obs, near-by_agents,2(x,y))
        process_start = time.time()
        scale = float(self.scene_funcs.pixel_scale_dict[file_name[0]])
        offset = self.scene_funcs.offset_dict[file_name[0]]
        xy_copy = xy.clone()
        xy_copy[:, :, :, 1] = scale * (
                xy[:, :, :, 1] - offset[0])  # second dimension is the longer axes, horizontal one
        xy_copy[:, :, :, 0] = -scale * (xy[:, :, :, 0] - offset[1])
        rotated_scene, resampled_scene, xy_copy, theta = scene_preprocess(xy_copy, file_name, self.n_obs,
                                                                          self.resampling_dim, self.scene_funcs)
        center_line_rotated = augmentation.rotate_all_path_by_theta(self.center_line[file_name[0] + '.txt'],
                                                                    xy_copy[:, self.n_obs - 1:self.n_obs, 0:1], theta,
                                                                    centerline=1)
        self.optimizer.zero_grad()  # Sets gradients of all model parameters to zero. ( supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step(), )
        pixel_scale = torch.tensor([float(self.scene_funcs.pixel_scale_dict[i]) for i in file_name], device=self.device)

        if (self.model.__class__.__name__ == 'EDN'):
            prediction_v, prediction_p, prob, prediction_speed = self.model(obs=xy_copy[:, :self.n_obs, :, :],
                                                                            prediction_truth=xy_copy[:,
                                                                                             self.n_obs:].clone(),
                                                                            scene=resampled_scene,
                                                                            sample_rate=torch.tensor(sample_rate,
                                                                                                     device=self.device),
                                                                            pixel_scale=pixel_scale,
                                                                            center_line_dict=center_line_rotated,
                                                                            rotated_scene=rotated_scene[0],
                                                                            file_name=file_name, epoch=epoch,
                                                                            margin=self.scene_funcs.return_margin(
                                                                                file_name), iterations=iterations)
            m_nn = l2_dist_train(xy_copy[:, self.n_obs:], prediction_p[:, :, :, :2])
            best_mode_prediction_p = torch.gather(prediction_p, 1,
                                                  m_nn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_pred,
                                                                                                     5)).squeeze(1)
            best_mode_prediction_speed = torch.gather(prediction_speed, 1,
                                                      m_nn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                                         self.n_pred,
                                                                                                         1)).squeeze(1)
            loss_traj = self.gaussian_loss(best_mode_prediction_p / scale, xy_copy[:, self.n_obs:, 0] / scale, epoch)
            loss_traj_kd = 0 * loss_traj
            loss_scene_total = 0 * loss_traj
            loss_prob = 0 * loss_traj
            loss_speed = torch.sum(abs(best_mode_prediction_speed.squeeze(2) - (
                    xy_copy[:, self.n_obs:, 0] - xy_copy[:, self.n_obs - 1:-1, 0]).pow(2).sum(dim=2).pow(
                0.5))) / scale
            if (scene_loss):
                margin = self.scene_funcs.return_margin(file_name)
                min_position = torch.tensor([0, 0], dtype=torch.int32, device=self.device)
                max_position = torch.tensor([1.4 * margin - 1, 1.4 * margin - 1], dtype=torch.int32, device=self.device)
                loss_scene = []
                loss_scene_total = torch.tensor([0.0]).to(self.device)
                for i in range(xy_copy.size()[0]):  # i in batch_size
                    image = rotated_scene[i]
                    centered_pos = best_mode_prediction_p[i, :, :2].detach().type(torch.int32) - xy_copy[
                        i, 5 - 1, 0].type(torch.int32) + margin * 0.7
                    detached_pos_ = centered_pos
                    detached_pos = torch.min(torch.max(detached_pos_, min_position), max_position)
                    if (image[tuple(detached_pos[-1])] == 1 and image[tuple(detached_pos[
                                                                                0])] == -1  # if the last prediction is outside the road, we have to find the closest on-road point
                            and torch.prod(detached_pos_[-1] == torch.min(torch.max(detached_pos_[-1], min_position),
                                                                          max_position))):  # if the last point is not out because of the max or min instead of the scene margins
                        for j, pos in enumerate(best_mode_prediction_p[i]):
                            if (image[tuple(detached_pos[j])] == 1):
                                nearest_onroad_point = nearest_point_on_road(detached_pos[j], image, self.scene_funcs)
                                loss_scene_temp = abs(pos[:2] - nearest_onroad_point.type(torch.float32))
                                # print("This is the scene loss:",loss_scene_temp,'   This is the nearest point on the road:',nearest_onroad_point)
                                loss_scene.append(loss_scene_temp)
                                loss_scene_total = loss_scene_total + torch.sum(loss_scene_temp) / scale
                            else:
                                loss_scene.append(0)
            loss = (1 * loss_traj + 1 * loss_speed + 0.75 * loss_scene_total) / (1 + 1 + 1)
        elif (self.model.__class__.__name__ == 'EDN_M'):
            prediction_v, prediction_p, prob, prediction_speed = self.model(obs=xy_copy[:, :self.n_obs, :, :],
                                                                            prediction_truth=xy_copy[:,
                                                                                             self.n_obs:].clone(),
                                                                            scene=resampled_scene,
                                                                            sample_rate=torch.tensor(sample_rate,
                                                                                                     device=self.device),
                                                                            pixel_scale=pixel_scale,
                                                                            center_line_dict=center_line_rotated,
                                                                            rotated_scene=rotated_scene[0],
                                                                            file_name=file_name, epoch=epoch,
                                                                            margin=self.scene_funcs.return_margin(
                                                                                file_name), iterations=iterations)
            m_nn = l2_dist_train(xy_copy[:, self.n_obs:], prediction_p[:, :, :, :2])
            best_mode_prediction_p = torch.gather(prediction_p, 1,
                                                  m_nn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_pred,
                                                                                                     5)).squeeze(1)
            best_mode_prediction_speed = torch.gather(prediction_speed, 1,
                                                      m_nn.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                                         self.n_pred,
                                                                                                         1)).squeeze(1)
            prob = torch.nn.functional.softmax(prob, dim=1)
            loss_traj = self.gaussian_loss(best_mode_prediction_p / scale, xy_copy[:, self.n_obs:, 0] / scale, epoch)
            loss_traj_kd = 0 * loss_traj
            loss_speed = torch.sum(abs(best_mode_prediction_speed.squeeze(2) - (
                    xy_copy[:, self.n_obs:, 0] - xy_copy[:, self.n_obs - 1:-1, 0]).pow(2).sum(dim=2).pow(
                0.5))) / scale
            loss_prob = 100 * torch.nn.functional.cross_entropy(prob, m_nn, reduction='sum') * (
                    "multimodal" in self.model.__class__.__name__)  # check if prob is estimated by the model

            loss = (1 * loss_traj + loss_prob + loss_speed) / (1 + 1 + 1)
        elif ('RRB' in self.model.__class__.__name__):
            prediction_kd_noMixed, prediction_p, prob, prediction_kd, prediction_speed = self.model(
                obs=xy_copy[:, :self.n_obs, :, :], prediction_truth=xy_copy[:, self.n_obs:].clone(),
                scene=resampled_scene, sample_rate=torch.tensor(sample_rate, device=self.device),
                pixel_scale=pixel_scale, center_line_dict=center_line_rotated, rotated_scene=rotated_scene[0],
                file_name=file_name, epoch=epoch, margin=self.scene_funcs.return_margin(file_name),
                iterations=iterations)
            m_kd = l2_dist_train(xy_copy[:, self.n_obs:], prediction_kd[:, :, :, :2])
            m_kd_noMixed = l2_dist_train(xy_copy[:, self.n_obs:], prediction_kd_noMixed[:, :, :, :2])
            best_mode_prediction_kd = torch.gather(prediction_kd, 1,
                                                   m_kd.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_pred,
                                                                                                      5)).squeeze(1)
            best_mode_prediction_kd_noMixed = torch.gather(prediction_kd_noMixed, 1,
                                                           m_kd_noMixed.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,
                                                                                                                      1,
                                                                                                                      self.n_pred,
                                                                                                                      5)).squeeze(
                1)
            best_mode_prediction_p = torch.gather(prediction_p, 1,
                                                  m_kd_noMixed.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                                             self.n_pred,
                                                                                                             2)).squeeze(
                1)

            # barrier_loss = torch.sum(-torch.log(abs(best_mode_prediction_p)) + -torch.log(1-abs(best_mode_prediction_p)))

            best_mode_prediction_speed = torch.gather(prediction_speed, 1,
                                                      m_kd.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                                         self.n_pred,
                                                                                                         1)).squeeze(1)
            loss_traj_res = self.gaussian_loss(best_mode_prediction_kd_noMixed / scale,
                                               xy_copy[:, self.n_obs:, 0] / scale, epoch)
            loss_traj_kd = self.gaussian_loss(best_mode_prediction_kd / scale, xy_copy[:, self.n_obs:, 0] / scale,
                                              epoch)
            loss_traj = 0 * loss_traj_kd

            loss_speed = torch.sum(abs(best_mode_prediction_speed.squeeze(2) - (
                    xy_copy[:, self.n_obs:, 0] - xy_copy[:, self.n_obs - 1:-1, 0]).pow(2).sum(dim=2).pow(
                0.5))) / scale
            loss_prob = 100 * torch.nn.functional.cross_entropy(prob, m_kd, reduction='sum') * (
                    "multimodal" in self.model.__class__.__name__)  # check if prob is estimated by the model

            if (epoch < 1):
                loss = (loss_traj_kd + loss_traj_res / 2 + loss_prob + loss_speed / 2) / 3
            elif (epoch < 4):
                loss = (loss_traj_kd + loss_traj_res / 2 + loss_prob + loss_speed / 2) / 3
            else:
                loss = (loss_traj_kd + loss_traj_res / 4 + loss_prob + loss_speed) / 3
        loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves
        self.optimizer.step()

        loss_dict = {'loss': loss.item() / self.n_pred, 'loss_prob': loss_prob.item() / self.n_pred,
                     'loss_traj_kd': loss_traj_kd.item() / self.n_pred, 'loss_traj_nn': loss_traj.item() / self.n_pred,
                     'loss_speed': loss_speed.item() / self.n_pred}
        model_process = time.time() - process_start
        self.model_process += model_process
        return loss_dict

    def val_batch(self, xy, file_name, sample_rate, epoch, draw_batch):
        xy_copy = xy.clone()
        scale = float(self.scene_funcs.pixel_scale_dict[file_name[0]])
        offset = self.scene_funcs.offset_dict[file_name[0]]
        xy_copy[:, :, :, 1] = scale * (
                xy[:, :, :, 1] - offset[0])  # second dimension is the longer axes, horizontal one
        xy_copy[:, :, :, 0] = -scale * (xy[:, :, :, 0] - offset[1])
        rotated_scene, resampled_scene, xy_copy, theta = scene_preprocess(xy_copy, file_name, self.n_obs,
                                                                          self.resampling_dim, self.scene_funcs)
        center_line_rotated = augmentation.rotate_all_path_by_theta(self.center_line[file_name[0] + '.txt'],
                                                                    xy_copy[:, self.n_obs - 1:self.n_obs, 0:1], theta,
                                                                    centerline=1)
        process_start = time.time()
        self.model.eval()
        with torch.no_grad():
            pixel_scale = torch.tensor([float(self.scene_funcs.pixel_scale_dict[i]) for i in file_name],
                                       device=self.device)
            if ('RRB' in self.model.__class__.__name__):
                prediction_v, prediction_p, prob, prediction_kd, prediction_speed = self.model(
                    obs=xy_copy[:, :self.n_obs], prediction_truth=xy_copy[:, self.n_obs:].clone(),
                    scene=resampled_scene, sample_rate=torch.tensor(sample_rate, device=self.device),
                    pixel_scale=pixel_scale, center_line_dict=center_line_rotated, )
                prediction_p = prediction_p[:, :, :, :2]
                prediction_kd = prediction_kd[:, :, :, :2]
            elif ('EDN' in self.model.__class__.__name__):
                prediction_v, prediction_p, prob, prediction_speed = self.model(obs=xy_copy[:, :self.n_obs],
                                                                                prediction_truth=xy_copy[:,
                                                                                                 self.n_obs:].clone(),
                                                                                scene=resampled_scene,
                                                                                sample_rate=torch.tensor(sample_rate,
                                                                                                         device=self.device),
                                                                                pixel_scale=pixel_scale,
                                                                                center_line_dict=center_line_rotated, )
                prediction_p = prediction_p[:, :, :, :2]
                prediction_kd = prediction_p * 0

            else:
                prediction_v, prediction_p, prob, prediction_speed = self.model(obs=xy_copy[:, :self.n_obs],
                                                                                prediction_truth=xy_copy[:,
                                                                                                 self.n_obs:].clone(),
                                                                                scene=resampled_scene,
                                                                                sample_rate=torch.tensor(sample_rate,
                                                                                                         device=self.device),
                                                                                pixel_scale=pixel_scale,
                                                                                center_line_dict=center_line_rotated, )
                prediction_kd = prediction_p * 0
            m_kd = torch.argmax(prob, 1)
            m_real = l2_dist_train(xy_copy[:, self.n_obs:], prediction_p)
            m_real_kd = l2_dist_train(xy_copy[:, self.n_obs:], prediction_kd)
            best_mode_prediction_p = torch.gather(prediction_kd, 1,
                                                  m_kd.view(-1, 1, 1, 1).repeat(1, 1, self.n_pred, 2)).squeeze(1)
            best_mode_prediction_p_real = torch.gather(prediction_kd, 1,
                                                       m_real_kd.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                                               self.n_pred,
                                                                                                               2)).squeeze(
                1)[:, :, 0:2]  # only consider mean

            loss_l2_kd = torch.sum(
                torch.sqrt(torch.sum((best_mode_prediction_p - xy_copy[:, -self.n_pred:, 0]) ** 2, 2))) / scale  # L2
            loss_minl2 = torch.sum(torch.sqrt(
                torch.sum((best_mode_prediction_p_real - xy_copy[:, -self.n_pred:, 0]) ** 2, 2))) / scale  # L2
            loss_prob = 1 * torch.nn.functional.cross_entropy(prob, m_real, reduction='sum')
            best_mode_prediction_nn_real = torch.gather(prediction_p, 1,
                                                        m_real.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                                                                             self.n_pred,
                                                                                                             2)).squeeze(
                1)
            loss_minl2_nn = torch.sum(torch.sqrt(
                torch.sum((best_mode_prediction_nn_real - xy_copy[:, -self.n_pred:, 0]) ** 2,
                          2))) / scale  # L2

            prediction_kd_rotated = prediction_kd.clone()
            road_viol = 30
            if (epoch % 5 == 0):
                for i in range(prediction_p.shape[1]):
                    prediction_kd_rotated[:, i] = augmentation.rotate_all_path_by_theta(prediction_kd[:, i],
                                                                                        center=xy_copy[:,
                                                                                               self.n_obs - 1, 0],
                                                                                        theta=-theta,
                                                                                        centerline=0)  # rotate back to original scene
                road_viol = offroad_detector_train(prediction_kd_rotated, file_name[0], self.scene_funcs.image)

        loss_dict = {'loss_l2_kd': loss_l2_kd.item() / self.n_pred, 'loss_minl2_kd': loss_minl2.item() / self.n_pred,
                     'loss_prob': loss_prob.item() / self.n_pred,
                     'loss_minl2_nn': loss_minl2_nn.item() / self.n_pred, 'road_viol': road_viol}
        self.model.train()
        return loss_dict


def main(epochs=35):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--scene-mode',
                        help='Model type to be trained, can be RRB, RRB_M, EDN, EDN_M')
    parser.add_argument('--n_obs', default=9, type=int,
                        help='number of observation frames')
    parser.add_argument('--n_pred', default=12, type=int,
                        help='number of prediction frames')
    parser.add_argument('--train-input-files', type=str,
                        help='glob expression for train input files')
    parser.add_argument('--val-input-files',
                        default='../trajnetdataset/output_3scenes_interaction/val_for_monitoring_training/*.ndjson',
                        help='glob expression for validation input files')
    parser.add_argument('--disable-cuda', default=1, type=int,
                        help='disable CUDA')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled state dictionary before training')

    args = parser.parse_args()
    # set model output file
    timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')
    # rename all previous output files to remove 'active' keyword
    baseAdd = 'output/'
    myList = os.listdir(baseAdd)
    outFiles = [i for i in myList if (i[:6] == 'active')]
    for i in outFiles:
        os.rename(baseAdd + i, baseAdd + i[7:])
    output_dir = 'output/{}_{}.pkl'.format(args.scene_mode, timestamp)
    # configure logging
    from pythonjsonlogger import jsonlogger
    import socket
    import sys
    file_handler = logging.FileHandler(output_dir + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })

    # add args.device
    args.device = torch.device('cpu')
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    print(args.device)
    # read in datasets
    train_scenes = list(trajnettools.load_all(args.train_input_files+'*.ndjson',
                                              sample={'syi.ndjson': 0.0}))
    val_scenes = list(trajnettools.load_all(args.val_input_files+'*.ndjson',
                                            sample={'syi.ndjson': 0.0}))

    print('number of train scenes =' + str(len(train_scenes)))
    for idd, i in enumerate(train_scenes):
        for id, jj in enumerate(i[2]):
            flag = 0
            for k in jj:
                if k[1] == k[1] and k[0] == k[0]:  # it is not nan
                    flag = 1
                    break
            if (flag == 0):
                print("in scene", idd, "frame", id, "all the pos are nan")
    baseAdd = args.train_input_files
    trainFiles = os.listdir(baseAdd)
    baseAdd = args.val_input_files
    valFiles = os.listdir(baseAdd)
    logging.info({'train files are : {trainFiles}'.format(trainFiles=trainFiles)})
    logging.info({'val files are : {valFiles}'.format(valFiles=valFiles)})
    # create model
    if (args.scene_mode == 'EDN'):
        model = EDN(n_obs=args.n_obs, n_pred=args.n_pred, device=args.device)
    elif (args.scene_mode == 'EDN_M'):
        model = EDN_M(n_obs=args.n_obs, n_pred=args.n_pred, device=args.device)
    elif (args.scene_mode == 'RRB'):
        model = RRB(n_obs=args.n_obs, n_pred=args.n_pred, device=args.device)
    elif (args.scene_mode == 'RRB_M'):
        model = RRB_M(n_obs=args.n_obs, n_pred=args.n_pred, device=args.device)
        with open(
                "output/final_models/EDN/EDN_M_sceneGeneralization.pkl.state_dict",
                'rb') as f:
            pretrained_dict = torch.load(f)
        trained_blocks = ['encoder_traj', 'decoder', 'encoder_vehs', 'regressor', 'cnn']
        model.load_state_dict(pretrained_dict, strict=False)
        for i in model.named_parameters():
            for j in trained_blocks:
                if (j in i[0]):
                    i[1].requires_grad = False
    else:
        print("ambigues model type")

    torch.backends.cudnn.enabled = False  # disabled as the function used to rotate scene didn't work without it
    weight_decay = 1e-2
    num_epochs_reduce_lr = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)  # weight_decay=1e-4
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, num_epochs_reduce_lr, gamma=0.5)

    if args.load_state:
        print("Loading Model Dict")
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=False)
        model = model.to(args.device)

    trainer = Trainer(timestamp, model, optimizer=optimizer, device=args.device, n_obs=args.n_obs, n_pred=args.n_pred,
                      lr_scheduler=lr_scheduler, scene_mode=args.scene_mode)
    trainer.loop(train_scenes, val_scenes, output_dir, epochs=args.epochs)
    trainer.writer.close()


if __name__ == '__main__':
    main()
