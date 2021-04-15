#from mpc import mpc
#from mpc.mpc import QuadCost, LinDx, GradMethods
import torch
import numpy as np
import torch.nn as nn
import pdb
from ..scene_funcs.cnn import CNN
from ..scene_funcs.scene_funcs import scene_funcs
from .. import augmentation
import time
from .utils import *
import cv2
import trajnetbaselines

import warnings
warnings.filterwarnings("ignore")

class BicycleModel(nn.Module):
    def __init__(self, scale=None, dt=None):
        super(BicycleModel, self).__init__()

        self.scale = scale
        self.dt = dt

    def forward(self, state, action):
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        u = action
        # pdb.set_trace()
        # print(u.shape)
        # print(state.shape)
        a = torch.clamp(u[:, 0], -self.scale * 8, self.scale * 4)
        psi = torch.clamp(u[:, 1], -45 * 0.017, 45 * 0.017)

        beta = torch.atan(torch.tan(psi) * 1.77 / (1.77 + 1.17))
        new_x_pos = state[:, 0] + state[:, 2] * self.dt * torch.cos(state[:, 3] + beta)
        new_y_pos = state[:, 1] + state[:, 2] * self.dt * torch.sin(state[:, 3] + beta)
        new_v = state[:, 2] + a * self.dt
        new_psi = state[:, 3] + state[:, 2] / (1.77 * self.scale) * self.dt * torch.sin(beta)

        x = torch.cat((new_x_pos.unsqueeze(1), new_y_pos.unsqueeze(1), new_v.unsqueeze(1), new_psi.unsqueeze(1)), dim=1)
        if squeeze:
            x = x.squeeze(0)
        return x

def mpc_function(predicted_trajectory, obs, scale, sample_rate, n_pred):
    pred_traj = predicted_trajectory[:, :, :, :2].cpu()  # to exclude variance when predicting Gauss
    mpc_trajc = pred_traj.view(-1, n_pred, 2)

    dx = BicycleModel(scale=scale[0, 0, 0].item(), dt=sample_rate[0, 0, 0].item())

    n_batch, n_state, n_ctrl, T = mpc_trajc.shape[0], 4, 2, n_pred + 1
    ctrl_penalty = 0.1
    n_sc = n_state + n_ctrl

    start_point = torch.zeros(mpc_trajc.shape[0], 2)
    compteur = 0
    for j in range(pred_traj.shape[0]):
        for k in range(pred_traj.shape[1]):
            start_point[compteur, :] = obs[j, -1, 0, :]
            compteur += 1

    v0 = torch.tensor(
        torch.sqrt((mpc_trajc[:, 0, 0] - start_point[:, 0]) ** 2 + (mpc_trajc[:, 0, 1] - start_point[:, 1]) ** 2) /
        sample_rate[0, 0, 0].item()).unsqueeze(1)
    psi0 = torch.tensor((torch.atan2((mpc_trajc[:, 0, 1] - start_point[:, 1]),
                                     (mpc_trajc[:, 0, 0] - start_point[:, 0])))).unsqueeze(1)

    x0 = start_point[:, 0].unsqueeze(1)
    y0 = start_point[:, 1].unsqueeze(1)
    x_init = torch.cat((x0, y0, v0, psi0), 1)
    output_network = torch.cat((mpc_trajc, torch.zeros(n_batch, 10, 2)), dim=2)
    goal_state = torch.cat((x_init.unsqueeze(1), output_network), dim=1).permute(1, 0, 2).unsqueeze(2)
    goal_weights = torch.Tensor((5., 5., 0., 0.)).repeat(T, n_batch, 1, 1)
    px = -torch.sqrt(goal_weights) * goal_state
    q = torch.cat((goal_weights, ctrl_penalty * torch.ones(n_ctrl).repeat(T, n_batch, 1, 1)), 3)
    p = torch.cat((px, torch.zeros(n_ctrl).repeat(T, n_batch, 1, 1)), 3).squeeze(2).to(device=obs.device)
    Q = (q * torch.eye((n_sc))).to(device=obs.device)

    u_lower = -torch.cat(
        (torch.ones(T, n_batch, 1) * 8 * scale[0, 0, 0].item(), torch.ones(T, n_batch, 1) * 45 * 0.017), dim=2).squeeze(
        2).to(device=obs.device)
    u_upper = torch.cat((torch.ones(T, n_batch, 1) * 4 * scale[0, 0, 0].item(), torch.ones(T, n_batch, 1) * 45 * 0.017),
                        dim=2).squeeze(2).to(device=obs.device)
    # pdb.set_trace()

    # .to(device=obs.device)
    x_pos, _, _ = mpc.MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=30,
        verbose=-1,
        n_batch=n_batch,
        grad_method=GradMethods.AUTO_DIFF,
        # grad_method=GradMethods.FINITE_DIFF,
        exit_unconverged=False,
        detach_unconverged=False,
        backprop=True,
        slew_rate_penalty=0.1,
    )(x_init.to(device=obs.device), QuadCost(Q, p), dx)

    del u_upper, u_lower, Q, p
    mpc_position = x_pos.permute(1, 0, 2)[:, 1:, :2].view(pred_traj.shape[0], pred_traj.shape[1], pred_traj.shape[2], 2)
    # mpc_position = x_pos.permute(1, 0, 2)[:,1:,:2].contiguous().view(pred_traj.shape[0],pred_traj.shape[1],pred_traj.shape[2],2)
    # pdb.set_trace()
    if (pred_traj.shape[3] == 4):  # if it is gaussian
        mpc_position = torch.cat([mpc_position, predicted_trajectory[:, :, :, 2:]], dim=3)

    return mpc_position

def min_distance_2_points(beg_x, beg_y, end_x, end_y, line):
    # min_dist_beg, indice_beg = torch.min(((beg_x - line[0]) ** 2 + (beg_y - line[1]) ** 2) ** 0.5, 0)
    min_dist_beg, indice_beg = torch.min((abs(beg_x - line[0]) + abs(beg_y - line[1])), 0)
    # min_dist_end, indice_end = torch.min(((end_x - line[0][indice_beg.item():]) ** 2 + (end_y - line[1][indice_beg:]) ** 2) ** 0.5, 0)
    min_dist_end, indice_end = torch.min((abs(end_x - line[0][indice_beg.item():]) + abs(end_y - line[1][indice_beg:])),
                                         0)
    if (indice_end + 20 > len(line[0])):
        dist = len(line[0]) - (indice_end + 20 + 1)
    else:
        dist = 0

    return min_dist_beg + min_dist_end, indice_beg - 2, (indice_end + indice_beg) + 20 + dist

def best_road(obs, pred, center_lines, device, k, iteration, epochs=None, num_roads=1, rotated_scene=None, margin=None):
    dist_beg_obs = torch.sum(abs(center_lines - obs[0:2].unsqueeze(0).unsqueeze(2).type(torch.float64)), dim=1)
    dist_end_obs = torch.sum(abs(center_lines - obs[2:].unsqueeze(0).unsqueeze(2).type(torch.float64)), dim=1)
    dist_beg_pred = torch.sum(abs(center_lines - pred[0:2].unsqueeze(0).unsqueeze(2).type(torch.float64)), dim=1)
    dist_end_pred = torch.sum(abs(center_lines - pred[2:].unsqueeze(0).unsqueeze(2).type(torch.float64)), dim=1)
    min_dist_beg_obs, min_dist_end_obs = torch.min(dist_beg_obs, dim=1), torch.min(dist_end_obs, dim=1)
    min_dist_beg_pred, min_dist_end_pred = torch.min(dist_beg_pred, dim=1), torch.min(dist_end_pred, dim=1)

    boolean_obs = torch.ones((min_dist_beg_obs.values.shape[0]), dtype=torch.float64).cuda()
    boolean_pred = torch.ones((min_dist_beg_pred.values.shape[0]), dtype=torch.float64).cuda()
    # l2 = time.time()

    boolean_obs = (min_dist_beg_obs.indices > min_dist_end_obs.indices).type(torch.float64) * 10000 + 1
    boolean_pred = (min_dist_beg_pred.indices > min_dist_end_pred.indices).type(torch.float64) * 10000 + 1
    # l3 = time.time()
    if (num_roads == 1):
        #
        min_dist_obs = (min_dist_beg_obs.values + min_dist_end_obs.values) * boolean_obs
        min_dist_pred = (min_dist_beg_pred.values + min_dist_end_pred.values) * boolean_pred
        sorted_obs, best_centers_obs = torch.sort(min_dist_obs,
                                                  dim=0)  # the starting point is more mportant than the end because we want the vehicle to continue from it's current position
        best_center = torch.min(min_dist_obs * 2 + min_dist_pred, dim=0).indices.item()
        # pdb.set_trace()
    else:
        ###This part should be fixed like the previous bock
        distances_fixMePleaseLikePreviousBlock = (min_dist_beg.values + min_dist_end.values) * boolean
        min_args = torch.argsort(distances,
                                 dim=0)  # the starting point is more mportant than the end because we want the vehicle to continue from it's current position
        best_center = min_args[0].item()
        second_center = min_args[0].item()
        max_of_min_dist_end = min_dist_end.values[best_center]
        for i in min_args:
            if (distances[i] < distances[best_center] + 20 and min_dist_end.values[i] > max_of_min_dist_end):
                second_center = i.item()
                max_of_min_dist_end = min_dist_end.values[i]

    indice_beg_final = min_dist_beg_pred.indices[best_center]

    if (num_roads == 1):
        return (center_lines[best_center][0][indice_beg_final:]), (center_lines[best_center][1][indice_beg_final:])
    return (center_lines[best_center][0][indice_beg_final:]), (center_lines[best_center][1][indice_beg_final:]), (
    center_lines[second_center][0][indice_beg_final:]), (center_lines[second_center][1][indice_beg_final:])

def extract_center_line_of_interset(obsrvation, prediction_p, center_line, device, iteration, epochs, n_pred,
                                    num_roads=1):
    if (num_roads == 1):
        center_lines = torch.zeros((prediction_p.shape[0], prediction_p.shape[1], 600, 2), device=device)
    else:
        center_lines = torch.zeros((prediction_p.shape[0], num_roads, 600, 2), device=device)
    for i in range(int(prediction_p.shape[0])):  # batch size
        for j in range(int(prediction_p.shape[1])):  # pred modes
            # lap1 = time.time()
            pred = torch.tensor([], device=device)
            obs = torch.tensor([], device=device)
            beg_pred_x, beg_pred_y, end_pred_x, end_pred_y = prediction_p[i, j, 0, 0], prediction_p[i, j, 0, 1], \
                                                             prediction_p[i, j, n_pred - 1, 0], prediction_p[
                                                                 i, j, n_pred - 1, 1]
            beg_obs_x, beg_obs_y, end_obs_x, end_obs_y = obsrvation[i, 0, 0, 0], obsrvation[i, 0, 0, 1], obsrvation[
                i, -1, 0, 0], obsrvation[i, -1, 0, 1]

            pred = torch.cat((pred, beg_pred_x.unsqueeze(-1), beg_pred_y.unsqueeze(-1), end_pred_x.unsqueeze(-1),
                              end_pred_y.unsqueeze(-1)))
            obs = torch.cat((obs, beg_obs_x.unsqueeze(-1), beg_obs_y.unsqueeze(-1), end_obs_x.unsqueeze(-1),
                             end_obs_y.unsqueeze(-1)))
            result = best_road(obs, pred, center_line[i], device, i, iteration, epochs, num_roads)
            ##lap2 = time.time()
            if (len(result[0]) > 600):
                center_lines[i, j, :, 0] = result[0][:600]
                center_lines[i, j, :, 1] = result[1][:600]
            else:
                center_lines[i, j, :len(result[0]), 0] = result[0]
                center_lines[i, j, :len(result[0]), 1] = result[1]
            if (num_roads is not 1):
                if (len(result[2]) > 600):
                    center_lines[i, 1, :, 0] = result[2][:600]
                    center_lines[i, 1, :, 1] = result[3][:600]
                else:
                    center_lines[i, 1, :len(result[2]), 0] = result[2]
                    center_lines[i, 1, :len(result[2]), 1] = result[3]
    return center_lines
