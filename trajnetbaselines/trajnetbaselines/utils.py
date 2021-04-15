import numpy as np
import torch
import cv2
import torch.nn.functional as F
import trajnettools
from casadi import *
from . import augmentation
from .scene_funcs.scene_funcs import scene_preprocess


def l2_dist_train(prediction_truth, prediction_p):
    # pdb.set_trace()
    l2_dist = (prediction_truth[:, :, 0].unsqueeze(1).repeat(1, prediction_p.size(1), 1, 1) - prediction_p).pow(
        2).sum(dim=3).sum(dim=2)
    _, indices = l2_dist.min(dim=1)  # ground_truth best mode (batch-wise)
    return indices


def offroad_detector_train(prediction, file_name, scene_info, prob=None):
    """It doesn't need to be devided by 10 since the scene is read from numpy file not the image """
    prediction_clone = prediction.clone()
    cnt = 0
    n_batches = prediction_clone.size(0)
    for k in range(n_batches):
        for modes in prediction_clone[k]:  # between different modes predicted
            for i, seq in enumerate(modes):  # between different modes predicted
                # for seq in modes: #between n_obs + n_pred data points that we have
                if (seq[0] >= scene_info[file_name].size(0) or seq[1] >= scene_info[file_name].size(1)):
                    cnt += 1  # Since the speed varies and margin is not enough is some cases  #cnt[i] += 1
                elif (scene_info[file_name][(seq[0].type(torch.LongTensor)), (seq[1].type(torch.LongTensor))] == 1):
                    cnt += 1
    return cnt / prediction_clone.size(1)  # devide by number of modes


def offroad_detector(prediction, file_name, scene_info, prob=None):
    """It doesn't need to be devided by 10 since the scene is read from numpy file not the image """
    prediction_clone = prediction.clone()
    cnt = np.zeros([prediction_clone.size(0)])
    for i, modes in enumerate(prediction_clone):  # between different modes predicted
        for seq in modes:  # between n_obs + n_pred data points that we have
            # if(scene_info[file_name][(seq[0].astype(int)[0],seq[0].astype(int)[1])]==1):
            if seq[0] >= scene_info[file_name].size(0) or seq[1] >= scene_info[file_name].size(1):
                cnt[i] += 1  # Since the speed varies and margin is not enough is some cases  #cnt[i] += 1
            elif scene_info[file_name][(seq[0].type(torch.LongTensor)), (seq[1].type(torch.LongTensor))] == 1:
                cnt[i] += 1

    if (prob is None):
        return np.sum(cnt)
    return np.sum(prob * cnt)


def draw_scene(scene_funcs, xy, file_name, rotated_scene, n_obs, n_pred, prob, outputs, rotation_enabled,
               ped_id, first_frame, model_name, prediction_nn, prediction_kd):
    """ This function stores the tracks and images to evaluate the performance qualitatively.
    """
    scale = 1
    if ('DJI' in file_name):
        scale = 5
    xy = xy.clone() // scale
    outputs = outputs.clone() // scale
    prediction_nn = prediction_nn.clone() // scale
    prediction_kd = prediction_kd.clone() // scale
    image = scene_funcs.scene_data(xy[0, n_obs - 1, 0, :].view(1, 2), resampling=False, file_name=[file_name])[0]
    cor1, cor4 = scene_funcs.scene_data(xy[0, n_obs - 1, 0, :], file_name=[file_name], find_cor=True)
    if (rotation_enabled):
        rotated_scene_cpu_ = rotated_scene[0].cpu().numpy()
    else:
        rotated_scene_cpu_ = image.cpu().numpy()

    (dim1, dim2) = np.shape(rotated_scene_cpu_)
    rotated_scene_cpu = np.ones([dim1, dim2, 3])
    rotated_scene_cpu[:, :, 0] = rotated_scene_cpu_

    if (rotation_enabled):
        margin = np.shape(rotated_scene_cpu)[0] // 2
    else:
        margin = ((cor4 - cor1) // 2).type(torch.float32)[0]
    margin = np.shape(rotated_scene_cpu)[0] // 2
    # observation => what is used to predict
    for i in range(n_obs - 1):
        xy_cpu = xy.cpu()
        aa = tuple(xy_cpu[0, i, 0, :] - xy_cpu[0, n_obs - 1, 0, :] + margin)
        cv2.circle(rotated_scene_cpu, (aa[1], aa[0]), 5, (0, -1, 0), -1)  # green => observation
        cv2.rectangle(rotated_scene_cpu, (aa[1] - 4, aa[0] - 4), (aa[1] + 4, aa[0] + 4), (0, -1, 0),
                      -1)  # green => observation
    aa = tuple(xy_cpu[0, i + 1, 0, :] - xy_cpu[0, n_obs - 1, 0, :] + margin)
    # cv2.circle(rotated_scene_cpu,(aa[1],aa[0]), 10, (0,-1,0), -1)  # green => observation
    cv2.rectangle(rotated_scene_cpu, (aa[1] - 10, aa[0] - 10), (aa[1] + 10, aa[0] + 10), (0, -1, 0),
                  2)  # green => observation
    for i in range(xy_cpu.shape[2] - 1):  # drawing other agents
        veh1 = xy_cpu[0, n_obs - 1, i + 1, :]
        if (veh1[0] == veh1[0] and veh1[1] == veh1[1]):
            aa = tuple(veh1 - xy_cpu[0, n_obs - 1, 0, :] + margin)
            cv2.rectangle(rotated_scene_cpu, (aa[1] - 10, aa[0] - 10), (aa[1] + 10, aa[0] + 10), (0, -1, 0),
                          2)  # green => observation

    colors = [(-1, -1, 0), (0, -1, -1), (-1, 0, -1),
              (-1, -1, -1)]  # up to four modes, feel free to add colors if you want :)
    placements = [(0, 20), (0, 40), (0, 60), (0, 80)]  # positions for the texts, same as above :)

    cv2.putText(rotated_scene_cpu, 'Ground truth', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    # .... , white
    size_decrease = 0  # descrese size of prediction to better see mode collapse...
    for m in range(outputs.shape[0]):  # outputs.shape[0] corresponds to the number of modes
        if (2 > 1):
            m = 2
            pred_m = outputs[0]
            color = colors[m]
            placement = placements[m]
            for i in range(n_pred):
                out = pred_m.cpu()
                aa = tuple(out[i, :] - xy_cpu[0, n_obs - 1, 0, :] + margin)
                cv2.circle(rotated_scene_cpu, (aa[1], aa[0]), 5, color, -1)

            cv2.putText(rotated_scene_cpu, 'RRB', placement, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            size_decrease -= 1
            m = 1
            pred_m = prediction_nn[0]
            color = colors[m]
            placement = placements[m]
            for i in range(n_pred):
                out = pred_m.cpu()
                aa = tuple(out[i, :] - xy_cpu[0, n_obs - 1, 0, :] + margin)
                cv2.circle(rotated_scene_cpu, (aa[1], aa[0]), 5, color, -1)

            cv2.putText(rotated_scene_cpu, 'EDN', placement, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            size_decrease -= 1
            m = 0
            pred_m = prediction_kd[0]
            color = colors[m]
            placement = placements[m]
            for i in range(n_pred):
                out = pred_m.cpu()
                aa = tuple(out[i, :] - xy_cpu[0, n_obs - 1, 0, :] + margin)
                cv2.circle(rotated_scene_cpu, (aa[1], aa[0]), 5, color, -1)

            cv2.putText(rotated_scene_cpu, 'KD', placement, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            size_decrease -= 1

    # ground truth prediction
    for i in range(n_pred):
        aa = tuple(xy_cpu[0, i + n_obs, 0, :] - xy_cpu[0, n_obs - 1, 0, :] + margin)
        cv2.circle(rotated_scene_cpu, (aa[1], aa[0]), 4, (0, 0, 0), -1)  # black => ground truth

    # save to disk
    import os
    directory = '/output/generated_pics/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory + 'results_' + str(
        file_name) + '_' + str(ped_id) + '-' + str(first_frame) + '_' + model_name + '.jpg',
                rotated_scene_cpu * (-256))


def l2_dist(prediction_truth, prediction_p):
    l2_dist = (prediction_truth[:, 0].repeat(prediction_p.size(0), 1, 1) - prediction_p).pow(2).sum(dim=2).sum(
        dim=1)
    _, indices = l2_dist.min(dim=0)  # ground_truth best mode (batch-wise)
    return indices


def mpc_fun(best_mode_prediction_kd, sample_rate, pixel_scale, n_pred, observation):
    # pdb.set_trace()
    obs = observation.detach().cpu().numpy()
    y_ref = best_mode_prediction_kd[:, 1].detach().cpu().numpy()
    x_ref = best_mode_prediction_kd[:, 0].detach().cpu().numpy()
    y_ref = np.concatenate((obs[:, -1, 0, 1], y_ref), axis=0)
    x_ref = np.concatenate((obs[:, -1, 0, 0], x_ref), axis=0)
    delta_t = sample_rate
    CONVERSION = pixel_scale[0].cpu().numpy()
    scale = pixel_scale[0].cpu().numpy()
    N = n_pred + 1
    opti = casadi.Opti();
    x = opti.variable(1, N);
    y = opti.variable(1, N);
    psi = opti.variable(1, N);
    v = opti.variable(1, N);
    beta = opti.variable(1, N);

    u1 = opti.variable(1, N - 1);  # acceleration
    u2 = opti.variable(1, N - 1);  # angle

    err_u1 = opti.variable(1, N - 1);
    err_u2 = opti.variable(1, N - 1);
    for k in range(N - 2):
        opti.subject_to(err_u1[k + 1] == u1[k + 1] - u1[k])
        opti.subject_to(err_u2[k + 1] == u2[k + 1] - u2[k])

    p = opti.parameter(4, 1);
    opti.minimize(
        5 * sumsqr(y - np.array(y_ref, ndmin=2)) + 5 * sumsqr(x - np.array(x_ref, ndmin=2)) + 0.1 * sumsqr(
            err_u1) + 5 * sumsqr(err_u2) + 0.1 * sumsqr(u1));

    opti.subject_to(u1 <= 4 * scale);
    opti.subject_to(u2 <= 45 * np.pi / 180);
    opti.subject_to(u1 >= -8 * scale);
    opti.subject_to(u2 >= -45 * np.pi / 180);

    x[0] = p[0]
    y[0] = p[1]
    psi[0] = p[2]
    v[0] = p[3]
    for k in range(N - 1):
        opti.subject_to(beta[k] == np.arctan(np.tan(u2[k]) * 1.77 / (1.77 + 1.17)))
        # best model
        # opti.subject_to(beta[k] == np.arctan(np.tan(u2[k])/2))
        opti.subject_to(x[k + 1] == x[k] + v[k] * delta_t * cos(psi[k] + beta[k]))
        opti.subject_to(y[k + 1] == y[k] + v[k] * delta_t * sin(psi[k] + beta[k]))
        # best model
        # opti.subject_to(psi[k + 1] == psi[k] + v[k] / (1.5 * CONVERSION) * delta_t * sin(beta[k]))
        opti.subject_to(psi[k + 1] == psi[k] + v[k] / (1.77 * CONVERSION) * delta_t * sin(beta[k]))
        opti.subject_to(v[k + 1] == v[k] + u1[k] * delta_t)

    opti.print_header = False
    opti.print_iteration = False
    opti.print_time = False
    opti.solver('ipopt');
    pixel_d = np.sqrt((obs[:, -1, 0, 1] - obs[:, -2, 0, 1]) ** 2 + (obs[:, -1, 0, 0] - obs[:, -2, 0, 0]) ** 2)
    velocity = pixel_d / delta_t
    psi_ref = np.arctan2((obs[:, -1, 0, 1] - obs[:, -2, 0, 1]), (obs[:, -1, 0, 0] - obs[:, -2, 0, 0]))

    # pdb.set_trace()
    opti.set_value(p, [obs[:, -1, 0, 0], obs[:, -1, 0, 1], psi_ref, velocity])
    sol = opti.solve();
    x = np.expand_dims(sol.value(x), axis=1)
    y = np.expand_dims(sol.value(y), axis=1)

    return torch.tensor(np.concatenate((x, y), axis=1))
