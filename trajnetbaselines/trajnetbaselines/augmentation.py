import math
import random
import torch
import torch.nn.functional as F
import trajnettools


def rotate_path(path, theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    return [trajnettools.TrackRow(r.frame, r.pedestrian, ct * r.x + st * r.y, -st * r.x + ct * r.y)
            for r in path]


def random_rotation_of_paths(paths):
    theta = random.random() * 2.0 * math.pi
    return [rotate_path(path, theta) for path in paths]


def rotate_obs_to_vertical(scene, n_obs):
    """
    This function rotates the path of the vehicle of interest in a way that the path will be vertival with respect to the last observation
    It rotates all vehicles in the scene according to the way the main veh was rotated
    """
    device = scene.device
    last_p = scene[:, n_obs - 1, 0, :]  # final observation point
    bef_last_p = scene[:, n_obs - 2, 0, :]  # one point before final observation
    temp = (bef_last_p[:, 1] - last_p[:, 1]) / (bef_last_p[:, 0] - last_p[:, 0])
    theta = -torch.atan(temp)
    # Faster implementation
    theta[torch.isinf(temp)] = (-torch.tensor(math.pi) / 2).to(device)
    theta[torch.isinf(temp) * (bef_last_p[:, 1] > last_p[:, 1])] = torch.tensor(math.pi, device=device) / 2
    theta[torch.isnan(temp)] = torch.tensor(math.pi, device=device) / 2
    theta = theta / (2 * torch.tensor(math.pi)) * 360 + (
            2 * (last_p[:, 0] < bef_last_p[:, 0]).type(torch.float32) - 1) * 90 - 90  #

    ct = torch.cos(theta * 2 * math.pi / 360)
    st = torch.sin(theta * 2 * math.pi / 360)
    # rotate
    ps = scene  # points to rotate
    last_p = last_p.view(-1, 1, 1, 2)
    ct = ct.view(-1, 1, 1)
    st = st.view(-1, 1, 1)
    ps = ps - last_p
    temp1 = torch.round(ps[:, :, :, 0] * ct - ps[:, :, :, 1] * st + last_p[:, :, :, 0])
    temp2 = torch.round(ps[:, :, :, 0] * st + ps[:, :, :, 1] * ct + last_p[:, :, :, 1])
    ps[:, :, :, 0] = temp1
    ps[:, :, :, 1] = temp2
    return theta, ps


def rotate_all_path_by_theta(path, center, theta, centerline=0):
    """
    This function rotates each path by theta degree around center
    all trajs in path are rotated
    scene is [batch, n_pred, 2] for centerline=0
    path is [batch, n_pred, 2] for centerline=0
    """
    if centerline:
        ps = path.clone().unsqueeze(0).repeat(center.size(0), 1, 1, 1).permute(0, 1, 3, 2).type(torch.float64)
        ct = torch.cos(theta * 2 * math.pi / 360)
        st = torch.sin(theta * 2 * math.pi / 360)
        ct = ct.view(-1, 1, 1).type(torch.float64)
        st = st.view(-1, 1, 1).type(torch.float64)
        ps = ps - center.type(torch.float64)
        temp1 = torch.round(ps[:, :, :, 0] * ct - ps[:, :, :, 1] * st + center[:, :, :, 0].type(torch.float64))
        temp2 = torch.round(ps[:, :, :, 0] * st + ps[:, :, :, 1] * ct + center[:, :, :, 1].type(torch.float64))
        ps[:, :, :, 0] = temp1
        ps[:, :, :, 1] = temp2
        return ps.permute(0, 1, 3, 2)
    else:
        center_sqz = center.clone().unsqueeze(1)
        ps = path.clone()  # .unsqueeze(0).repeat(center.size(0),1,1,1)#.permute(0,1,3,2).type(torch.float64)
        ct = torch.cos(theta * 2 * math.pi / 360)
        st = torch.sin(theta * 2 * math.pi / 360)
        ct = ct.view(-1, 1)  # .type(torch.float64)
        st = st.view(-1, 1)  # .type(torch.float64)
        ps = ps - center_sqz  # .type(torch.float64)
        temp1 = torch.round(ps[:, :, 0] * ct - ps[:, :, 1] * st + center_sqz[:, :, 0])  # .type(torch.float64))
        temp2 = torch.round(ps[:, :, 0] * st + ps[:, :, 1] * ct + center_sqz[:, :, 1])  # .type(torch.float64))
        ps[:, :, 0] = temp1
        ps[:, :, 1] = temp2

        return ps  # .permute(0,1,3,2)


def rotate_path_by_theta(path, center, n_pred, theta):
    """
    This function rotates each path by theta degree around center 
    ***Caution*** only veh of interest is rotated
    scene is [n_pred, 2]
    """
    ct = torch.cos(theta * 2 * math.pi / 360)
    st = torch.sin(theta * 2 * math.pi / 360)
    ps = path  # points to rotate
    ct = ct.view(-1, 1)
    st = st.view(-1, 1)
    ps = ps - center
    temp1 = torch.round(ps[:, 0] * ct - ps[:, 1] * st + center[:, 0])
    temp2 = torch.round(ps[:, 0] * st + ps[:, 1] * ct + center[:, 1])
    ps[:, 0] = temp1
    ps[:, 1] = temp2

    return ps


def img_rotator(obs, batch_size, file_name, theta, scene_funcs, device='cpu'):
    """
    https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/2?u=simenthys
    This func rotates the image by theta degrees
    """

    scale = 1
    if ('DJI' in file_name[0]):  # used for internal dataset
        scale = 5

    scene = scene_funcs.scene_data(obs.view(batch_size, 2), resampling=False, file_name=file_name)
    # Rotation adds some margins in the image which should be considered as off-road. As the values in the margins
    # are 0 and should be considered as offroad, we transform the values to 0,1 and  then will come back to our
    # previous notation.
    scene = -0.5 * scene + 0.5  # offroad (1) will become 0
    marg = scene_funcs.return_margin(file_name=file_name)
    cor1, cor4 = scene_funcs.scene_data(obs, file_name=file_name, find_cor=True)
    cor1 = cor1.to(device)
    cor4 = cor4.to(device)
    scene_center = torch.zeros(batch_size, marg * 2, marg * 2, device=device)
    start_x = (torch.tensor(marg, dtype=torch.float32, device=device) - (
            obs[:, 0] // scale - cor1[:, 0].type(torch.float32))).type(
        torch.int32)  # devided by 5 as the images are resided to 1/5 so the pixel values should be devided by 5
    start_y = (torch.tensor(marg, dtype=torch.float32, device=device) - (
            obs[:, 1] // scale - cor1[:, 1].type(torch.float32))).type(torch.int32)
    for i in range(batch_size):
        scene_center[i, start_x[i]:start_x[i] + cor4[i, 0] - cor1[i, 0],
        start_y[i]:start_y[i] + cor4[i, 1] - cor1[i, 1]] = scene[i, 0:cor4[i, 0] - cor1[i, 0],
                                                           0:cor4[i, 1] - cor1[i, 1]]

    rotated_scene = torch.zeros(batch_size, 2 * marg, 2 * marg, device=device)
    for i in range(batch_size):
        x = scene_center[i].clone()
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        angle = (theta[i] / (180.)) * math.pi
        teta = torch.zeros(1, 2, 3, device=device)
        teta[:, :, :2] = torch.tensor([[math.cos(angle), -1.0 * math.sin(angle)],
                                       [math.sin(angle), math.cos(angle)]])
        grid = F.affine_grid(teta, x.size())
        rotated_scene[i] = F.grid_sample(x, grid)
    rotated_scene = rotated_scene.to(device)
    rotated_scene = rotated_scene * (-2) + 1
    margin = int(0.3 * marg)
    croped_img = rotated_scene[:, margin:-margin, margin:-margin]
    half_img = croped_img[:, :int(croped_img.size()[1] / 2), :]
    return half_img, croped_img
