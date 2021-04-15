import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import augmentation


class scene_funcs(torch.nn.Module):
    """  This class has all we want to do with the scene
    """

    def __init__(self, device='cpu'):
        super(scene_funcs, self).__init__()
        self.device = device
        file = open('./segmentedImgs/scales.txt')  # generate dictionary for scale values
        lines = file.read().split()
        self.pixel_scale_dict = dict(zip(lines[::2], lines[1::2]))
        self.scene_info = dict()
        file = open('./segmentedImgs/offset.txt')  # generate dictionary for scale values
        lines = file.read().split()
        self.offset_dict = {}
        list1 = lines[::3]
        list2 = lines[1::3]
        list3 = lines[2::3]
        for i in list1:
            self.offset_dict[i] = []
        for i in range(len(list1)):
            self.offset_dict[list1[i]].append(float(list2[i]))
            self.offset_dict[list1[i]].append(float(list3[i]))

        baseAdd_image = './segmentedImgs/segmentedImgs_jpg/'
        self.image = {}
        for i in list1:
            array = np.round(cv2.imread(baseAdd_image + i + '.jpg',
                                        cv2.IMREAD_GRAYSCALE) / 255 - 1)  # reads the image, off-road is zero and road is -1
            self.image[i] = torch.tensor(1 + 2 * array, device=self.device)

        margin = 425
        self.index_mat_x = torch.transpose(torch.tensor([range(2 * margin)], dtype=torch.float32, device=self.device),
                                           1, 0).repeat([1, 2 * margin])
        self.index_mat_y = torch.tensor([range(2 * margin)], dtype=torch.float32, device=self.device).repeat(
            [2 * margin, 1])

    def scene_data(self, loc, resampling=True, find_cor=False, file_name=None):
        """This function calculates the scene information ,
        the area will be a resampled_dim*resampled_dim box """
        loc = torch.tensor(loc, device=self.device)

        scale = 1
        if ('DJI' in file_name[0]):
            scale = 5

        batch_size = loc.size()[0]
        dim_x, dim_y = self.return_image(file_name[
                                             0]).size()  # In training all the data in the batch are from the same scene. in test, we only have one batch so it is enough to look at the first batch
        scene_size = [dim_x, dim_y]
        img_binary = torch.zeros(batch_size, dim_x, dim_y,
                                 device=self.device)  # size = (batch_size, image_size[0], image_size[1])  #
        location = loc.type(torch.int32) // scale
        marg = self.return_margin(file_name=file_name)
        cor1 = torch.min(torch.max(location - marg, torch.tensor([0, 0], dtype=torch.int32, device=self.device)),
                         torch.tensor([scene_size], dtype=torch.int32, device=self.device)).type(torch.int32)
        cor4 = torch.min(torch.max(location + marg, torch.tensor([0, 0], dtype=torch.int32, device=self.device)),
                         torch.tensor([scene_size], dtype=torch.int32, device=self.device)).type(torch.int32)
        if (find_cor):
            return cor1, cor4
        for i in range(batch_size):
            img_binary[i,] = self.return_image(file_name[i])

        img = torch.zeros(batch_size, 2 * marg, 2 * marg, device=self.device)
        img.fill_(
            1)  # fill it with off road, so if in any xases we dont have a matrix equal to margin* margin dimention we will have off-road
        for i in range(batch_size):
            # img[i, 0:cor4[i,1]-cor1[i,1], 0:cor4[i,0]-cor1[i,0]] = torch.transpose(img_binary[i,cor1[i,0]:cor4[i,0],cor1[i,1]:cor4[i,1]],0,1) # as we have the transpose, the dimentions is in inverse mode
            img[i, 0:cor4[i, 0] - cor1[i, 0], 0:cor4[i, 1] - cor1[i, 1]] = img_binary[i, cor1[i, 0]:cor4[i, 0],
                                                                           cor1[i, 1]:cor4[
                                                                               i, 1]]  # as we have the transpose, the dimentions is in inverse mode
        return img

    def return_image(self, file_name=None):
        return self.image[file_name]

    def return_margin(self,
                      file_name=None):  # margin should be returned due to the scene and scene size so we provide a new function for that
        # dim_x, dim_y = self.return_image(file_name[0]).size()
        marg_meter = 40  # meters

        scale = 1
        if ('DJI' in file_name[0]):  # used for internal dataset
            scale = 5
        pixel_scale = float(
            self.pixel_scale_dict[file_name[0]]) * scale  # multiplied by 5 as the scene image is rescaled by 5
        return int(marg_meter * pixel_scale)


def resampler(img, resampled_dim, batch_size=1):
    (resampled_dim_x, resampled_dim_y) = resampled_dim
    stride_x = np.ceil((img.size()[1]) / (resampled_dim_x)).astype(int).item()  # convert to python int
    stride_y = np.ceil((img.size()[2]) / (resampled_dim_y)).astype(int).item()  # convert to python int
    pooler = nn.AvgPool2d((stride_x, stride_y), stride=(stride_x, stride_y))
    resampled = pooler(img.view(batch_size, 1, img.size()[1], img.size()[2]))
    return resampled.view(batch_size, resampled.size()[2], resampled.size()[3])


def scene_preprocess(xy_copy, file_name, n_obs, resampling_dim, scene_funcs):
    batch_size = xy_copy.size(0)
    theta, xy_copy = augmentation.rotate_obs_to_vertical(xy_copy, n_obs)
    rotated_half_scene, rotated_scene = augmentation.img_rotator(obs=xy_copy[:, n_obs - 1, 0, :],
                                                                 batch_size=batch_size, file_name=file_name,
                                                                 theta=theta,
                                                                 device=xy_copy.device, scene_funcs=scene_funcs)
    resampled_scene = resampler(rotated_half_scene, resampled_dim=resampling_dim, batch_size=batch_size)
    resampled_scene = torch.squeeze(F.interpolate(torch.unsqueeze(resampled_scene, 1), size=resampling_dim),
                                    1)  # fix the size to (76,38). mode of interpolator is nearest which means that it extends off-road and on-road
    return rotated_scene, resampled_scene, xy_copy, theta


def nearest_point_on_road(locc, image, scene_funcs):  # used for road loss
    device = locc.device
    loc = locc.clone().type(torch.int32)
    dim1, dim2 = image.size()
    cor1 = torch.tensor([0, 0], device=locc.device, dtype=torch.int32)
    cor4 = torch.tensor([dim1, dim2], device=locc.device, dtype=torch.int32)
    loc = torch.min(torch.max(loc, torch.tensor([0, 0], dtype=torch.int32, device=device)),
                    torch.tensor([dim1, dim2], dtype=torch.int32, device=device))
    search_area_ = 100
    loc = locc.type(torch.int32)
    search_area = torch.min(torch.min((loc - cor1), cor4 - loc),
                            torch.tensor([search_area_, search_area_], dtype=torch.int32, device=device))

    img_size = image.size()
    if (image[tuple(
            loc)] == -1):  # if it is inside the road - as we have the image croped from cor1, the coordinates should be transfered to the new system
        print('the point is inside the road! no need for loss!!')
    loc_mvd = loc  # -cor1

    # dist = (loc_x_mat - self.index_mat_x[:img_size[0],:img_size[1]])**2+(loc_y_mat-self.index_mat_y[:img_size[0],:img_size[1]])**2
    dist1 = (loc[0] - scene_funcs.index_mat_x[loc[0] - search_area[0] // 2:loc_mvd[0] + search_area[0] // 2,
                      loc_mvd[1] - search_area[1] // 2:loc_mvd[1] + search_area[1] // 2]) ** 2 + (
                    loc_mvd[1] - scene_funcs.index_mat_y[
                                 loc_mvd[0] - search_area[0] // 2:loc_mvd[0] + search_area[0] // 2,
                                 loc_mvd[1] - search_area[1] // 2:loc_mvd[1] + search_area[1] // 2]) ** 2
    mask = torch.tensor([(search_area_) ** 2], dtype=torch.float32, device=device) * ((image[
                                                                                       loc_mvd[0] - search_area[0] // 2:
                                                                                       loc_mvd[0] + search_area[0] // 2,
                                                                                       loc_mvd[1] - search_area[1] // 2:
                                                                                       loc_mvd[1] + search_area[
                                                                                           1] // 2] > 0).type(
        torch.float32))  # to mask points outside the road - points outside the raod will be a large value and points inside will be 0
    dist = dist1 + mask
    closest_point = torch.tensor([0, 0], device=device)
    if (dist.size(0) == 0 or dist.size(1) == 0):  # if it is on the boundry, just let it go.
        return locc
    closest_point = torch.tensor(
        [(torch.argmin(dist) / dist.size(1)).type(torch.int32), (torch.argmin(dist) % dist.size(1)).type(torch.int32)],
        device=device)  # convert argument to row and column numbers

    return closest_point + loc - search_area // 2  # loc_mvd-search_area//2+cor1 = loc-search_area//2 ,return point in actual map (not the croped one)
