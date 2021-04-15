import torch.nn as nn
from .utils import *

warnings.filterwarnings("ignore")
# import gc

use_dmpc = 0


class RRB(torch.nn.Module):
    """
    We should use centerline. For the speed, we can use the last value or the value of the closest veh infront.
    """

    def __init__(self, n_obs, n_pred, device):
        super(RRB, self).__init__()
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.num_of_neighbors = 3
        self.device = device if device is not None else torch.device('cpu')
        self.counter = 0
        self.scene_mode = 'RRB'  # to be used in evaluation part
        self.resampling_dim = None  # this var stores resampling value for the time model is loaded to restore the value

        self.M = 1
        self.encoder_traj = nn.Sequential(
            nn.Linear(2 * (n_obs - 1), 32),
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.encoder_vehs = nn.Sequential(
            nn.Linear(2 * 3 * self.num_of_neighbors, 32),
            # 2 for x y, 3 as we used relative coordinates for the last three observations
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 64 + 64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU()
        )

        self.map_to_center = nn.Sequential(
            torch.nn.Linear(64 + 128, 128), nn.ReLU(),
        )
        self.residual_regress_mean = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            torch.nn.Linear(64, n_pred * 2), nn.Tanh()
        )
        self.residual_regress_var = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_pred * (3 + 4))  # varX,rhoX (3) + var(XY)(4)
        )
        self.input_embeddings = nn.Sequential(
            nn.Linear(self.n_pred * 2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )
        self.cnn = CNN()
        self.cnn = self.cnn.to(self.device)
        self.mean = 0
        self.num = 0
        self.kd_covar_2 = torch.tensor([[142.4469, 14.6],
                                        [546.3336, 10.4],
                                        [1214.3231, 35.05],
                                        [2144.3779, 89.3],
                                        [3329.2144, 177.8],
                                        [4768.9585, 306.8623],
                                        [6452.0522, 475.8614],
                                        [8370.7070, 686.2020],
                                        [10499.2588, 940.2],
                                        [12799.6650, 1243]], device=self.device)
        self.kd_crossvar = torch.tensor([0.12, -0.49, 2.38, 10.81, 23.1, 36.05, 45.6862,
                                         47.88, 33.77, -2.14], device=self.device)

    def forward(self, obs, prediction_truth=None, n_pred=None, center_line_dict=None, scene=None, sample_rate=None,
                pixel_scale=None, scene_funcs=None, file_name=None, rotated_scene=None, epoch=None, margin=None,
                iterations=None):
        """
        inputs: obs (tensor, shape=(n_batch, n_obs,n_nearby_objects(it depends on the scene),2) )
        n_pred is used for compatibility with LSTM model
        """
        batch_size = np.shape(obs)[0]
        # ----------------encoding --------------------
        # trajectory
        v = obs[:, 1:, 0, :] - obs[:, :-1, 0,
                               :]  # generate velocity from locations and exclude neighbors, shape=(n_obs,1,2)
        pixel_scale = pixel_scale.unsqueeze(1).unsqueeze(2)
        sample_rate = sample_rate.unsqueeze(1).unsqueeze(2)
        v[:, :, 0] = v[:, :, 0] + 18  # To bias data towords zero!
        v = v * sample_rate / pixel_scale  # normalize velocity to meters per second
        v = v.view(batch_size, 1, -1)  # input to the nn.linear should be : (N, *, in_features)
        encoded_traj = self.encoder_traj(v)
        # scene
        scene_features = self.cnn(scene)
        scene_features = scene_features.view(batch_size, 1, 128)
        # Other vehicles
        """filter out the vehicles except the ones infront and left (As they are the ones that influence in roundabouts). Note we are in a normalized view of the scene.
        In the visualization (cv2), if we want to show (10,1000) we have to move 10 pixels down and 1000 forward. As the trajectories are normalized to the top
        veiw according to the visualizations, we keep the same method. So a vehicle in front of (a,b) means (a-20,b)
        """
        # select the forward and right vehicles
        obs_cpy = obs.clone()
        temp = 1 - (obs_cpy[:, -1:, 0:1, 0:1] >= obs_cpy[:, -1:, :, 0:1]).type(
            torch.int)  # put zeros for vehicles in front and ones for others
        temp_ = temp.repeat(1, obs_cpy.size(1), 1, obs_cpy.size(3)).type(torch.bool)
        obs_cpy[temp_] = -1000
        obs_cpy[obs_cpy != obs_cpy] = -1000  # replace nans with zeros
        # select the 3 closest ones
        dist = (obs_cpy[:, -1] - obs_cpy[:, -1, 0:1, :]).pow(2).sum(
            dim=2)  # returns the dist between each point and veh of interest in the last observation point
        closest_dist, closest_indexes = torch.topk(dist, k=min(self.num_of_neighbors + 1, obs.size(2)), dim=1,
                                                   largest=False)  # find 3 smallest values (we find 4 as the first one is egovehicle)
        closest_dist = closest_dist[:, 1:]  # removing the egovehicle
        closest_indexes = closest_indexes[:, 1:]  # removing the egovehicle
        closest_nghbrs = torch.gather(obs[:, -3:], 2, closest_indexes.unsqueeze(1).unsqueeze(3).repeat(1, 3, 1,
                                                                                                       2))  # consider the last three observation for each vehicle
        vehs_rel_pos = obs[:, -1:, 0:1, :] - closest_nghbrs
        if (obs.size(2) == 1):  # there were no neighbors
            vehs_rel_pos = obs[:, -3:, 0:1]  # let's add the egovehicle and we will change it to 2000 next line
            closest_dist = obs[:, 0, 0, 0:1]
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0:1, :, -1:].repeat(1, 1, self.num_of_neighbors - obs.size(2), 1)], dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors - obs.size(2))],
                dim=1)
        elif (obs.size(
                2) < self.num_of_neighbors + 1):  # To handle the cases in test mode that does not have enough neighbors
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0, :, -1:].repeat(1, 1, self.num_of_neighbors + 1 - obs.size(2), 1)],
                dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors + 1 - obs.size(2))],
                dim=1)
            closest_dist = closest_dist.repeat(1, self.num_of_neighbors + 1 - obs.size(2))
        for j in range(batch_size):
            for i in range(vehs_rel_pos.size(2)):  # in every vehicle
                if closest_dist[j, i] > 600000 or torch.isnan(vehs_rel_pos[j, :, i].sum()):
                    vehs_rel_pos[j, :, i] = 2000
        if torch.sum(torch.isnan(vehs_rel_pos)):
            "nan in the data!"
            pdb.set_trace()
        vehs_rel_pos = vehs_rel_pos * (sample_rate.unsqueeze(3)) / (pixel_scale.unsqueeze(3))
        vehs_rel_pos[:, :, :, 0] = vehs_rel_pos[:, :, :, 0] - 10
        vehs_rel_pos[:, :, :, 1] = vehs_rel_pos[:, :, :, 1] - 7

        vehs_rel_pos = vehs_rel_pos.view(batch_size, 1, -1)  # flaten the tensor
        encoded_vehs = self.encoder_vehs(vehs_rel_pos)
        # ----------------decoding --------------------
        # (1) Generate trajectory positions using constant velocity.
        pos_temp = obs[:, -1, 0:1, :].unsqueeze(1)
        last_vel = obs[:, -1, 0:1, :].unsqueeze(1) - obs[:, -2, 0:1, :].unsqueeze(1)
        prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        for idx in range(self.n_pred):
            pos_temp = last_vel + pos_temp
            prediction_pos[:, :, idx:idx + 1] = pos_temp

        last_speed = last_vel.pow(2).sum(dim=3).pow(0.5).unsqueeze(1)
        prediction_speed = last_speed * torch.ones([batch_size, self.M, self.n_pred, 1], device=obs.device)

        # (2) Generate trajectory positions Using kalman filter
        # prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        # for i in range(batch_size):
        #    prediction_pos[i] = trajnetbaselines.kalman.predict_modified(obs[i,:,0])
        #        
        # prediction_v = prediction_pos[:, :, 1:, :] - prediction_pos[:, :, :-1,:]
        # first_v = (prediction_pos[:, :, 0, :] -  obs[:, -1, 0:1, :]).unsqueeze(2)
        # prediction_v = torch.cat((first_v,prediction_v),dim=2)
        # prediction_speed = prediction_v.pow(2).sum(dim=3).pow(0.5).unsqueeze(3)

        center_road = extract_center_line_of_interset(obs.detach(), prediction_pos.detach(), center_line_dict,
                                                      obs.device, iterations, epoch, self.n_pred)
        old_position = obs[:, -1, 0, :].unsqueeze(1).unsqueeze(1).repeat(1, 1, center_road.shape[2], 1)
        d = (center_road - old_position).pow(2).sum(dim=3).pow(0.5)
        indices = torch.tensor([], dtype=torch.float).to(device=obs.device)
        multimodel_center_road_traj = torch.tensor([], dtype=torch.float).to(device=obs.device)
        for k in range(self.M):  # number of modes of predictions
            center_road_traj = torch.tensor([], dtype=torch.float, device=obs.device)
            quant_err = 0 * prediction_speed[:, 0, 0]
            for t in range(self.n_pred):
                distances = torch.abs(d[:, k, :] - (prediction_speed[:, k, t] - quant_err))
                min_value, indices = torch.min(distances, dim=1)
                for j in range(batch_size):
                    center_road[j, k, :indices[j], :] = 50000
                temporal = torch.gather(center_road[:, k], 1, indices.view(-1, 1, 1).repeat(1, 1, 2)).squeeze(1)
                old_position = temporal.unsqueeze(1).repeat(1, center_road.shape[2], 1)
                d[:, k] = (center_road[:, k] - old_position).pow(2).sum(dim=2).pow(0.5)
                center_road_traj = torch.cat((center_road_traj, temporal), dim=1)
                center_road_traj_rshpd = center_road_traj.view(batch_size, -1, 2)
                if t != 0:  # lets skip the first predicted point (since finding speed is difficult)
                    mapped_speed = (center_road_traj_rshpd[:, -1:] - center_road_traj_rshpd[:, -2:-1]).pow(2).sum(
                        2).pow(0.5)
                    quant_err = mapped_speed - prediction_speed[:, k, t]
            multimodel_center_road_traj = torch.cat((multimodel_center_road_traj, center_road_traj.unsqueeze(1)), dim=1)
        multimodel_center_road_traj = multimodel_center_road_traj.view(batch_size, self.M, self.n_pred, 2).detach()

        # fix jumps for the predictions from obs to average behavior
        offset = obs[:, -1:, 0, 1:2] - multimodel_center_road_traj[:, :, 0,
                                       1:2]  # offset of the first prediction on dimension 1
        multimodel_center_road_traj[:, :, :, 1] = multimodel_center_road_traj[:, :, :, 1] + offset * 3 / 4

        center_line_v = (multimodel_center_road_traj[:, :, :, :] - obs[:, -1:, 0:1, :])
        center_line_v[:, :, :, 0] = center_line_v[:, :, :, 0] + 70  # + 110
        center_line_v[:, :, :, 1] = center_line_v[:, :, :, 1] + 0.5  # - 1

        kd_embedded = self.input_embeddings(center_line_v[:, 0, :, :].view(batch_size, 1, -1))
        residual_feature1 = self.decoder(torch.cat((encoded_traj, encoded_vehs, kd_embedded), 2))
        residual_feature = self.decoder2(residual_feature1)

        residual_mean = self.residual_regress_mean(residual_feature).view(batch_size, 1, self.n_pred, 2)
        residual_var = self.residual_regress_var(residual_feature).view(batch_size, 1, self.n_pred, 7)
        residual_var[:, :, :, 0:2] = torch.exp(residual_var[:, :, :, 0:2])
        residual_var[:, :, :, 2:] = torch.tanh(residual_var[:, :, :, 2:])

        reference_traj_noMixed = torch.cat(
            (multimodel_center_road_traj + residual_mean * 33, residual_var[:, :, :, :3]), dim=3)

        # W1X + W2Y | X=NN, Y = KD | X=[x1,x2]
        res_var = residual_var[:, :, :, 0:2]  # sigma x1 and x2
        res_covarx1x2 = residual_var[:, :, :, 2] * res_var[:, :, :, 0] * res_var[:, :, :, 1]  # sigmax1x2
        res_crossvarXY_cpy = residual_var[:, :, :, 3:]  # sigmaXY
        res_crossvarXY = res_crossvarXY_cpy.clone()
        res_crossvarXY[:, :, :, 0] = res_crossvarXY_cpy[:, :, :, 0] * res_var[:, :, :, 0] * (
            self.kd_covar_2[:, 0].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax1y1
        res_crossvarXY[:, :, :, 1] = res_crossvarXY_cpy[:, :, :, 1] * res_var[:, :, :, 0] * (
            self.kd_covar_2[:, 1].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax1y2
        res_crossvarXY[:, :, :, 2] = res_crossvarXY_cpy[:, :, :, 2] * res_var[:, :, :, 1] * (
            self.kd_covar_2[:, 0].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax2y1
        res_crossvarXY[:, :, :, 3] = res_crossvarXY_cpy[:, :, :, 3] * res_var[:, :, :, 1] * (
            self.kd_covar_2[:, 1].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax2y2

        sigma_x1y1_x2y2 = torch.cat([res_crossvarXY[:, :, :, 0:1], res_crossvarXY[:, :, :, 3:]], dim=3)
        dinaminator = self.kd_covar_2 + res_var.pow(2) - 2 * sigma_x1y1_x2y2
        W1 = 0.5 + 0.5 * nn.functional.tanh(
            (self.kd_covar_2 - sigma_x1y1_x2y2) / dinaminator)  # bound betw [0,1] using Relu
        W2 = 1 - W1

        mixed_mean = (multimodel_center_road_traj + residual_mean * 33) * W1 + W2 * multimodel_center_road_traj
        mixed_var = (W1.pow(2) * res_var.pow(2) + W2.pow(2) * self.kd_covar_2 + 2 * W1 * W2 * sigma_x1y1_x2y2).pow(0.5)
        mixed_covar = (W1[:, :, :, 0] ** 2) * res_covarx1x2 + (W2[:, :, :, 0] ** 2) * (
            self.kd_crossvar.unsqueeze(0).unsqueeze(0)) + W1[:, :, :, 0] * W2[:, :, :, 1] * \
                      res_crossvarXY[:, :, :, 1] + W1[:, :, :, 1] * W2[:, :, :, 0] * res_crossvarXY[:, :, :, 2]
        mixed_rho = (mixed_covar / (mixed_var[:, :, :, 0] * mixed_var[:, :, :, 1])).unsqueeze(
            3)  # because in the loss we have rho not the covariance
        reference_traj = torch.cat((mixed_mean, mixed_var, mixed_rho), dim=3)
        return reference_traj_noMixed, multimodel_center_road_traj, torch.zeros([batch_size, self.M],
                                                                                device=obs.device), reference_traj, prediction_speed


class RRB_M(torch.nn.Module):

    def __init__(self, n_obs, n_pred, device):
        super(RRB_M, self).__init__()
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.num_of_neighbors = 3
        self.device = device if device is not None else torch.device('cpu')
        self.counter = 0
        self.scene_mode = 'RRB_M'  # to be used in evaluation part
        self.resampling_dim = None  # this var stores resampling value for the time model is loaded to restore the value

        self.M = 2
        self.encoder_traj = nn.Sequential(
            nn.Linear(2 * (n_obs - 1), 32),
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.encoder_vehs = nn.Sequential(
            nn.Linear(2 * 3 * self.num_of_neighbors, 32),
            # 2 for x y, 3 as we used relative coordinates for the last three observations
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 64 + 128, 128), nn.ReLU()
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
        )
        self.regressor_mean = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, (n_pred * 2) * self.M)  # for predicting speed
        )
        self.regressor_var = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_pred * 3 * self.M)
        )
        self.encoder_tra2 = nn.Sequential(
            nn.Linear(2 * (n_obs - 1), 32),
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.encoder_veh2 = nn.Sequential(
            nn.Linear(2 * 3 * self.num_of_neighbors, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.decoder_res = nn.Sequential(
            nn.Linear(64 + 64 + 64 * 2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.decoder2_res = nn.Sequential(
            # nn.Linear(64+ 128, 128), nn.ReLU()
            nn.Linear(128, 128), nn.ReLU()
        )
        self.input_embeddings = nn.Sequential(
            nn.Linear(self.n_pred * 2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )
        self.residual_regress_mean = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            torch.nn.Linear(64, n_pred * 2 * self.M), nn.Tanh()
        )
        self.residual_regress_var = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_pred * (3 + 4) * self.M)  # varX,rhoX (3) + var(XY)(4)
        )
        self.prob_estimator = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            torch.nn.Linear(32, self.M)
        )

        self.kd_covar_2 = torch.tensor([[142.4469, 14.6],
                                        [546.3336, 10.4],
                                        [1214.3231, 35.05],
                                        [2144.3779, 89.3],
                                        [3329.2144, 177.8],
                                        [4768.9585, 306.8623],
                                        [6452.0522, 475.8614],
                                        [8370.7070, 686.2020],
                                        [10499.2588, 940.2],
                                        [12799.6650, 1243]], device=self.device)
        self.kd_crossvar = torch.tensor([0.12, -0.49, 2.38, 10.81, 23.1, 36.05, 45.6862,
                                         47.88, 33.77, -2.14], device=self.device)

        self.cnn = CNN()
        self.cn2 = CNN()
        self.cn2 = self.cn2.to(self.device)
        self.cnn = self.cnn.to(self.device)

    def forward(self, obs, prediction_truth=None, n_pred=None, center_line_dict=None, scene=None, sample_rate=None,
                pixel_scale=None, scene_funcs=None, file_name=None, rotated_scene=None, epoch=None, margin=None,
                iterations=None):
        """
        inputs: obs (tensor, shape=(n_batch, n_obs,n_nearby_objects(it depends on the scene),2) )
        n_pred is used for compatibility with LSTM model
        """
        batch_size = np.shape(obs)[0]
        # ----------------encoding --------------------
        # trajectory
        v = obs[:, 1:, 0, :] - obs[:, :-1, 0,
                               :]  # generate velocity from locations and exclude neighbors, shape=(n_obs,1,2)
        pixel_scale = pixel_scale.unsqueeze(1).unsqueeze(2)
        sample_rate = sample_rate.unsqueeze(1).unsqueeze(2)
        v = v * sample_rate / pixel_scale  # normalize velocity to meters per second
        v = v.view(batch_size, 1, -1)  # input to the nn.linear should be : (N, *, in_features)
        encoded_traj = self.encoder_traj(v)  # output to the nn.linear is : (N, *, out_features)
        # scene
        scene_features = self.cnn(scene)
        feature_scene_features = self.cn2(scene)
        scene_features = scene_features.view(batch_size, 1, 128)
        feature_scene_features = feature_scene_features.view(batch_size, 1, 128)
        # Other vehicles
        """filter out the vehicles except the ones in front and left (As they are the ones that influence in roundabouts). Note we are in a normalized view of the scene.
        In the visualization (cv2), if we want to show (10,1000) we have to move 10 pixels down and 1000 forward. As the trajectories are normalized to the top
        veiw according to the visualizations, we keep the same method. So a vehicle infront of (a,b) means (a-20,b)
        """
        # select the forward and right vehicles
        obs_cpy = obs.clone()
        temp = 1 - (obs_cpy[:, -1:, 0:1, 0:1] >= obs_cpy[:, -1:, :,
                                                 0:1]).type(
            torch.int)  # put zeros for vehicles in front and ones for others
        temp_ = temp.repeat(1, obs_cpy.size(1), 1, obs_cpy.size(3)).type(torch.bool)
        obs_cpy[temp_] = -1000
        obs_cpy[obs_cpy != obs_cpy] = -1000  # replace nans with zeros
        # select the 3 closest ones
        dist = (obs_cpy[:, -1] - obs_cpy[:, -1, 0:1, :]).pow(2).sum(
            dim=2)  # returns the dist between each point and veh of interest in the last observation point
        closest_dist, closest_indexes = torch.topk(dist, k=min(self.num_of_neighbors + 1, obs.size(2)), dim=1,
                                                   largest=False)  # .indices[:,1:]  # find 3 smallest values (we find 4 as the first one is egovehicle)
        closest_dist = closest_dist[:, 1:]  # removing the egovehicle
        closest_indexes = closest_indexes[:, 1:]  # removing the egovehicle
        closest_nghbrs = torch.gather(obs[:, -3:], 2, closest_indexes.unsqueeze(1).unsqueeze(3).repeat(1, 3, 1,
                                                                                                       2))  # consider the last three observation for each vehicle
        vehs_rel_pos = obs[:, -1:, 0:1, :] - closest_nghbrs
        if (obs.size(2) == 1):  # there were no neighbors
            vehs_rel_pos = obs[:, -3:, 0:1]  # let's add the egovehicle and we will change it to 2000 next line
            closest_dist = obs[:, 0, 0, 0:1]
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0:1, :, -1:].repeat(1, 1, self.num_of_neighbors - obs.size(2), 1)], dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors - obs.size(2))],
                dim=1)
        elif (obs.size(
                2) < self.num_of_neighbors + 1):  # To handle the cases in test mode that does not have enough neighbors
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0, :, -1:].repeat(1, 1, self.num_of_neighbors + 1 - obs.size(2), 1)],
                dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors + 1 - obs.size(2))],
                dim=1)
            closest_dist = closest_dist.repeat(1, self.num_of_neighbors + 1 - obs.size(2))
        for j in range(batch_size):
            for i in range(vehs_rel_pos.size(2)):  # in every vehicle
                if (closest_dist[j, i] > 600000 or torch.isnan(vehs_rel_pos[j, :, i].sum())):
                    vehs_rel_pos[j, :, i] = 2000
        if (torch.sum(torch.isnan(vehs_rel_pos))):
            pdb.set_trace()
        # vehs_rel_pos = vehs_rel_pos * (pixel_scale.unsqueeze(3)) * (sample_rate.unsqueeze(3))
        vehs_rel_pos = vehs_rel_pos * (sample_rate.unsqueeze(3)) / (pixel_scale.unsqueeze(3))
        vehs_rel_pos = vehs_rel_pos.view(batch_size, 1, -1)  # flaten the tensor
        encoded_vehs = self.encoder_vehs(vehs_rel_pos)
        feature_encoded_vehs = self.encoder_veh2(vehs_rel_pos)
        # ----------------decoding --------------------
        features_2 = self.decoder(torch.cat((encoded_traj, encoded_vehs, scene_features), 2))
        prediction_v_mean = self.regressor_mean(features_2).view(batch_size, self.M, self.n_pred, 2)
        prediction_v_var = self.regressor_var(features_2).view(batch_size, self.M, self.n_pred, 3)
        prediction_v_var[:, :, :, 0:2] = torch.exp(prediction_v_var[:, :, :, 0:2])
        prediction_v_var[:, :, :, 2:] = torch.tanh(prediction_v_var[:, :, :, 2:])
        prediction_v = torch.cat((prediction_v_mean, prediction_v_var), dim=3)
        pixel_scale = pixel_scale.unsqueeze(3)
        sample_rate = sample_rate.unsqueeze(3)
        prediction_v = prediction_v * pixel_scale / (sample_rate)
        pos_temp = obs[:, -1, 0:1, :].unsqueeze(1)
        # Generate trajectory positions given the predicted velocity.
        prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        for idx in range(self.n_pred):
            pos_temp = prediction_v[:, :, idx:idx + 1, 0:2] + pos_temp
            prediction_pos[:, :, idx:idx + 1] = pos_temp
        prediction_pos = torch.cat((prediction_pos, prediction_v_var), dim=3)

        center_road = extract_center_line_of_interset(obs.detach(), prediction_pos[:, :, :, :2].detach(),
                                                      center_line_dict, obs.device, iterations, epoch, self.n_pred)
        # ----------------decoding of features --------------------
        # (1) Generate trajectory positions given the predicted velocity.
        pos_temp = obs[:, -1, 0:1, :].unsqueeze(1)
        last_vel = obs[:, -1, 0:1, :].unsqueeze(1) - obs[:, -2, 0:1, :].unsqueeze(1)
        prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        for idx in range(self.n_pred):
            pos_temp = last_vel + pos_temp
            prediction_pos[:, :, idx:idx + 1] = pos_temp

        prediction_v = last_vel * torch.ones([batch_size, self.M, self.n_pred, 2], device=obs.device)
        last_speed = last_vel.pow(2).sum(dim=3).pow(0.5).unsqueeze(1)
        prediction_speed = last_speed * torch.ones([batch_size, self.M, self.n_pred, 1], device=obs.device)

        # (2) Generate trajectory positions Using kalman filter
        # prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        # for i in range(batch_size):
        #    prediction_pos[i] = trajnetbaselines.kalman.predict_modified(obs[i,:,0])
        # prediction_v = prediction_pos[:, :, 1:, :] - prediction_pos[:, :, :-1,:]
        # first_v = (prediction_pos[:, :, 0, :] -  obs[:, -1, 0:1, :]).unsqueeze(2)
        # prediction_v = torch.cat((first_v,prediction_v),dim=2)
        # prediction_speed = prediction_v.pow(2).sum(dim=3).pow(0.5).unsqueeze(3)

        old_position = obs[:, -1, 0, :].unsqueeze(1).unsqueeze(1).repeat(1, 1, center_road.shape[2], 1)
        d = (center_road - old_position).pow(2).sum(dim=3).pow(0.5)
        multimodel_center_road_traj = torch.tensor([], dtype=torch.float).to(device=obs.device)

        for k in range(self.M):  # number of modes of predictions
            center_road_traj = torch.tensor([], dtype=torch.float, device=obs.device)
            quant_err = 0 * prediction_speed[:, 0, 0]
            for t in range(self.n_pred):
                distances = torch.abs(d[:, k, :] - (prediction_speed[:, k, t] - quant_err))
                min_value, indice = torch.min(distances, dim=1)
                for j in range(batch_size):
                    center_road[j, k, :indice[j], :] = 50000
                temporal = torch.gather(center_road[:, k], 1, indice.view(-1, 1, 1).repeat(1, 1, 2)).squeeze(1)
                old_position = temporal.unsqueeze(1).repeat(1, center_road.shape[2], 1)
                d[:, k] = (center_road[:, k] - old_position).pow(2).sum(dim=2).pow(0.5)
                center_road_traj = torch.cat((center_road_traj, temporal), dim=1)
                center_road_traj_rshpd = center_road_traj.view(batch_size, -1, 2)
                if (t != 0):  # lets skip the first predicted point (since finding speed is difficult)
                    mapped_speed = (center_road_traj_rshpd[:, -1:] - center_road_traj_rshpd[:, -2:-1]).pow(2).sum(
                        2).pow(0.5)
                    quant_err = mapped_speed - prediction_speed[:, k, t]
            multimodel_center_road_traj = torch.cat((multimodel_center_road_traj, center_road_traj.unsqueeze(1)), dim=1)
        multimodel_center_road_traj = multimodel_center_road_traj.view(batch_size, self.M, self.n_pred, 2).detach()

        # fix jumps for the predictions from obs to average behavior
        offset = obs[:, -1:, 0, 1:2] - multimodel_center_road_traj[:, :, 0,
                                       1:2]  # offset of the first prediction on dimension 1
        multimodel_center_road_traj[:, :, :, 1] = multimodel_center_road_traj[:, :, :, 1] + offset * 3 / 4

        center_line_v = (multimodel_center_road_traj[:, :, :, :] - obs[:, -1:, 0:1, :])
        center_line_v[:, :, :, 0] = center_line_v[:, :, :, 0] + 70  # + 110
        center_line_v[:, :, :, 1] = center_line_v[:, :, :, 1] + 0.5  # - 1

        kd_embedded = torch.cat((self.input_embeddings(center_line_v[:, 0, :, :].view(batch_size, 1, -1)),
                                 self.input_embeddings(center_line_v[:, 1, :, :].view(batch_size, 1, -1))), 2)
        residual_feature1 = self.decoder_res(torch.cat((encoded_traj, encoded_vehs, kd_embedded), 2))
        residual_feature = self.decoder2_res(residual_feature1)

        residual_mean = self.residual_regress_mean(residual_feature).view(batch_size, self.M, self.n_pred, 2)
        residual_var = self.residual_regress_var(residual_feature).view(batch_size, self.M, self.n_pred, 7)
        residual_var[:, :, :, 0:2] = torch.exp(residual_var[:, :, :, 0:2])
        residual_var[:, :, :, 2:] = torch.tanh(residual_var[:, :, :, 2:])
        prob = self.prob_estimator(residual_feature).squeeze(1)

        reference_traj_noMixed = torch.cat(
            (multimodel_center_road_traj + residual_mean * 25, residual_var[:, :, :, :3]), dim=3)

        # W1X + W2Y | X=NN, Y = KD | X=[x1,x2]
        res_var = residual_var[:, :, :, 0:2]  # sigma x1 and x2
        res_covarx1x2 = residual_var[:, :, :, 2] * res_var[:, :, :, 0] * res_var[:, :, :, 1]  # sigmax1x2
        res_crossvarXY_cpy = residual_var[:, :, :, 3:]  # sigmaXY
        res_crossvarXY = res_crossvarXY_cpy.clone()
        res_crossvarXY[:, :, :, 0] = res_crossvarXY_cpy[:, :, :, 0] * res_var[:, :, :, 0] * (
            self.kd_covar_2[:, 0].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax1y1
        res_crossvarXY[:, :, :, 1] = res_crossvarXY_cpy[:, :, :, 1] * res_var[:, :, :, 0] * (
            self.kd_covar_2[:, 1].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax1y2
        res_crossvarXY[:, :, :, 2] = res_crossvarXY_cpy[:, :, :, 2] * res_var[:, :, :, 1] * (
            self.kd_covar_2[:, 0].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax2y1
        res_crossvarXY[:, :, :, 3] = res_crossvarXY_cpy[:, :, :, 3] * res_var[:, :, :, 1] * (
            self.kd_covar_2[:, 1].pow(0.5).unsqueeze(0).unsqueeze(0))  # sigmax2y2

        sigma_x1y1_x2y2 = torch.cat([res_crossvarXY[:, :, :, 0:1], res_crossvarXY[:, :, :, 3:]], dim=3)
        dinaminator = self.kd_covar_2 + res_var.pow(2) - 2 * sigma_x1y1_x2y2
        W1 = 0.5 + 0.5 * nn.functional.tanh(
            (self.kd_covar_2 - sigma_x1y1_x2y2) / dinaminator)  # bound betw [0,1] using Relu
        W2 = 1 - W1  # W2=nn.functional.relu((res_var.pow(2)-sigma_x1y1_x2y2)/dinaminator)

        mixed_mean = (multimodel_center_road_traj + residual_mean * 20) * W1 + W2 * multimodel_center_road_traj
        mixed_var = (W1.pow(2) * res_var.pow(2) + W2.pow(2) * self.kd_covar_2 + 2 * W1 * W2 * sigma_x1y1_x2y2).pow(0.5)
        mixed_covar = (W1[:, :, :, 0] ** 2) * res_covarx1x2 + (W2[:, :, :, 0] ** 2) * (
            self.kd_crossvar.unsqueeze(0).unsqueeze(0)) + W1[:, :, :, 0] * W2[:, :, :, 1] * \
                      res_crossvarXY[:, :, :, 1] + W1[:, :, :, 1] * W2[:, :, :, 0] * res_crossvarXY[:, :, :, 2]
        mixed_rho = (mixed_covar / (mixed_var[:, :, :, 0] * mixed_var[:, :, :, 1])).unsqueeze(
            3)  # because in the loss we have rho not the covariance
        reference_traj = torch.cat((mixed_mean, mixed_var, mixed_rho), dim=3)

        return reference_traj_noMixed, multimodel_center_road_traj, prob, reference_traj, prediction_speed


class EDN(torch.nn.Module):
    def __init__(self, n_obs, n_pred, device):
        super(EDN, self).__init__()
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.num_of_neighbors = 3
        self.counter = 0
        self.M = 1
        self.encoder_traj = nn.Sequential(
            nn.Linear(2 * (n_obs - 1), 32),
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.encoder_vehs = nn.Sequential(
            nn.Linear(2 * 3 * self.num_of_neighbors, 32),
            # 2 for x y, 3 as we used relative coordinates for the last three observations
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 64 + 128, 128), nn.ReLU()
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
        )
        self.regressor_mean = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, (n_pred * 2) * self.M)  # for predicting speed
        )
        self.regressor_var = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_pred * 3 * self.M)  # , nn.Sigmoid()
        )

        self.device = device if device is not None else torch.device('cpu')
        self.cnn = CNN()
        self.cnn = self.cnn.to(self.device)
        self.scene_mode = 'EDN'  # to be used in evaluation part
        self.resampling_dim = None  # this var stores resampling value for the time model is loaded to restore the value

    def forward(self, obs, prediction_truth=None, n_pred=None, center_line_dict=None, scene=None, sample_rate=None,
                pixel_scale=None, scene_funcs=None, file_name=None, rotated_scene=None, epoch=None, margin=None,
                iterations=None):
        """
        inputs: obs (tensor, shape=(n_batch, n_obs,n_nearby_objects(it depends on the scene),2) )
        n_pred is used for compatibility with LSTM model
        """
        batch_size = np.shape(obs)[0]
        # ----------------encoding --------------------
        # trajectory
        v = obs[:, 1:, 0, :] - obs[:, :-1, 0,
                               :]  # generate velocity from locations and exclude neighbors, shape=(n_obs,1,2)
        pixel_scale = pixel_scale.unsqueeze(1).unsqueeze(2)
        sample_rate = sample_rate.unsqueeze(1).unsqueeze(2)
        # v = v * pixel_scale * sample_rate  # normalize velocity to meters per second
        v[:, :, 0] = v[:, :, 0] + 18  # To bias data towords zero!
        v = v * sample_rate / pixel_scale  # normalize velocity to meters per second
        v = v.view(batch_size, 1, -1)  # input to the nn.linear should be : (N, *, in_features)
        encoded_traj = self.encoder_traj(v)  # output to the nn.linear is : (N, *, out_features)
        # scene
        scene_features = self.cnn(scene)
        scene_features = scene_features.view(batch_size, 1, 128)
        # Other vehicles
        """filter out the vehicles except the ones infront and left (As they are the ones that influence in roundabouts). Note we are in a normalized view of the scene.
        In the visualization (cv2), if we want to show (10,1000) we have to move 10 pixels down and 1000 forward. As the trajectories are normalized to the top
        veiw according to the visualizations, we keep the same method. So a vehicle infront of (a,b) means (a-20,b)
        """
        # select the forward and right vehicles
        obs_cpy = obs.clone()
        temp = 1 - (obs_cpy[:, -1:, 0:1, 0:1] >= obs_cpy[:, -1:, :,
                                                 0:1]).type(
            torch.int)  # put zeros for vehicles in front and ones for others
        temp_ = temp.repeat(1, obs_cpy.size(1), 1, obs_cpy.size(3)).type(torch.bool)
        obs_cpy[temp_] = -1000
        obs_cpy[obs_cpy != obs_cpy] = -1000  # replace nans with zeros
        # if (obs_cpy.size(2) < 4): obs_cpy = obs_cpy.repeat(1, 1, 3, 1)
        # select the 3 closest ones
        dist = (obs_cpy[:, -1] - obs_cpy[:, -1, 0:1, :]).pow(2).sum(
            dim=2)  # returns the dist between each point and veh of interest in the last observation point
        closest_dist, closest_indexes = torch.topk(dist, k=min(self.num_of_neighbors + 1, obs.size(2)), dim=1,
                                                   largest=False)  # find 3 smallest values (we find 4 as the first one is egovehicle)
        closest_dist = closest_dist[:, 1:]  # removing the egovehicle
        closest_indexes = closest_indexes[:, 1:]  # removing the egovehicle
        closest_nghbrs = torch.gather(obs[:, -3:], 2, closest_indexes.unsqueeze(1).unsqueeze(3).repeat(1, 3, 1,
                                                                                                       2))  # consider the last three observation for each vehicle
        vehs_rel_pos = obs[:, -1:, 0:1, :] - closest_nghbrs
        if (obs.size(2) == 1):  # there were no neighbors
            vehs_rel_pos = obs[:, -3:, 0:1]  # let's add the egovehicle and we will change it to 2000 next line
            closest_dist = obs[:, 0, 0, 0:1]
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0:1, :, -1:].repeat(1, 1, self.num_of_neighbors - obs.size(2), 1)], dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors - obs.size(2))],
                dim=1)
        elif (obs.size(
                2) < self.num_of_neighbors + 1):  # To handle the cases in test mode that does not have enough neighbors
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0, :, -1:].repeat(1, 1, self.num_of_neighbors + 1 - obs.size(2), 1)],
                dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors + 1 - obs.size(2))],
                dim=1)
            closest_dist = closest_dist.repeat(1, self.num_of_neighbors + 1 - obs.size(2))
        for j in range(batch_size):
            for i in range(vehs_rel_pos.size(2)):  # in every vehicle
                if (closest_dist[j, i] > 600000 or torch.isnan(vehs_rel_pos[j, :, i].sum())):
                    vehs_rel_pos[j, :, i] = 2000
        if (torch.sum(torch.isnan(vehs_rel_pos))):
            pdb.set_trace()
        vehs_rel_pos = vehs_rel_pos * (sample_rate.unsqueeze(3)) / (pixel_scale.unsqueeze(3))
        vehs_rel_pos[:, :, :, 0] = vehs_rel_pos[:, :, :, 0] - 10
        vehs_rel_pos[:, :, :, 1] = vehs_rel_pos[:, :, :, 1] - 7
        vehs_rel_pos = vehs_rel_pos.view(batch_size, 1, -1)  # flaten the tensor
        encoded_vehs = self.encoder_vehs(vehs_rel_pos)
        # ----------------decoding --------------------
        features_2 = self.decoder(torch.cat((encoded_traj, encoded_vehs, scene_features), 2))
        prediction_v_mean = self.regressor_mean(features_2).view(batch_size, self.M, self.n_pred, 2)
        prediction_v_var = self.regressor_var(features_2).view(batch_size, self.M, self.n_pred, 3)
        prediction_v_var[:, :, :, 0:2] = torch.exp(prediction_v_var[:, :, :, 0:2])
        prediction_v_var[:, :, :, 2:] = torch.tanh(prediction_v_var[:, :, :, 2:])
        prediction_v = torch.cat((prediction_v_mean, prediction_v_var), dim=3)
        prediction_speed = prediction_v_mean.pow(2).sum(dim=3).pow(0.5).unsqueeze(3)
        pixel_scale = pixel_scale.unsqueeze(3)
        sample_rate = sample_rate.unsqueeze(3)
        # prediction_v = prediction_v / (sample_rate * pixel_scale)
        prediction_v = prediction_v * pixel_scale / (sample_rate)
        prediction_speed = prediction_speed * pixel_scale / (sample_rate)
        pos_temp = obs[:, -1, 0:1, :].unsqueeze(1)
        # Generate trajectory positions given the predicted velocity.
        prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        for idx in range(self.n_pred):
            pos_temp = prediction_v[:, :, idx:idx + 1, 0:2] + pos_temp
            prediction_pos[:, :, idx:idx + 1] = pos_temp
        prediction_pos = torch.cat((prediction_pos, prediction_v_var), dim=3)

        return prediction_v, prediction_pos, torch.zeros([batch_size, self.M], device=obs.device), prediction_speed


class EDN_M(torch.nn.Module):
    def __init__(self, n_obs, n_pred, device):
        super(EDN_M, self).__init__()
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.num_of_neighbors = 3
        self.counter = 0
        self.M = 2
        self.encoder_traj = nn.Sequential(
            nn.Linear(2 * (n_obs - 1), 32),
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.encoder_vehs = nn.Sequential(
            nn.Linear(2 * 3 * self.num_of_neighbors, 32),
            # 2 for x y, 3 as we used relative coordinates for the last three observations
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 64 + 128, 128), nn.ReLU()
            # size is n_obs-1 because concerning velocity, the number of inputs are n_obs-1
        )
        self.regressor_mean = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, (n_pred * 2) * self.M)  # for predicting speed
        )
        self.regressor_var = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_pred * 3 * self.M)
        )
        self.prob_estimator = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, self.M)
        )

        self.device = device if device is not None else torch.device('cpu')
        self.cnn = CNN()
        self.cnn = self.cnn.to(self.device)
        self.scene_mode = 'EDN_M'  # to be used in evaluation part
        self.resampling_dim = None  # this var stores resampling value for the time model is loaded to restore the value

    def forward(self, obs, prediction_truth=None, n_pred=None, center_line_dict=None, scene=None, sample_rate=None,
                pixel_scale=None, scene_funcs=None, file_name=None, rotated_scene=None, epoch=None, margin=None,
                iterations=None):
        """
        inputs: obs (tensor, shape=(n_batch, n_obs,n_nearby_objects(it depends on the scene),2) )
        n_pred is used for compatibility with LSTM model
        """
        batch_size = np.shape(obs)[0]
        # ----------------encoding --------------------
        # trajectory
        v = obs[:, 1:, 0, :] - obs[:, :-1, 0,
                               :]  # generate velocity from locations and exclude neighbors, shape=(n_obs,1,2)
        pixel_scale = pixel_scale.unsqueeze(1).unsqueeze(2)
        sample_rate = sample_rate.unsqueeze(1).unsqueeze(2)
        v = v * sample_rate / pixel_scale  # normalize velocity to meters per second
        v = v.view(batch_size, 1, -1)  # input to the nn.linear should be : (N, *, in_features)
        encoded_traj = self.encoder_traj(v)  # output to the nn.linear is : (N, *, out_features)
        # scene
        scene_features = self.cnn(scene)
        scene_features = scene_features.view(batch_size, 1, 128)
        # Other vehicles
        """filter out the vehicles except the ones infront and left (As they are the ones that influence in roundabouts). Note we are in a normalized view of the scene.
        In the visualization (cv2), if we want to show (10,1000) we have to move 10 pixels down and 1000 forward. As the trajectories are normalized to the top
        veiw according to the visualizations, we keep the same method. So a vehicle infront of (a,b) means (a-20,b)
        """
        # select the forward and right vehicles
        obs_cpy = obs.clone()
        temp = 1 - (obs_cpy[:, -1:, 0:1, 0:1] >= obs_cpy[:, -1:, :,
                                                 0:1]).type(
            torch.int)  # put zeros for vehicles in front and ones for others
        temp_ = temp.repeat(1, obs_cpy.size(1), 1, obs_cpy.size(3)).type(torch.bool)
        obs_cpy[temp_] = -1000
        obs_cpy[obs_cpy != obs_cpy] = -1000  # replace nans with zeros
        # if (obs_cpy.size(2) < 4): obs_cpy = obs_cpy.repeat(1, 1, 3, 1)
        # select the 3 closest ones
        dist = (obs_cpy[:, -1] - obs_cpy[:, -1, 0:1, :]).pow(2).sum(
            dim=2)  # returns the dist between each point and veh of interest in the last observation point
        closest_dist, closest_indexes = torch.topk(dist, k=min(self.num_of_neighbors + 1, obs.size(2)), dim=1,
                                                   largest=False)  # find 3 smallest values (we find 4 as the first one is egovehicle)
        closest_dist = closest_dist[:, 1:]  # removing the egovehicle
        closest_indexes = closest_indexes[:, 1:]  # removing the egovehicle
        closest_nghbrs = torch.gather(obs[:, -3:], 2, closest_indexes.unsqueeze(1).unsqueeze(3).repeat(1, 3, 1,
                                                                                                       2))  # consider the last three observation for each vehicle
        vehs_rel_pos = obs[:, -1:, 0:1, :] - closest_nghbrs
        if (obs.size(2) == 1):  # there were no neighbors
            vehs_rel_pos = obs[:, -3:, 0:1]  # let's add the egovehicle and we will change it to 2000 next line
            closest_dist = obs[:, 0, 0, 0:1]
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0:1, :, -1:].repeat(1, 1, self.num_of_neighbors - obs.size(2), 1)], dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors - obs.size(2))],
                dim=1)
        elif (obs.size(
                2) < self.num_of_neighbors + 1):  # To handle the cases in test mode that does not have enough neighbors
            vehs_rel_pos = torch.cat(
                [vehs_rel_pos, vehs_rel_pos[0, :, -1:].repeat(1, 1, self.num_of_neighbors + 1 - obs.size(2), 1)],
                dim=2)
            vehs_rel_pos[:, :, obs.size(2) - 1:, :] = 2000
            closest_dist = torch.cat(
                [closest_dist, closest_dist[:, -1:].repeat(1, self.num_of_neighbors + 1 - obs.size(2))],
                dim=1)
            closest_dist = closest_dist.repeat(1, self.num_of_neighbors + 1 - obs.size(2))
        for j in range(batch_size):
            for i in range(vehs_rel_pos.size(2)):  # in every vehicle
                if (closest_dist[j, i] > 600000 or torch.isnan(vehs_rel_pos[j, :, i].sum())):
                    vehs_rel_pos[j, :, i] = 2000
        if (torch.sum(torch.isnan(vehs_rel_pos))):
            pdb.set_trace()
        # vehs_rel_pos = vehs_rel_pos * (pixel_scale.unsqueeze(3)) * (sample_rate.unsqueeze(3))
        vehs_rel_pos = vehs_rel_pos * (sample_rate.unsqueeze(3)) / (pixel_scale.unsqueeze(3))
        vehs_rel_pos = vehs_rel_pos.view(batch_size, 1, -1)  # flaten the tensor
        encoded_vehs = self.encoder_vehs(vehs_rel_pos)
        # ----------------decoding --------------------
        features_2 = self.decoder(torch.cat((encoded_traj, encoded_vehs, scene_features), 2))
        prediction_v_mean = self.regressor_mean(features_2).view(batch_size, self.M, self.n_pred, 2)
        prediction_v_var = self.regressor_var(features_2).view(batch_size, self.M, self.n_pred, 3)
        prediction_v_var[:, :, :, 0:2] = torch.exp(prediction_v_var[:, :, :, 0:2])
        prediction_v_var[:, :, :, 2:] = torch.tanh(prediction_v_var[:, :, :, 2:])
        prediction_v = torch.cat((prediction_v_mean, prediction_v_var), dim=3)
        prob = self.prob_estimator(features_2).squeeze(1)
        prediction_speed = prediction_v_mean.pow(2).sum(dim=3).pow(0.5).unsqueeze(3)
        pixel_scale = pixel_scale.unsqueeze(3)
        sample_rate = sample_rate.unsqueeze(3)
        prediction_v = prediction_v * pixel_scale / (sample_rate)
        prediction_speed = prediction_speed * pixel_scale / (sample_rate)
        pos_temp = obs[:, -1, 0:1, :].unsqueeze(1)
        # Generate trajectory positions given the predicted velocity.
        prediction_pos = torch.zeros([batch_size, self.M, self.n_pred, 2], device=obs.device)
        for idx in range(self.n_pred):
            pos_temp = prediction_v[:, :, idx:idx + 1, 0:2] + pos_temp
            prediction_pos[:, :, idx:idx + 1] = pos_temp
        prediction_pos = torch.cat((prediction_pos, prediction_v_var), dim=3)

        return prediction_v, prediction_pos, prob, prediction_speed
