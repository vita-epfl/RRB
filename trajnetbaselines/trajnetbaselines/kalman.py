import numpy as np
import pykalman
import trajnettools
import torch
import pdb

def predict(paths, n_obs, file_name=None, sample_rate=None, pixel_scale=None, scene_funcs=None, timestamp=None, store_image=0,center_line=None ):
    constant_vel = 1
    n_frames = len(paths[0])
    path = paths[0]
    # prepare predictions
    frame_diff = path[1].frame - path[0].frame
    first_frame = path[n_obs].frame
    ped_id = path[-1].pedestrian
    if (constant_vel):
        path_np = np.zeros([15,2])
        j = 0
        for r1 in path:
            path_np[j,0] = r1.x
            path_np[j,1] = r1.y
            j += 1
        pos_temp = path_np[4, :]
        last_vel = path_np[4, :] - path_np[3, :]
        predictions = np.zeros([10, 2])
        for idx in range(10):
            pos_temp = last_vel + pos_temp
            predictions[idx:idx + 1] = pos_temp
        scale = float(scene_funcs.pixel_scale_dict[file_name])
        offset = scene_funcs.offset_dict[file_name]
        preds = predictions.copy()
        preds[:,1] = scale*(preds[:,1]-offset[0]) #second dimension is the longer axes, horizontal one
        preds[:,0] = -scale*(preds[:,0]-offset[1])
        scene_violation = offroad_detector(preds, file_name, scene_funcs.image) 
        return [trajnettools.TrackRow(first_frame + i * frame_diff, ped_id, x, y)
                for i, (x, y) in enumerate(predictions)],0, scene_violation, 0

    else:
        initial_state_mean = [path[0].x, 0, path[0].y, 0]

        transition_matrix = [[1, 1, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0],
                              [0, 0, 1, 0]]

        kf = pykalman.KalmanFilter(transition_matrices=transition_matrix,
                                   observation_matrices=observation_matrix,
                                   transition_covariance=1e-5 * np.eye(4),
                                   observation_covariance=0.05**2 * np.eye(2),
                                   initial_state_mean=initial_state_mean)
        # kf.em([(r.x, r.y) for r in path[:9]], em_vars=['transition_matrices',
        #                                                'observation_matrices'])
        kf.em([(r.x, r.y) for r in path[:n_obs]])
        observed_states, _ = kf.smooth([(r.x, r.y) for r in path[:n_obs]])



        # sample predictions (first sample corresponds to last state)
        # average 5 sampled predictions
        predictions = None
        for _ in range(5):
            _, pred = kf.sample(n_frames - n_obs + 1 , initial_state=observed_states[-1]) # I don't know why but sven didn't consider the first sample so we have one more sample and in the last line we start from the sample 1.
            if predictions is None:
                predictions = pred
            else:
                predictions += pred
        predictions /= 5.0                
        preds = predictions.data[1:]
        scale = float(scene_funcs.pixel_scale_dict[file_name])
        offset = scene_funcs.offset_dict[file_name]
        preds = preds.copy()
        preds[:,1] = scale*(preds[:,1]-offset[0]) #second dimension is the longer axes, horizontal one
        preds[:,0] = -scale*(preds[:,0]-offset[1])
        scene_violation = offroad_detector(preds, file_name, scene_funcs.image) 
        return [trajnettools.TrackRow(first_frame + i * frame_diff, ped_id, x, y)
                for i, (x, y) in enumerate(predictions[1:])],0, scene_violation, 0

def predict_modified(xy, file_name=None, sample_rate=None, pixel_scale=None, scene_funcs=None, timestamp=None, store_image=0 ):

    n_frames = 15 
    n_obs = 5
    initial_state_mean = [xy[0,0], 0, xy[0,1], 0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    kf = pykalman.KalmanFilter(transition_matrices=transition_matrix,
                               observation_matrices=observation_matrix,
                               transition_covariance=1e-5 * np.eye(4),
                               observation_covariance=0.05**2 * np.eye(2),
                               initial_state_mean=initial_state_mean)
    # kf.em([(r.x, r.y) for r in path[:9]], em_vars=['transition_matrices',
    #                                                'observation_matrices'])
    #pdb.set_trace()
    kf.em([(r[0], r[1]) for r in xy[1:n_obs].cpu().numpy()])
    observed_states, _ = kf.smooth([(r[0], r[1]) for r in xy[1:n_obs].cpu().numpy()])

    # sample predictions (first sample corresponds to last state)
    # average 5 sampled predictions
    predictions = None
    for _ in range(3):
        _, pred = kf.sample(n_frames - n_obs + 1 , initial_state=observed_states[-1]) # I don't know why but sven didn't consider the first sample so we have one more sample and in the last line we start from the sample 1.
        if predictions is None:
            predictions = pred
        else:
            predictions += pred
    predictions /= 3.0
    #pdb.set_trace()
    return torch.tensor(predictions[1:].data).cuda()
    
    


def offroad_detector (prediction, file_name, scene_info, prob=None):        
        """It doesn't need to be devided by 10 since the scene is read from numpy file not the image """
        prediction_clone = torch.tensor([prediction])[0]
        cnt = 0
        for seq in prediction_clone: #between n_obs + n_pred data points that we have                
            #if(scene_info[file_name][(seq[0].astype(int)[0],seq[0].astype(int)[1])]==1):
            if(seq[0]>=scene_info[file_name].size(0) or seq[1]>=scene_info[file_name].size(1)):
                cnt += 1  #Since the speed varies and margin is not enough is some cases  #cnt[i] += 1  
            elif(scene_info[file_name][(seq[0].type(torch.LongTensor)),(seq[1].type(torch.LongTensor))]==1):
                cnt += 1
                    
        return cnt