from __future__ import division

import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def final_l2(path1, path2):
    row1 = path1[-1]
    row2 = path2[-1]
    return np.linalg.norm((row2.x.item() - row1.x, row2.y.item() - row1.y))


def average_l2(path1, path2, n_predictions):
    assert len(path1) >= n_predictions
    assert len(path2) >= n_predictions
    path1 = path1[-n_predictions:]
    path2 = path2[-n_predictions:]  
    path1_np_x = np.zeros([1,n_predictions])
    path1_np_y = np.zeros([1,n_predictions])
    path2_np_x = np.zeros([1,n_predictions])
    path2_np_y = np.zeros([1,n_predictions])
    j = 0
    for r1, r2 in zip(path1, path2):
        path1_np_x[0,j] = r1.x
        path1_np_y[0,j] = r1.y
        path2_np_x[0,j] = r2.x
        path2_np_y[0,j] = r2.y
        j += 1
    return np.sum(np.sqrt((path1_np_x-path2_np_x)**2+(path1_np_y-path2_np_y)**2)) / n_predictions    
    #np.sum(np.sqrt(np.sum((r1.x - r2.x.item())**2,(r1.y - r2.y.item())**2),dim=1)))/1
    #return sum(np.linalg.norm((r1.x - r2.x.item(), r1.y - r2.y.item())) 
    #           for r1, r2 in zip(path1, path2)) / n_predictions



def cross_track(gt, pred, n_predictions):
    """ Extend prediction to le length of ground truth, and find the l2 distance between the 
        last points."""
    assert len(gt) >= n_predictions
    assert len(pred) >= n_predictions

    gt = gt[-n_predictions:]
    pred = pred[-n_predictions:]  
    gt_np = np.zeros([n_predictions,2])
    pred_np = np.zeros([n_predictions,2])
    j = 0
    for r1, r2 in zip(gt, pred):
        gt_np[j,0] = r1.x
        gt_np[j,1] = r1.y
        pred_np[j,0] = r2.x
        pred_np[j,1] = r2.y
        j += 1
    pred_speed = np.power(np.sum(np.power(pred_np[1:]-pred_np[:-1],2),1),0.5)
    gt_speed = np.power(np.sum(np.power(gt_np[1:]-gt_np[:-1],2),1),0.5)
    pred_speed_cum = pred_speed.cumsum()
    gt_speed_cum = gt_speed.cumsum()
    diff = abs(pred_speed_cum[-1] - gt_speed_cum[-1])
    if(pred_speed_cum[-1] < gt_speed_cum[-1]): #if ground-truth is longer than prediction, prediction should be extrapolated
        scale = diff/(pred_speed_cum[n_predictions-2]-pred_speed_cum[n_predictions-3]+0.01) + 1
        mapped_p = scale*pred_np[n_predictions-1] + (1-scale)*pred_np[n_predictions-2]                
    else:    
        for i in range(n_predictions-1): # minus one, because we have speed profile
            if(pred_speed_cum[n_predictions-2-i]<gt_speed_cum[-1]):
                diff = gt_speed_cum[-1] - pred_speed_cum[n_predictions-2-i]
                scale = diff/(pred_speed_cum[n_predictions-2-i+1]-pred_speed_cum[n_predictions-2-i]+0.01)#+1
                mapped_p = scale*pred_np[n_predictions-1-i+1] + (1-scale)*pred_np[n_predictions-1-i]                
                break
    if(gt_speed_cum[-1]<=pred_speed_cum[0]): #gt is smaller than all of them
        diff = gt_speed_cum[-1] - pred_speed_cum[0]
        scale = diff/(pred_speed_cum[0]+0.01)+1
        mapped_p = scale*pred_np[1] + (1-scale)*pred_np[0]                

    try:
        return np.power(np.sum(np.power(gt_np[-1] - mapped_p,2),0),0.5)
    except:
        pdb.set_trace()
        

    
    