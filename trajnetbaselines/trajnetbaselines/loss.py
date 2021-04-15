import math
import torch
import pdb
from .scene_funcs.cnn import CNN
from .scene_funcs.scene_funcs import scene_funcs
import numpy as np



class PredictionLoss_L2(torch.nn.Module):  # the class is inherited from torch.nn.Module
    """L2 loss
    """
    def __init__(self, scene_funcs=None):
        super(PredictionLoss_L2, self).__init__()
 
    def forward(self, inputs, targets, temp1=0, temp2=0, file_name=None):
        loss = torch.sum(abs(inputs[:,:,0:2]-targets))
        #loss = torch.sum((inputs[:,:,0:2]-targets)**2)

        return(loss)


       
class gaussian_2d(torch.nn.Module):  # the class is inherited from torch.nn.Module
    """ This supports backward().
        Insprired by
        https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/blob/master/loss_functions.py
        x1 stands for x, x2 stands for y
        """
    def __init__(self, scene_funcs=None):
        super(gaussian_2d, self).__init__()
        self.background_rate = 0.2
    def gaussian_func (self, mu1mu2s1s2, x1x2, epoch, prob=None):
        x1, x2 = x1x2[:, :, 0], x1x2[:, :, 1]
        if(prob is not None): #Meaning we have multiple modes
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            mu1, mu2, s1, s2, rho = (
            #mu1, mu2, s1, s2 = (
                mu1mu2s1s2[:, :, :, 0],
                mu1mu2s1s2[:, :, :, 1],
                mu1mu2s1s2[:, :, :, 2],
                mu1mu2s1s2[:, :, :, 3],
                mu1mu2s1s2[:, :, :, 4], 
            )
        else:    
            mu1, mu2, s1, s2, rho = (
            #mu1, mu2, s1, s2 = (
                mu1mu2s1s2[:, :, 0],
                mu1mu2s1s2[:, :, 1],
                mu1mu2s1s2[:, :, 2],
                mu1mu2s1s2[:, :, 3],
                mu1mu2s1s2[:, :, 4],
            )
        threshold = 0.01
        #pdb.set_trace()
        if (epoch>5):
            self.background_rate = 0.1 
        if (epoch>10):
            self.background_rate = 0.0 
        s1 = s1.clamp(min=threshold, max=100)
        s2 = s2.clamp(min=threshold, max=100)    
        rho = rho.clamp(min=-0.99, max=0.99)    
        norm1 = x1 - mu1
        norm2 = x2 - mu2
        sigma1sigma2 = (s1*s2).clamp(min=threshold,max=10000)
        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2
        #z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2
        z = z.clamp(min=0.01)
        if(prob is None):
            loss = z / 2  + torch.log(2 * math.pi * sigma1sigma2)
        else:
            loss = prob * torch.sum(z / 2  + torch.log(2 * math.pi * sigma1sigma2),dim=2)
        loss = z / (2* (1 - rho ** 2))  + torch.log(2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2))
        if(torch.sum(torch.isnan(loss))!=0):
            pdb.set_trace()
        return loss
        
    def forward(self, inputs, targets, epoch, prob=None):
        inputs_bg = inputs.clone()
        if(prob is None):
            inputs_bg[:, :, 2] = 40.0  # sigma_x
            inputs_bg[:, :, 3] = 40.0  # sigma_y
        else:
            inputs_bg[:, :, :, 2] = 40.0  # sigma_x
            inputs_bg[:, :, :, 3] = 40.0  # sigma_y
        values = self.background_rate * self.gaussian_func(inputs_bg, targets,epoch, prob) + (1 - self.background_rate) * self.gaussian_func(inputs, targets, epoch, prob)
        
        return torch.sum(values)

class PredictionLoss_L2_sceneLoss(torch.nn.Module):  # the class is inherited from torch.nn.Module
    """L2 loss
    """
    def __init__(self, scene_funcs):
        super(PredictionLoss_L2_sceneLoss, self).__init__()
        self.scene_funcs = scene_funcs
        self.sceneLoss_scale = 10
    def forward(self, inputs, targets, start_pred_point, file_name):
        loss_L1 = torch.sum(abs(inputs[:,:,0:2]-targets))
        position = (inputs[:,:,0:2]+ torch.unsqueeze(start_pred_point,1))
        gama = 5 
        scene_map = self.scene_funcs.scene_info
        loss_scene = []
        loss_scene_total = 0
        for i in range(inputs.size()[0]): #i in batch_size
            for _,pos in enumerate(position[i]):
                detached_pos = pos.detach().type(torch.int32)
                detached_pos = torch.min(torch.max(detached_pos,torch.tensor([0,0],  dtype=torch.int32,device=inputs.device)),torch.tensor([2159,3839], dtype=torch.int32, device=inputs.device))
                '''if(detached_pos[0]<0):
                    detached_pos[0] = 0
                elif(detached_pos[0]>2159):
                    detached_pos[0] = torch.tensor(2159)
                if(detached_pos[1]<0):
                    detached_pos[1] = 0
                elif(detached_pos[1]>3839):
                    detached_pos[1] = torch.tensor(3839)  '''              
                #detached_pos[0] = torch.tensor(detached_pos[0]>0)*detached_pos[0] #if prediction was off, prevent negative values
                #detached_pos[0] = torch.tensor(detached_pos[0]>2160)*torch.tensor(2160)+torch.tensor(1-detached_pos[0]>2160)*detached_pos[0] #if prediction was off, prevent negative values
                #detached_pos[1] = torch.tensor(detached_pos[1]>0)*detached_pos[1]           
                #detached_pos[1] = torch.tensor(detached_pos[1]>3840)*torch.tensor(3840)+torch.tensor(1-detached_pos[1]>3840)*detached_pos[1] #if prediction was off, prevent negative values
                if(scene_map[file_name[i]][tuple(detached_pos)]==1): #if prediction is outside the road       
                    nearest_onroad_point = (self.scene_funcs.nearest_point_in_road(locc=detached_pos, start_pred_point=start_pred_point,file=file_name[i]))                
                    loss_scene_temp = abs(pos- nearest_onroad_point.type(torch.float32))*self.sceneLoss_scale
                    loss_scene.append(loss_scene_temp)                    
                    loss_scene_total = loss_scene_total + sum(loss_scene_temp)
                else:
                    loss_scene.append(0)    
        
        '''for i,pos in enumerate(position):
            position_mat = torch.zeros(np.shape(scene_map))
            position_mat[pos] = 1
            loss_scene.append(torch.sum(scene_map*position_mat))'''
        #scene_invalid = [self.scene_funcs.scene_data(i, check_valid = True) for i in position] #detecting invalid predictions (the ones outside the road)
        #loss_scene =  torch.sum(scene_invalid )

        return(loss_L1 + loss_scene_total*gama)
        
        
