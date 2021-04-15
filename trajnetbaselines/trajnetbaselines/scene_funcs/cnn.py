import torch
import torch.nn.functional as F
import pdb
import torch.nn as nn
from .scene_funcs import resampler 

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride = 1) #**********(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride = 1) #**********(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride = 1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride = 1)              
        self.conv5 = torch.nn.Conv2d(32, 64, 3, stride = 1)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, stride = 1)
        self.fc1 = torch.nn.Linear(1408, 128) #(in_features, out_features, bias=True)
        self.dropout = torch.nn.Dropout2d(p=0.2)
    def forward(self, input):
        batch_size = input.size()[0]
        x = input.clone() # to make sure data is at the same size
        x = torch.unsqueeze(x,1)        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        return x
		