import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class QNet(nn.Module):
    def __init__(self,obs_shape, nb_action, init_w=3e-3):
        super(QNet, self).__init__()
        self.obs_shape = obs_shape
        self.nb_action = nb_action
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, nb_action)
        
        
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())
        self.fc5.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)