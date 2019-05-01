import io
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from arguments import Singleton_arger

from model import QNet
from memory import Memory

class DQN(object):
    def __init__(self):
        agent_args = Singleton_arger()['agent']
        self.critic_lr = agent_args['critic_lr']
        self.lr_decay = agent_args['lr_decay']
        self.l2_critic = agent_args['l2_critic']
        self.batch_size = agent_args['batch_size']
        self.discount = agent_args['discount']
        self.tau = agent_args['tau']
        self.with_cuda = agent_args['with_cuda']
        self.buffer_size = int(agent_args['buffer_size'])
        self.num_update_time = 10
        
    def setup(self,obs_shape,nb_action):
        self.lr_coef = 1
        self.epsilon = 1
        self.nb_action = nb_action
        model_args = Singleton_arger()['model']
        
        qnet  = QNet(obs_shape,nb_action)
        
        self.qnet         = copy.deepcopy(qnet)
        self.target_qnet  = copy.deepcopy(qnet)
        self.memory = Memory(self.buffer_size,nb_action, self.with_cuda)
        
        if self.with_cuda:
            self.qnet.cuda()
            self.target_qnet.cuda()
        
        self.qnet_optim  = Adam(self.qnet.parameters(), lr=self.critic_lr)
        
    def reset_noise(self):
        pass
        
    def before_epoch(self):
        pass
    
    def before_cycle(self):
        pass
        
    def before_iter(self):
        self.epsilon =max((self.epsilon- (1-0.01)/250000),0.01)   
    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        #s_t = torch.tensor(s_t,dtype = torch.float32,requires_grad = False)
        self.memory.append(s_t, a_t, r_t, s_t1, done_t)
        
    def update_target(self):
        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)         
    def update(self):
        batch = self.memory.sample(self.batch_size)
        tensor_obs0 = batch['obs0'].float()
        tensor_obs1 = batch['obs1'].float()
        self.qnet_optim.zero_grad()

        q_eval = self.qnet(tensor_obs0).gather(1,batch['actions'])
        with torch.no_grad():
            _ , a_next = self.qnet(tensor_obs1).max(1)
            q_next = self.target_qnet(tensor_obs1).gather(1, a_next.unsqueeze(1))
            q_target = batch['rewards'].float() + self.discount * (1-batch['terminals1']) * q_next
        value_loss = nn.functional.mse_loss(q_eval, q_target)
        
        value_loss.backward()
        self.qnet_optim.step()
        return value_loss.item()
    
    def calc_last_error(self):
        # Sample batch
        batch = self.memory.sample_last(self.batch_size)
        tensor_obs0 = batch['obs0'].float()
        tensor_obs1 = batch['obs1'].float()
        # Prepare for the target q batch
        with torch.no_grad():
            q_eval = self.qnet(tensor_obs0).gather(1, batch['actions'])
            _ , a_next = self.qnet(tensor_obs1).max(1)
            q_next = self.target_qnet(tensor_obs1).gather(1, a_next.unsqueeze(1))
            q_target = batch['rewards'].float() + self.discount * (1-batch['terminals1']) * q_next
            value_loss = nn.functional.mse_loss(q_eval, q_target)
        return value_loss.item()
        
    def apply_lr_decay(self):
        if self.lr_decay > 0:
            self.lr_coef = self.lr_decay*self.lr_coef/(self.lr_coef+self.lr_decay)
            for group in self.qnet_optim.param_groups:
                group['lr'] = self.critic_lr * self.lr_coef
                    
    def select_action(self, s_t, apply_noise):
        if apply_noise and np.random.rand()<self.epsilon: 
            return np.random.random_integers(0, self.nb_action-1)
        s_t = torch.tensor(np.expand_dims(np.array(s_t), axis = 0),dtype = torch.float32,requires_grad = False)
        if self.with_cuda:
            s_t = s_t.cuda()
        with torch.no_grad():
            q_value = self.qnet(s_t)
        action = np.argmax(q_value.cpu().numpy().squeeze(0))
        return action
        
    def load_weights(self, output): 
        self.qnet  = torch.load('{}/qnet.pkl'.format(output) )
            
    def save_model(self, output):
        torch.save(self.qnet ,'{}/qnet.pkl'.format(output) )
            
    def get_qnet_buffer(self):
        qnet_buffer = io.BytesIO()
        torch.save(self.qnet, qnet_buffer)
        return qnet_buffer
        
        
if __name__ == "__main__":
    pass