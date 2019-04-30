import torch
import numpy as np

#class Memory(object):
#    def __init__(self, limit, with_cuda):
#        """Create Replay buffer.
#        Parameters
#        ----------
#        limit: int
#            Max number of transitions to store in the buffer. When the buffer
#            overflows the old memories are dropped.
#        """
#        self.with_cuda = with_cuda
#        self._storage = []
#        self._maxsize = limit
#        self._next_idx = 0
#    
#    
#    def nb_entries(self):
#        return len(self._storage)
#        
#    def reset(self):
#        self._storage = []
#        self._next_idx = 0
#        
#    def __getitem(self, idx):
#        return self._encode_sample([idx,])
#        
#    def __len__(self):
#        return len(self._storage)
#        
#    def append(self, obs0, action, reward, obs1, terminal1):
#        data = (obs0, action, reward, obs1, terminal1)
#
#        if self._next_idx >= len(self._storage):
#            self._storage.append(data)
#        else:
#            self._storage[self._next_idx] = data
#        self._next_idx = (self._next_idx + 1) % self._maxsize
#
#    def _encode_sample(self, idxes):
#        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
#        for i in idxes:
#            data = self._storage[i]
#            obs0, action, reward, obs1, terminal1 = data
#            obses_t.append(np.array(obs0, copy=False))
#            actions.append(np.array(action, copy=False))
#            rewards.append(reward)
#            obses_tp1.append(np.array(obs1, copy=False))
#            dones.append(terminal1)
#        sample_dict = {'obs0': np.array(obses_t), 'actions':np.array(actions), 'rewards':np.array(rewards), 'obs1':np.array(obses_tp1), 'terminals1':np.array(dones)}
#        if self.with_cuda:
#            return {key : torch.as_tensor(value,dtype = torch.long if key is 'actions' else torch.float32).cuda() for key,value in sample_dict.items()}
#        else:
#            return {key : torch.as_tensor(value,dtype = torch.long if key is 'actions' else torch.float32) for key,value in sample_dict.items()}
#        
#    def sample_last(self, batch_size):
#        idxes = [len(self._storage)-1-i for i in range(batch_size)]
#        return self._encode_sample(idxes)
#     
#    def sample(self, batch_size):
#        """Sample a batch of experiences.
#        Parameters
#        ----------
#        batch_size: int
#            How many transitions to sample.
#        Returns
#        -------
#        obs_batch: np.array
#            batch of observations
#        act_batch: np.array
#            batch of actions executed given obs_batch
#        rew_batch: np.array
#            rewards received as results of executing act_batch
#        next_obs_batch: np.array
#            next set of observations seen after executing act_batch
#        done_mask: np.array
#            done_mask[i] = 1 if executing act_batch[i] resulted in
#            the end of an episode and 0 otherwise.
#        """
#        idxes = [np.random.random_integers(0, len(self._storage) - 1) for _ in range(batch_size)]
#        return self._encode_sample(idxes)
        
        
        
import torch

class Memory(object):
    def __init__(self, limit, action_number, with_cuda):
        self.limit = limit
        self._next_entry = 0
        self._nb_entries = 0
        self.with_cuda = with_cuda
        
        self.data_buffer = {}
        self.data_buffer['obs0'      ] = torch.zeros((limit,4 ,84,84),dtype = torch.uint8)
        self.data_buffer['obs1'      ] = torch.zeros((limit,4 ,84,84),dtype = torch.uint8)
        self.data_buffer['actions'   ] = torch.zeros((limit,1),dtype = torch.long)
        self.data_buffer['rewards'   ] = torch.zeros((limit,1))
        self.data_buffer['terminals1'] = torch.zeros((limit,1))
        if self.with_cuda:
            for key,value in self.data_buffer.items():
                self.data_buffer[key] = self.data_buffer[key].cuda()

    def __getitem(self, idx):
        return {key: value[idx] for key,value in self.data_buffer.items()}
    
    def sample_last(self, batch_size):
        batch_idxs = torch.arange(self._next_entry - batch_size ,self._next_entry)%self._nb_entries
        if self.with_cuda:
            batch_idxs = batch_idxs.cuda()
        return_dict = {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}
        return_dict['obs0'] = torch.as_tensor(return_dict['obs0'], dtype=torch.float, device=torch.device('cuda'))
        return_dict['obs1'] = torch.as_tensor(return_dict['obs1'], dtype=torch.float, device=torch.device('cuda'))    
        return return_dict
    
    
    def sample(self, batch_size):
        batch_idxs = torch.randint(0,self._nb_entries, (batch_size,),dtype = torch.long)
        if self.with_cuda:
            batch_idxs = batch_idxs.cuda()
        return_dict = {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}
        return_dict['obs0'] = torch.as_tensor(return_dict['obs0'], dtype=torch.float, device=torch.device('cuda'))
        return_dict['obs1'] = torch.as_tensor(return_dict['obs1'], dtype=torch.float, device=torch.device('cuda'))
        return return_dict
    
    @property
    def nb_entries(self):
        return self._nb_entries
    
    def reset(self):
        self._next_entry = 0
        self._nb_entries = 0
        
    def append(self, obs0, action, reward, obs1, terminal1):
        self.data_buffer['obs0'][self._next_entry] = torch.as_tensor(obs0)
        self.data_buffer['obs1'][self._next_entry] = torch.as_tensor(obs1)
        self.data_buffer['actions'][self._next_entry] = torch.as_tensor(action,dtype = torch.long)
        self.data_buffer['rewards'][self._next_entry] = torch.as_tensor(reward)
        self.data_buffer['terminals1'][self._next_entry] = torch.as_tensor(terminal1)
        
        if self._nb_entries < self.limit:
            self._nb_entries += 1
            
        self._next_entry = (self._next_entry + 1)%self.limit