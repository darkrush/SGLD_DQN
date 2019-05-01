import torch
import numpy as np

class Memory(object):
    def __init__(self, limit,action_number, with_cuda):
        """Create Replay buffer.
        Parameters
        ----------
        limit: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.with_cuda = with_cuda
        self._storage = []
        self._maxsize = limit
        self._next_idx = 0
        self.data_buffer = {}
        self.data_buffer['actions'   ] = torch.zeros((limit,1),dtype = torch.long, device=torch.device('cuda'))
        self.data_buffer['rewards'   ] = torch.zeros((limit,1), device=torch.device('cuda'))
        self.data_buffer['terminals1'] = torch.zeros((limit,1), device=torch.device('cuda'))
    
    def nb_entries(self):
        return len(self._storage)
        
    def reset(self):
        self._storage = []
        self._next_idx = 0
        
    def __getitem(self, idx):
        return self._encode_sample([idx,])
        
    def __len__(self):
        return len(self._storage)  
    def append(self, obs0, action, reward, obs1, terminal1):
        data = (obs0,obs1)
        self.data_buffer['actions'][self._next_idx] = torch.as_tensor(action,dtype = torch.long, device=torch.device('cuda'))
        self.data_buffer['rewards'][self._next_idx] = torch.as_tensor(reward, device=torch.device('cuda'))
        self.data_buffer['terminals1'][self._next_idx] = torch.as_tensor(terminal1, device=torch.device('cuda'))
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    def _encode_sample(self, idxes):
        obses_t, obses_tp1 = [], []
        for i in idxes:
            data = self._storage[i]
            obs0, obs1 = data
            obses_t.append(np.array(obs0, copy=False))
            obses_tp1.append(np.array(obs1, copy=False))
        return_dict = {'obs0': torch.as_tensor(np.array(obses_t), device=torch.device('cuda')),'obs1':torch.as_tensor(np.array(obses_tp1), device=torch.device('cuda'))}
        batch_idxs = torch.as_tensor(idxes,dtype = torch.long, device=torch.device('cuda'))
        return_dict['rewards'] = torch.index_select(self.data_buffer['rewards'],0,batch_idxs)
        return_dict['actions'] = torch.index_select(self.data_buffer['actions'],0,batch_idxs)
        return_dict['terminals1'] = torch.index_select(self.data_buffer['terminals1'],0,batch_idxs)
        return return_dict
        #if self.with_cuda:
        #    return {key :  for key,value in sample_dict.items()}
        #else:
        #    return {key : torch.as_tensor(value,dtype = torch.long if key is 'actions' else torch.float32) for key,value in sample_dict.items()}
        
    def sample_last(self, batch_size):
        idxes = [len(self._storage)-1-i for i in range(batch_size)]
        return self._encode_sample(idxes)
     
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [np.random.random_integers(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)