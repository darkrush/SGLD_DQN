import torch
import numpy as np

class Memory(object):
    def __init__(self, limit, with_cuda):
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
    
    @property
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
        data = (obs0, action, reward, obs1, terminal1)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs0, action, reward, obs1, terminal1 = data
            obses_t.append(np.array(obs0, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs1, copy=False))
            dones.append(terminal1)
        sample_dict = {'obs0': np.array(obses_t), 'actions':np.array(actions), 'rewards':np.array(rewards), 'obs1':np.array(obses_tp1), 'terminals1':np.array(dones)}
        if self.with_cuda:
            return {key : torch.as_tensor(value,dtype = torch.long if key is 'actions' else torch.float32).cuda() for key,value in sample_dict.items()}
        else:
            return {key : torch.as_tensor(value,dtype = torch.long if key is 'actions' else torch.float32) for key,value in sample_dict.items()}
        
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