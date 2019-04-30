import numpy as np
import os
import pickle
import torch
import gym
from copy import deepcopy

from atari_wrappers import wrap_deepmind, make_atari
from arguments import Singleton_arger
from logger import Singleton_logger
from evaluator import Singleton_evaluator

from DQN import DQN

class DQN_trainer(object):
    def __init__(self):
        train_args = Singleton_arger()['train']
        self.nb_epoch = train_args['nb_epoch']
        self.nb_cycles_per_epoch = train_args['nb_cycles_per_epoch']
        self.nb_rollout_steps = train_args['nb_rollout_steps']
        self.nb_iter_steps = train_args['nb_iter_steps']
        self.nb_warmup_steps = train_args['nb_warmup_steps']
        
    def setup(self):
        main_args = Singleton_arger()['main']
        Singleton_logger.setup(main_args['result_dir'],multi_process = main_args['multi_process'])

        Singleton_evaluator.setup(main_args['env'], logger = Singleton_logger, num_episodes = 10, model_dir = main_args['result_dir'], multi_process = main_args['multi_process'], visualize = False, rand_seed = main_args['rand_seed'])

        self.env = wrap_deepmind(make_atari(main_args['env']), frame_stack = True)

        if main_args['rand_seed']>= 0:
            self.env.seed(main_args['rand_seed'])
        
        self.obs_shape = self.env.observation_space.shape 
        self.nb_action = self.env.action_space.n
        self.agent = DQN()
        self.agent.setup(self.obs_shape,self.nb_action)
        self.result_dir = main_args['result_dir']
        self.reset()
        
    def reset(self):
        self.last_episode_length = 0
        self.current_episode_length = 0
        self.current_episode_reward = 0.
        self.last_episode_reward = 0.
        self.total_step = 0
        self.last_observation = self.env.reset()
        #self.agent.reset_noise()
        
    def warmup(self):
        for t_warmup in range(self.nb_warmup_steps):
            #pick action by actor randomly
            self.apply_action(np.random.random_integers(0, self.nb_action-1))
                
    def train(self):
        #50_epoch * 100_cycle * 2500_iter * 4_step
        for epoch in range(self.nb_epoch):
            #apply hyperparameter decay
            self.agent.before_epoch()
            self.agent.apply_lr_decay()
            for cycle in range(self.nb_cycles_per_epoch):
                self.agent.before_cycle()
                for iter in range(self.nb_iter_steps):
                    self.agent.before_iter()
                    for t_rollout in range(self.nb_rollout_steps):
                        self.apply_action(self.agent.select_action(s_t = self.last_observation, apply_noise = True))
                        self.total_step += 1
                    q_loss_mean = self.agent.update()
                    
                    
                self.agent.update_target()
                
                
                last_error = self.agent.calc_last_error()
                Singleton_logger.trigger_log('last_error', last_error,self.total_step)
                Singleton_logger.trigger_log('train_episode_length', self.last_episode_length,self.total_step)
                Singleton_logger.trigger_log('train_episode_reward', self.last_episode_reward,self.total_step)
                Singleton_logger.trigger_log('q_loss_mean', q_loss_mean, self.total_step)
                Singleton_logger.trigger_save()
                
                
            self.agent.save_model(self.result_dir)
            #trigger evaluation and log_save
            Singleton_evaluator.trigger_load_from_file(actor_dir = self.result_dir)
            Singleton_evaluator.trigger_eval_process(total_cycle = self.total_step)
            Singleton_logger.trigger_save()
        
        
    def apply_action(self, action):
        #apply the action to environment and get next state, reawrd and other information
        obs, reward, done, info = self.env.step(action)
        self.current_episode_reward += reward
        self.current_episode_length += 1
        #obs = deepcopy(obs)
        #store the transition into agent's replay buffer and update last observation
        self.agent.store_transition(self.last_observation, action,np.array([reward,]), obs, np.array([done,],dtype = np.float32))
        #self.last_observation = deepcopy(obs)
        self.last_observation = obs

        #if current episode is done
        if done:
            self.last_observation = self.env.reset()
            self.agent.reset_noise()
            self.last_episode_reward = self.current_episode_reward
            self.last_episode_length = self.current_episode_length
            self.current_episode_reward = 0.
            self.current_episode_length = 0
            
    def __del__(self):
        Singleton_evaluator.trigger_close()
        Singleton_logger.trigger_close()


if __name__ == "__main__":
    trainer = DQN_trainer()
    trainer.setup()
    trainer.warmup()
    trainer.train()
   