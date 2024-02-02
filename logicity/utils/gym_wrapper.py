import gym
import torch
import numpy as np
from gym.spaces import Box, Dict
import torch.nn.functional as F

from ..core.config import *
from ..utils.find import find_nearest_building, find_building_mask

import logging
logger = logging.getLogger(__name__)

def CPU(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda() if isinstance(x, torch.Tensor) else x

class GymCityWrapper(gym.core.Env):
    def __init__(self, env):
        '''The Gym Wrapper of the CityEnv in single-agent mode.
        :param City env: the CityEnv instance
        '''        
        self.env = env
        self.logic_grounding_shape = self.env.logic_grounding_shape
        # self.observation_space = Dict({
        #     "map": Box(low=-1.0, high=1.0, shape=(3, self.fov, self.fov), dtype=np.float32),  # Adjust the shape as needed
        #     "position": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        # })
        self.observation_space = Box(low=-1.0, high=1.0, shape=(self.logic_grounding_shape, ), dtype=np.float32)
        self.last_dist = -1
        self.agent_name = env.rl_agent["agent_name"]
        self.max_horizon = env.rl_agent["max_horizon"]
        self.horizon = self.max_horizon
        self.agent_type = self.agent_name.split("_")[0]
        agent_id = self.agent_name.split("_")[1] # this is agent id in the yaml file
        for agent in self.env.agents:
            if agent.type == self.agent_type:
                if agent.id == int(agent_id):
                    self.agent = agent
                    self.agent_layer_id = agent.layer_id
        assert self.agent_layer_id is not None, "Agent not found! Recheck Your agent_name in the config file!"
        action_space = self.env.rl_agent["action_space"]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_space, ), dtype=np.float32)
        self.action_mapping = env.rl_agent["action_mapping"]
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.scale = [25, 7, 3.5, 8.3]
        self.mini_scale = [0, 0, -1, 0]
        self.t = 0
        
    def _flatten_obs(self, obs_dict):
        # Create a new image with a 0 background
        # neighborhood_obs = CPU(obs_dict["World"][0:3])
        # map_obs = np.ones((3, self.fov, self.fov), dtype=np.float32)
        # start_pos = CPU(self.agent.start)
        # cur_pos = CPU(self.agent.pos)
        # goal_pos = CPU(self.agent.goal)

        # # Calculate the region of the city image that falls within the agent's field of view
        # x_start = max(cur_pos[0] - self.fov//2, 0)
        # y_start = max(cur_pos[1] - self.fov//2, 0)
        # x_end = min(cur_pos[0] + self.fov//2, neighborhood_obs.shape[1])
        # y_end = min(cur_pos[1] + self.fov//2, neighborhood_obs.shape[2])

        # # Calculate where this region should be placed in the map_obs
        # new_x_start = max(self.fov//2 - cur_pos[0], 0)
        # new_y_start = max(self.fov//2 - cur_pos[1], 0)
        # new_x_end = new_x_start + (x_end - x_start)
        # new_y_end = new_y_start + (y_end - y_start)

        # # Place the part of the city image that's within the UAV's field of view into the map_obs
        # map_obs[:, new_x_start:new_x_end, new_y_start:new_y_end] = neighborhood_obs[:, x_start:x_end, y_start:y_end]
        # start_pos = np.asarray(start_pos, dtype=np.float32) / 240.
        # cur_pos = np.asarray(cur_pos, dtype=np.float32) / 240.
        # goal_pos = np.asarray(goal_pos, dtype=np.float32) / 240.
        # pos_data = np.concatenate([start_pos, cur_pos, goal_pos])

        return obs_dict["World_state"][0]
        
    def _get_reward(self, obs_dict):
        ''' Get the reward for the current step.
        :param dict obs_dict: the observation dictionary
        :return: the reward
        '''
        # cur_pos = self.agent.pos
        # goal_pos = self.agent.goal
        
        # dist_goal = np.abs(cur_pos - goal_pos).sum()
        # if self.last_dist == -1:
        #     self.last_dist = dist_goal
        # rew = (self.last_dist - dist_goal) * 5
        # self.last_dist = dist_goal
        # # print(cur_pos, goal_pos)
        # # print(cur_pos, obs_dict["World"][2].shape, obs_dict["World"][2][cur_pos[0], cur_pos[1]])
        # if self.agent_type == 'Pedestrian':
        #     count_on_road = len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == 1.0)[0]) + \
        #                         len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == -1)[0])
        # elif self.agent_type == 'Car':
        #     count_on_road = len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == 2.0)[0]) + \
        #                         len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == -1)[0])
        # # print(rew, 1-count_on_road)
        # rew -= (1-count_on_road)        # reward weighting
        # agent_action = obs_dict["Agent_actions"][0]
        # expert_action = obs_dict["Expert_actions"][0]
        # rew = 0
        # if (agent_action == expert_action).all():
        #     rew += 1
        # else:
        #     rew -= 5
        return obs_dict["Reward"][0] - 1/self.horizon
    
    
    def reset(self, return_info=False):
        logger.info("***Reset RL Agent in Env***")
        self.t = 0
        self.agent.init(self.env.city_grid)
        self.horizon = min(self.max_horizon, len(self.agent.global_traj))
        agent_code = self.type2label[self.agent_type]
        # draw agent
        # print('start: ', self.agent.start, 'pos: ', self.agent.pos)
        agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
        start = self.agent.start
        goal = self.agent.goal
        agent_layer[start[0], start[1]] = agent_code
        agent_layer[goal[0], goal[1]] = agent_code + AGENT_GOAL_PLUS
        self.env.city_grid[self.agent_layer_id] = agent_layer
        one_hot_action = torch.zeros_like(self.agent.action_dist, dtype=torch.float32)
        one_hot_action[-1] = 1
        ob_dict = self.env.update(one_hot_action, self.agent_layer_id)
        obs = self._flatten_obs(ob_dict)
        self.last_dist = -1
        self.last_pos = None
        return obs
    
    # def reinit(self): 
    #     agent_code = self.type2label[self.agent_type]
    #     # draw agent
    #     # print('start: ', self.agent.start, 'pos: ', self.agent.pos)
    #     agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
    #     start = self.agent.start
    #     goal = self.agent.goal
    #     agent_layer[start[0], start[1]] = agent_code
    #     agent_layer[goal[0], goal[1]] = agent_code + AGENT_GOAL_PLUS
    #     self.env.city_grid[self.agent_layer_id] = agent_layer
    #     assert len((agent_layer == agent_code).nonzero()) == 1, \
    #         ValueError("RL agent should be unique in the world matrix, now start is {}, goal is {}".format(start, goal))

    def step(self, action):
        self.t += 1
        index = np.argmax(action) # if action is a numpy array
        one_hot_action = torch.tensor(self.action_mapping[index], dtype=torch.float32)
        info = {}
        ob_dict = self.env.update(one_hot_action, self.agent_layer_id)
        # ob_dict = self.env.update()
        info.update(ob_dict)
        obs = self._flatten_obs(ob_dict)
        rew = self._get_reward(ob_dict)
        
        # offset the index by 3 layers 0,1,2 are static in world matrix
        done = self.agent.reach_goal
        if done:
            info["succcess"] = True
            rew += 2
            logger.info("will reset agent by success")
            self.reset()
        
        if self.t >= self.horizon: 
            done = True
            rew -= 2
            info["success"] = False
            info["overtime"] = True
            logger.info("Reset agent by overtime")
            self.reset()
            
        return obs, rew, done, info
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def check_success(self):
        return self.agent.reach_goal
    
    def seed(self, seed=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
    
