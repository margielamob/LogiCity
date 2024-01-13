import gym
import torch
import numpy as np
from gym.spaces import Box, Dict
import torch.nn.functional as F

from ..core.config import *
from ..utils.find import find_nearest_building, find_building_mask

import logging

logger = logging.getLogger(__name__)
TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def CPU(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda() if isinstance(x, torch.Tensor) else x

class GymCityWrapper(gym.core.Env):
    def __init__(self, env, agent_id: int=3, horizon: int=500, ped_idx=[3, 25], car_idx=[25, 35]):
        '''The Gym Wrapper of the CityEnv in single-agent mode.
        :param City env: the CityEnv instance
        :param int agent_id: the agent id
        '''        
        self.env = env
        self.fov = AGENT_FOV*2
        self.observation_space = Dict({
            "map": Box(low=-1.0, high=1.0, shape=(3, 50, 50), dtype=np.float32),  # Adjust the shape as needed
            "position": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(5, ), dtype=np.float32)
        self.n_agents = len(env.agents)
        self.ped_idx = [i+3 for i in range(self.n_agents) if env.agents[i].type == "Pedestrian"]
        self.car_idx = [i+3 for i in range(self.n_agents) if env.agents[i].type == "Car"]
        self.num_ped = len(self.ped_idx)
        self.num_car = len(self.car_idx)
        self.last_dist = -1
        self.agent_name = "Pedestrian_{}".format(agent_id) if agent_id in self.ped_idx \
                        else "Car_{}".format(agent_id)
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id-3]
        
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.scale = [25, 7, 3.5, 8.3]
        self.mini_scale = [0, 0, -1, 0]
        self.horizon = horizon
        self.t = 0
        
    def _flatten_obs(self, obs_dict):
        # Create a new image with a 0 background
        # TODO: may need layers from other agents
        neighborhood_obs = CPU(obs_dict["World"][0:3])
        map_obs = np.ones((3, self.fov, self.fov), dtype=np.float32)
        start_pos = CPU(self.agent.start)
        cur_pos = CPU(self.agent.pos)
        goal_pos = CPU(self.agent.goal)

        # Calculate the region of the city image that falls within the agent's field of view
        x_start = max(cur_pos[0] - self.fov//2, 0)
        y_start = max(cur_pos[1] - self.fov//2, 0)
        x_end = min(cur_pos[0] + self.fov//2, neighborhood_obs.shape[1])
        y_end = min(cur_pos[1] + self.fov//2, neighborhood_obs.shape[2])

        # Calculate where this region should be placed in the map_obs
        new_x_start = max(self.fov//2 - cur_pos[0], 0)
        new_y_start = max(self.fov//2 - cur_pos[1], 0)
        new_x_end = new_x_start + (x_end - x_start)
        new_y_end = new_y_start + (y_end - y_start)

        # Place the part of the city image that's within the UAV's field of view into the map_obs
        map_obs[:, new_x_start:new_x_end, new_y_start:new_y_end] = neighborhood_obs[:, x_start:x_end, y_start:y_end]
        start_pos = np.asarray(start_pos, dtype=np.float32) / 240.
        cur_pos = np.asarray(cur_pos, dtype=np.float32) / 240.
        goal_pos = np.asarray(goal_pos, dtype=np.float32) / 240.
        pos_data = np.concatenate([start_pos, cur_pos, goal_pos])

        return {
                "map": map_obs,
                "position": pos_data
            }   
        
    def _get_reward(self, obs_dict):
        ''' Get the reward for the current step.
        :param dict obs_dict: the observation dictionary
        :return: the reward
        '''
        cur_pos = self.env.agents[self.agent_id-3].pos
        goal_pos = self.env.agents[self.agent_id-3].goal
        
        dist_goal = np.linalg.norm(cur_pos - goal_pos, ord=1, axis=0)
        if self.last_dist == -1:
            self.last_dist = dist_goal
        rew = self.last_dist - dist_goal
        self.last_dist = dist_goal
        # print(cur_pos, goal_pos)
        # print(cur_pos, obs_dict["World"][2].shape, obs_dict["World"][2][cur_pos[0], cur_pos[1]])
        if self.agent.type == 'Pedestrian':
            count_on_road = len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == 1.0)[0]) + \
                                len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == -1)[0])
        elif self.agent.type == 'Car':
            count_on_road = len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == 2.0)[0]) + \
                                len(np.where(obs_dict["World"][2][cur_pos[0], cur_pos[1]] == -1)[0])
        # print(rew, 1-count_on_road)
        rew -= (1-count_on_road)        # reward weighting
        return rew
    
    
    def reset(self, return_info=False):
        self.t = 0
        # TODO: reset functions!
        WALKING_STREET = TYPE_MAP['Walking Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        self.agent.init(self.env.city_grid, rl_agent=True)
        self.reinit()
        print("=============")
        print("Reset Agent")
        ob_dict = self.env.update(torch.from_numpy(np.array([0, 0, 0, 0, 1])), self.agent_id)
        obs = self._flatten_obs(ob_dict)
        self.last_dist = -1
        self.last_pos = None
        return obs
    
    def reinit(self): 
        agent_code = self.type2label[self.agent.type]
        # draw agent
        # print('start: ', self.agent.start, 'pos: ', self.agent.pos)
        agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
        agent_layer[self.agent.start[0], self.agent.start[1]] = agent_code
        agent_layer[self.agent.goal[0], self.agent.goal[1]] = agent_code + AGENT_GOAL_PLUS
        self.env.city_grid[self.agent_id] = agent_layer

    def step(self, action):
        self.t += 1
        index = np.argmax(action) # if action is a numpy array
        one_hot_action = torch.zeros(self.action_space.shape[0])
        one_hot_action[index] = 1.
        one_hot_action = one_hot_action.float()
        # assert len(action.shape) == 1, "Action must be a 1D array!"
        info = {}
        ob_dict = self.env.update(one_hot_action, self.agent_id)
        # ob_dict = self.env.update()
        info.update(ob_dict)
        obs = self._flatten_obs(ob_dict)
        rew = self._get_reward(ob_dict)
        
        # offset the index by 3 layers 0,1,2 are static in world matrix
        done = self.agent.reach_goal
        if done:
            info["succcess"] = True
            rew += 10
            self.agent.init(self.env.city_grid, rl_agent=True)
            self.reinit()
            print("Reset agent by success")
        if self.agent.pos[0] <= 0 or self.agent.pos[1] <= 0 or \
            self.agent.pos[0] >= 240 or self.agent.pos[1] >= 240: 
            info["success"] = False
            done = True
            rew -= 10
            self.agent.init(self.env.city_grid, rl_agent=True)
            self.reinit()
            print("Reset agent by oor")
        
        if self.t >= self.horizon: 
            done = True
            info["success"] = False
            info["overtime"] = True
            self.agent.init(self.env.city_grid, rl_agent=True)
            self.reinit()
            print("Reset agent by overtime")
            
        return obs, rew, done, info
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def check_success(self):
        return self.env.agents[self.agent_id-3].reach_goal
    
    def seed(self, seed=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
    
