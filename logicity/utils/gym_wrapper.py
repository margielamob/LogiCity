import gym
import torch
import numpy as np
from gym.spaces import Box, Dict
import torch.nn.functional as F
from ..core.config import *

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
        self.use_expert = env.rl_agent["use_expert"]
        for agent in self.env.agents:
            if agent.type == self.agent_type:
                if agent.id == int(agent_id):
                    self.agent = agent
                    self.agent_layer_id = agent.layer_id
        assert self.agent_layer_id is not None, "Agent not found! Recheck Your agent_name in the config file!"
        action_space = self.env.rl_agent["action_space"]
        if self.use_expert:
            self.expert_action = np.zeros(action_space, dtype=np.float32)
        else:
            self.expert_action = -1
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_space, ), dtype=np.float32)
        self.action_mapping = env.rl_agent["action_mapping"]
        self.max_priority = env.rl_agent["max_priority"]
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.scale = [25, 7, 3.5, 8.3]
        self.mini_scale = [0, 0, -1, 0]
        self.t = 0

    def full_action2one_hot(self, action):
        one_hot_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        # see agents/car.py
        if action[0] == 1:
            one_hot_action[0] = 1
            return one_hot_action
        elif action[4] == 1:
            one_hot_action[1] = 1
            return one_hot_action
        elif action[8] == 1:
            one_hot_action[2] = 1
            return one_hot_action
        else:
            one_hot_action[-1] = 1
            return one_hot_action
    
    def _flatten_obs(self, obs_dict):
        return obs_dict["World_state"][0]
        
    def _get_reward(self, obs_dict):
        ''' Get the reward for the current step.
        :param dict obs_dict: the observation dictionary
        :return: the reward
        '''
        if obs_dict["Fail"][0]:
            return -20
        else:
            moving_cost = self.action2cost(obs_dict["Agent_actions"][0])
            return moving_cost/self.horizon
    
    def action2cost(self, action):
        ''' Convert the action to cost.
        :param list action: the action list
        :return: the cost
        '''
        if action[0] == 1:
            # Slow
            return -2
        elif action[4] == 1:
            # Normal
            return -1
        elif action[8] == 1:
            # Fast
            return 0
        else:
            # Stop
            return -3
    
    
    def reset(self, return_info=False):
        logger.info("***Reset RL Agent in Env***")
        self.t = 0
        self.agent.init(self.env.city_grid)
        self.agent.reset_priority(self.max_priority)
        logger.info("Agent reset priority to {}/{}".format(self.agent.priority, self.max_priority))
        self.horizon = min(self.max_horizon, len(self.agent.global_traj))
        agent_code = self.type2label[self.agent_type]
        self.env.local_planner.reset()
        # draw agent
        # print('start: ', self.agent.start, 'pos: ', self.agent.pos)
        agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
        start = self.agent.start
        goal = self.agent.goal
        agent_layer[start[0], start[1]] = agent_code
        agent_layer[goal[0], goal[1]] = agent_code + AGENT_GOAL_PLUS
        self.env.city_grid[self.agent_layer_id] = agent_layer
        ob_dict = self.env.update(self.agent_layer_id)

        if self.use_expert:
            self.expert_action = self.full_action2one_hot(ob_dict["Expert_actions"][0])
        obs = self._flatten_obs(ob_dict)
        self.last_dist = -1
        self.last_pos = None
        self.current_obs = obs
        return self.current_obs

    def step(self, action):
        self.t += 1
        info = {}
        index = np.argmax(action) # if action is a numpy array
        one_hot_action = torch.tensor(self.action_mapping[index], dtype=torch.float32)
        # move and get reward
        current_obs = self.env.move_rl_agent(one_hot_action, self.agent_layer_id)
        rew = self._get_reward(current_obs)
        info.update(current_obs)
        new_ob_dict = self.env.update(self.agent_layer_id)
        if self.use_expert:
            self.expert_action = self.full_action2one_hot(new_ob_dict["Expert_actions"][0])
        # ob_dict = self.env.update()
        obs = self._flatten_obs(new_ob_dict)
        self.current_obs = obs
        
        # offset the index by 3 layers 0,1,2 are static in world matrix
        done = self.agent.reach_goal
        fail = current_obs["Fail"][0]

        if done:
            info["succcess"] = True
            logger.info("will reset agent by success")
            self.reset()
        
        if self.t >= self.horizon: 
            done = True
            rew -= 10
            info["success"] = False
            info["overtime"] = True
            logger.info("Reset agent by overtime")
            self.reset()
            
        if fail: 
            done = True
            info["success"] = False
            logger.info("Reset agent by failing")
            self.reset()

        return self.current_obs, rew, done, info
    
    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
    def save_episode(self):
        city_grid = self.env.city_grid.clone()
        agents = {}
        for agent in self.env.agents:
            name = agent.type + "_" + str(agent.id)
            agents[name] = {
                "start": agent.start.clone().numpy(),
                "goal": agent.goal.clone().numpy(),
                "concepts": agent.concepts,
                "pos": agent.pos.clone().numpy(),
                "layer_id": agent.layer_id,
                "id": agent.id,
            }
        episode = {
            "city_grid": city_grid,
            "agents": agents
        }
        return episode

    def check_success(self):
        return self.agent.reach_goal
    
    def seed(self, seed=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
    
