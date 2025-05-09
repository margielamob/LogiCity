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
        self.pred_grounding_index = self.env.pred_grounding_index
        # self.observation_space = Dict({
        #     "map": Box(low=-1.0, high=1.0, shape=(3, self.fov, self.fov), dtype=np.float32),  # Adjust the shape as needed
        #     "position": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        # })
        self.cat_length = env.rl_agent["cat_length"] if "cat_length" in env.rl_agent else False
        if self.cat_length:
            self.observation_space = Box(low=0.0, high=1.0, shape=(self.logic_grounding_shape + 1, ), dtype=np.float32)
        else:
            self.observation_space = Box(low=0.0, high=1.0, shape=(self.logic_grounding_shape, ), dtype=np.float32)
        self.last_dist = -1
        self.agent_name = env.rl_agent["agent_name"]
        self.horizon = env.rl_agent["max_horizon"]
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
        self.action_space = gym.spaces.Discrete(action_space)
        self.action_mapping = env.rl_agent["action_mapping"]
        self.max_priority = env.rl_agent["max_priority"]
        self.action_cost = env.rl_agent["action_cost"]
        self.reset_dist = env.rl_agent["reset_dist"] if "reset_dist" in env.rl_agent else None
        self.overtime_cost = env.rl_agent["overtime_cost"] if "overtime_cost" in env.rl_agent else -3
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.scale = [25, 7, 3.5, 8.3]
        self.mini_scale = [0, 0, -1, 0]
        self.t = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def full_action2index(self, action):
        # see agents/car.py
        if action[0] == 1:
            return 0
        elif action[4] == 1:
            return 1
        elif action[8] == 1:
            return 2
        else:
            return 3
    
    def _flatten_obs(self, obs_dict):
        if self.cat_length:
            return np.concatenate([obs_dict["World_state"][0], [self.normed_path_length]], axis=0, dtype=np.float32)
        else:
            return obs_dict["World_state"][0]
                
    def _get_reward(self, obs_dict):
        ''' Get the reward for the current step.
        :param dict obs_dict: the observation dictionary
        :return: the reward
        '''
        if obs_dict["Fail"][0]:
            # failing step do not normailze the reward
            return obs_dict["Reward"][0]
        else:
            moving_cost = self.action2cost(obs_dict["Agent_actions"][0])
            return (moving_cost + obs_dict["Reward"][0])/self.path_length
    
    def get_reward(self, obs_array, action):
        ''' Get the reward for the current step.
        :param np.array obs_array: the observation array
        :param int action: the action index
        :return: the reward
        '''
        # get the SAT reward/fail
        fail, sat_reward = self.env.local_planner.eval_state_action(obs_array, action)
        if fail:
            return sat_reward
        moving_cost = self.action2cost(action)
        return (moving_cost + sat_reward)/self.path_length

    def action2cost(self, action):
        ''' Convert the action to cost.
        :param list action: the action list
        :return: the cost
        '''
        if action[0] == 1:
            # Slow
            return self.action_cost[0]
        elif action[4] == 1:
            # Normal
            return self.action_cost[1]
        elif action[8] == 1:
            # Fast
            return self.action_cost[2]
        else:
            # Stop
            return self.action_cost[3]
    
    
    def reset(self, return_info=False):
        logger.info("***Reset RL Agent in Env***")
        self.t = 0
        self.agent.init(self.env.city_grid)
        self.agent.reset_concepts(self.max_priority, self.reset_dist)
        logger.info("Agent reset priority to {}/{}".format(self.agent.priority, self.max_priority))
        logger.info("Agent reset concepts to {}".format(self.agent.concepts))
        self.path_length = len(self.agent.global_traj)*4
        self.normed_path_length = len(self.agent.global_traj)/(2*self.agent.region)
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
        if return_info:
            episode = self.save_episode()
        ob_dict = self.env.update(self.agent_layer_id)
        obs = self._flatten_obs(ob_dict)
        self.current_obs = obs
        self.last_dist = -1
        self.last_pos = None
        self.current_episode_reward = 0
        self.current_episode_length = 0
        if self.use_expert:
            self.expert_action = self.full_action2index(ob_dict["Expert_actions"][0])
            if return_info:
                return self.current_obs, episode
            expert_info = {}
            expert_info["Next_grounding"] = ob_dict["Ground_dic"][0]
            expert_info["Next_sg"] = ob_dict["Expert_sg"][0]
            return self.current_obs, expert_info
        else:
            return self.current_obs
    
    def init(self):
        # init does not reset the agent
        logger.info("***Init RL Agent in Env***")
        self.t = 0
        self.path_length = len(self.agent.global_traj)*4
        self.normed_path_length = len(self.agent.global_traj)/(2*self.agent.region)
        self.env.local_planner.reset()
        ob_dict = self.env.update(self.agent_layer_id)
        if self.use_expert:
            self.expert_action = self.full_action2index(ob_dict["Expert_actions"][0])
        obs = self._flatten_obs(ob_dict)
        self.last_dist = -1
        self.last_pos = None
        self.current_obs = obs
        return self.current_obs
    
    def step(self, action):
        self.t += 1
        info = {}
        one_hot_action = torch.tensor(self.action_mapping[action], dtype=torch.float32)
        # move and get reward
        current_obs = self.env.move_rl_agent(one_hot_action, self.agent_layer_id)
        rew = self._get_reward(current_obs)
        info.update(current_obs)
        new_ob_dict = self.env.update(self.agent_layer_id)
        if self.use_expert:
            self.expert_action = self.full_action2index(new_ob_dict["Expert_actions"][0])
            info["Next_grounding"] = new_ob_dict["Ground_dic"][0]
            info["Next_sg"] = new_ob_dict["Expert_sg"][0]
        # ob_dict = self.env.update()
        self.current_episode_reward += rew
        self.current_episode_length += 1
        obs = self._flatten_obs(new_ob_dict)
        self.current_obs = obs
        
        # offset the index by 3 layers 0,1,2 are static in world matrix
        done = self.agent.reach_goal
        info["is_success"] = False

        if done:
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            info["is_success"] = True
            logger.info("will reset agent by success")
            self.reset()
        
        if self.t >= self.horizon: 
            done = True
            rew += self.overtime_cost
            info["overtime"] = True
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
            logger.info("Reset agent by overtime")
            self.reset()
            
        if info["Fail"][0]: 
            done = True
            logger.info("Reset agent by failing")
            info['episode'] = {'r': self.current_episode_reward, 'l': self.current_episode_length}
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
                "type": agent.type,
                "priority": agent.priority,
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
    
