import os
import sys
import time
import yaml
import torch
import argparse
import importlib
import numpy as np
import pickle as pkl
from tqdm import trange
from logicity.core.config import *
from logicity.utils.load import CityLoader
from logicity.utils.logger import setup_logger
from logicity.utils.vis import visualize_city
# RL
from logicity.rl_agent.alg import *
from logicity.utils.gym_wrapper import GymCityWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from logicity.utils.gym_callback import EvalCheckpointCallback

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log_rl")
    parser.add_argument('--exp', type=str, default="expert_40episode_val")
    parser.add_argument('--vis', action='store_true', help='Visualize the city.')
    # seed
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max_episodes', type=int, default=40)
    # RL
    parser.add_argument('--config', default='config/tasks/Nav/medium/experts/expert_episode_val.yaml', help='Configure file for this RL exp.')

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def make_env(simulation_config, return_cache=False): 
    # Unpack arguments from simulation_config and pass them to CityLoader
    city, cached_observation = CityLoader.from_yaml(**simulation_config)
    env = GymCityWrapper(city)
    if return_cache: 
        return env, cached_observation
    else:
        return env
    
def make_envs(simulation_config, rank):
    """
    Utility function for multiprocessed env.
    
    :param simulation_config: The configuration for the simulation.
    :param rank: Unique index for each environment to ensure different seeds.
    :return: A function that creates a single environment.
    """
    def _init():
        env = make_env(simulation_config)
        env.seed(rank + 1000)  # Optional: set a unique seed for each environment
        return env
    return _init

def main(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    # RL config
    rl_config = config['stable_baselines']
    logger.info("RL config: {}".format(rl_config))

    # Dynamic import of the RL algorithm
    algorithm_class = dynamic_import(
        "logicity.rl_agent.alg",  # Adjust the module path as needed
        rl_config["algorithm"]
    )
    
    # Checkpoint evaluation
    rew_list = []
    worlds = []
    success = []
    all_episodes = {}
    key = 0
    vis_id = [0, 1, 2, 3, 4]
    # 0: Slow, 1: Normal, 2: Fast, 3: Stop
    # Test
    # num_desired = {
    #     'police':{
    #         0: 10,
    #         2: 10,
    #         3: 5
    #     },
    #     'ambulance':{
    #         0: 5,
    #         3: 10
    #     },
    #     'reckless':{
    #         2: 10,
    #         3: 5
    #     },
    #     'bus':{
    #         2: 5,
    #         3: 10
    #     },
    #     'tiro': {
    #         0: 10,
    #         3: 5
    #     },
    #     'normal':{
    #         3: 15
    #     }
    # }
    num_counter = {
        'police':{
            0: 0,
            2: 0,
            3: 0
        },
        'ambulance':{
            0: 0,
            3: 0
        },
        'reckless':{
            2: 0,
            3: 0
        },
        'bus':{
            2: 0,
            3: 0
        },
        'tiro': {
            0: 0,
            3: 0
        },
        'normal':{
            3: 0
        }
    }
    # val
    num_desired = {
        'police':{
            0: 5,
            2: 5,
            3: 2
        },
        'ambulance':{
            0: 3,
            3: 5
        },
        'reckless':{
            2: 3,
            3: 2
        },
        'bus':{
            2: 2,
            3: 3
        },
        'tiro': {
            0: 4,
            3: 1
        },
        'normal':{
            3: 5
        }
    }
    while key < args.max_episodes: 
        # print current counter and desired in a table
        logger.info("Current counter and desired in a table:")
        logger.info("Concept | Speed | Counter | Desired")
        for concept in num_desired:
            for speed in num_desired[concept]:
                logger.info("{} | {} | {} | {}".format(concept, speed, num_counter[concept][speed], num_desired[concept][speed]))
        logger.info("Trying to creat episode {} ...".format(key))
        eval_env, cached_observation = make_env(simulation_config, True)
        assert rl_config["algorithm"] == "ExpertCollector"
        model = algorithm_class(eval_env)
        o, tem_episodes = eval_env.reset(True)
        # change cached_observation
        skip = False
        cached_observation["Static Info"]["Agents"]["Car_3"]['concepts'] = tem_episodes['agents']['Car_1']['concepts']
        # check counter
        for concept in num_desired:
            if concept in tem_episodes['agents']['Car_1']['concepts']:
                for speed in num_desired[concept]:
                    if num_counter[concept][speed] == num_desired[concept][speed]:
                        logger.info("Skipping episode due to counter".format(key))
                        skip = True
                        break
                break
        using_dict = num_counter[concept]
        checking_dict = num_desired[concept]
        if skip:
            continue
        rew = 0    
        step = 0   
        d = False
        save = False
        s = time.time()
        while not d:
            step += 1
            action, _ = model.predict(o, deterministic=True)
            o, r, d, i = eval_env.step(action)
            if key in vis_id:
                cached_observation["Time_Obs"][step] = i
            action = model.predict(o)[0]
            if (action in using_dict.keys()) and not save:
                if using_dict[action] < checking_dict[action]:
                    label_action = action
                    save = True
                    label_info = {
                        'concept': concept,
                        'action': label_action,
                    }
            rew += r
        if save and i["success"]:
            logger.info("Episode {} took {} steps.".format(key, step))
            logger.info("Episode {} took {} seconds.".format(key, time.time()-s))
            tem_episodes['label_info'] = label_info
            all_episodes[key] = tem_episodes
            rew_list.append(rew)
            success.append(1)
            if key in vis_id:
                worlds.append(cached_observation)
            logger.info("Episode {} achieved a score of {}".format(key, rew))
            logger.info("Episode {} has label info: {}".format(key, label_info))
            key += 1
            using_dict[label_action] += 1
    assert len(rew_list) == len(success) == args.max_episodes
    mean_reward = np.mean(rew_list)
    logger.info("Success rate: {}".format(np.mean(success)))
    logger.info("Mean Score achieved: {}".format(mean_reward))
    for ts in range(len(worlds)):
        with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
            pkl.dump(worlds[ts], f)

    with open(os.path.join(args.log_dir, "{}_episodes.pkl".format(args.exp)), "wb") as f:
        pkl.dump(all_episodes, f)

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    # Sim mode, will use the logic-based simulator to run a simulation (no learning)
    logger.info("Collecting {} episodes from expert...".format(args.max_episodes))
    logger.info("Loading simulation config from {}.".format(args.config))
    e = time.time()
    main(args, logger)
    logger.info("Total time spent: {}".format(time.time()-e))