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
    parser.add_argument('--exp', type=str, default="expert_100episode")
    parser.add_argument('--vis', action='store_true', help='Visualize the city.')
    # seed
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-episodes', type=int, default=100)
    # RL
    parser.add_argument('--config', default='config/tasks/Nav/easy/experts/expert_episode.yaml', help='Configure file for this RL exp.')

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

    while key < args.max_episodes: 
        logger.info("Trying to creat episode {} ...".format(key))
        eval_env, cached_observation = make_env(simulation_config, True)
        assert rl_config["algorithm"] == "ExpertCollector"
        model = algorithm_class(eval_env)
        o, tem_episodes = eval_env.reset(True)
        rew = 0    
        step = 0   
        d = False
        save = False
        while not d:
            step += 1
            action, _ = model.predict(o, deterministic=True)
            o, r, d, i = eval_env.step(action)
            if key in vis_id:
                cached_observation["Time_Obs"][step] = i
            action = model.predict(o)[0]
            if action[-1]:
                save = True
            rew += r
        if save:
            all_episodes[key] = tem_episodes
            rew_list.append(rew)
            success.append(1)
            if key in vis_id:
                worlds.append(cached_observation)
            logger.info("Episode {} achieved a score of {}".format(key, rew))
            key += 1
    assert len(rew_list) == len(worlds) == len(success) == args.max_episodes
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