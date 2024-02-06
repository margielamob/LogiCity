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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log_rl")
    parser.add_argument('--exp', type=str, default="3p6c2b_expert")
    parser.add_argument('--vis', action='store_true', help='Visualize the city.')
    # seed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=100)
    # RL
    parser.add_argument('--collect_only', action='store_true', help='Only collect expert data.')
    parser.add_argument('--use_gym', action='store_true', help='In gym mode, we can use RL alg. to control certain agents.')
    parser.add_argument('--config', default='config/tasks/Nav/easy/RL/bc.yaml', help='Configure file for this RL exp.')

    return parser.parse_args()

def main_collect(args, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    collection_config = config['collecting_config']
    logger.info("RL config: {}".format(collection_config))

    # Check if expert data collection is requested
    logger.info("Collecting expert demonstration data...")
    # Create an environment instance for collecting expert demonstrations
    expert_data_env, cached_observation = make_env(simulation_config, True)  # Use your existing environment setup function
    assert expert_data_env.use_expert  # Ensure the environment uses expert actions
    
    # Initialize the ExpertCollector with the environment and total timesteps
    collector = ExpertCollector(expert_data_env, **collection_config)
    _, full_world = collector.collect_data(cached_observation)
    
    # Save the collected expert demonstrations
    collector.save_data(f"{args.log_dir}/{args.exp}_expert_demonstrations.pkl")
    logger.info(f"Collected and saved expert demonstration data to {args.log_dir}/{args.exp}_expert_demonstrations.pkl")
    # Save the full world if needed
    if collection_config["return_full_world"]:
        for ts in range(len(full_world)):
            with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
                pkl.dump(full_world[ts], f)

def main(args, logger):
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Create a city instance with a predefined grid
    city, cached_observation = CityLoader.from_yaml(**simulation_config)
    visualize_city(city, 4*WORLD_SIZE, -1, "vis/init.png")
    # Main simulation loop
    steps = 0
    while steps < args.max_steps:
        logger.info("Simulating Step_{}...".format(steps))
        s = time.time()
        time_obs = city.update()
        e = time.time()
        logger.info("Time spent: {}".format(e-s))
        # Visualize the current state of the city (optional)
        if args.vis:
            visualize_city(city, 4*WORLD_SIZE, -1, "vis/step_{}.png".format(steps))
        steps += 1
        cached_observation["Time_Obs"][steps] = time_obs

    # Save the cached observation for better rendering
    with open(os.path.join(args.log_dir, "{}.pkl".format(args.exp)), "wb") as f:
        pkl.dump(cached_observation, f)

def main_gym(args, logger): 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config)
    # simulation config
    simulation_config = config["simulation"]
    logger.info("Simulation config: {}".format(simulation_config))
    # RL config
    rl_config = config['stable_baselines']
    logger.info("RL config: {}".format(rl_config))
    # Dynamic import of the features extractor class
    features_extractor_class = dynamic_import(
        rl_config["policy_kwargs"]["features_extractor_module"],
        rl_config["policy_kwargs"]["features_extractor_class"]
    )
    # Prepare policy_kwargs with the dynamically imported class
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": rl_config["policy_kwargs"]["features_extractor_kwargs"]
    }
    # Dynamic import of the RL algorithm
    algorithm_class = dynamic_import(
        "logicity.rl_agent.alg",  # Adjust the module path as needed
        rl_config["algorithm"]
    )
    # Load the entire eval_checkpoint configuration as a dictionary
    eval_checkpoint_config = config.get('eval_checkpoint', {})
    # Hyperparameters
    hyperparameters = rl_config["hyperparameters"]
    train = rl_config["train"]
    num_envs = rl_config["num_envs"]
    total_timesteps = rl_config["total_timesteps"]
    
    # data rollouts
    if train: 
        if num_envs > 1:
            logger.info("Running in RL mode with {} parallel environments.".format(num_envs))
            train_env = SubprocVecEnv([make_envs(simulation_config, i) for i in range(num_envs)])
        else:
            train_env = make_env(simulation_config)
        eval_env = make_env(simulation_config)
        train_env.reset()
        model = algorithm_class(rl_config["policy_kwargs"]["policy_network"], \
                                train_env, \
                                **hyperparameters, \
                                policy_kwargs=policy_kwargs)
        # RL training mode
        # Create the custom checkpoint and evaluation callback
        eval_checkpoint_callback = EvalCheckpointCallback(eval_env=eval_env, exp_name=args.exp, \
                                                          **eval_checkpoint_config)
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=eval_checkpoint_callback\
                    , tb_log_name=args.exp)
        # Save the model
        model.save(eval_checkpoint_config["name_prefix"])
        return
    
    # Checkpoint evaluation
    rew_list = []
    worlds = []

    for ts in range(10): 
        eval_env, cached_observation = make_env(simulation_config, True)
        if rl_config["algorithm"] == "ExpertCollector":
            model = algorithm_class(eval_env)
        else:
            model = algorithm_class.load(rl_config["checkpoint_path"], env=eval_env)
        o = eval_env.reset()
        ep_rew_list = []
        rew = 0    
        step = 0    
        collected = False
        while step<300:
            step += 1
            action, _ = model.predict(o, deterministic=True)
            o, r, d, i = eval_env.step(action)
            if not d and not collected:
                cached_observation["Time_Obs"][step] = i
            if d:
                collected = True
            action = model.predict(o)[0]
            ep_rew_list.append(r)
            rew += r
        rew_list.append(rew)
        worlds.append(cached_observation)
    mean_reward = np.mean(rew_list)
    print(mean_reward)
    for ts in range(len(worlds)):
        with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
            pkl.dump(worlds[ts], f)

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    if args.collect_only:
        logger.info("Running in data collection mode.")
        logger.info("Loading simulation config from {}.".format(args.config))
        main_collect(args, logger)
    elif args.use_gym:
        logger.info("Running in RL mode.")
        logger.info("Loading RL config from {}.".format(args.config))
        # RL mode, will use gym wrapper to learn and test an agent
        main_gym(args, logger)
    else:
        # Sim mode, will use the logic-based simulator to run a simulation (no learning)
        logger.info("Running in simulation mode.")
        logger.info("Loading simulation config from {}.".format(args.config))
        e = time.time()
        main(args, logger)
        logger.info("Total time spent: {}".format(time.time()-e))