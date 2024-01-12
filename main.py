import os
import sys
import argparse
import pickle as pkl
from utils.load import CityLoader
from utils.logger import setup_logger
from utils.vis import visualize_city
from core.config import *
import torch
import torch.nn as nn
import time
import numpy as np
from utils.gym_wrapper import GymCityWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from rl_agent.PPO_img import PPO, CustomCNN, policy_kwargs
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import trange


def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')

    # Add arguments for grid size, agent start and goal positions, etc.
    parser.add_argument('--map', type=str, default="config/maps/v1.1.yaml", help='YAML path to the map.')
    parser.add_argument('--agents', type=str, default="config/agents/debug.yaml", help='YAML path to the agent definition.')
    parser.add_argument('--rule_type', type=str, default="Z3_Local", help='We support ["LNN", "Z3_Global", "Z3_Local"].')
    parser.add_argument('--rules', type=str, default="config/rules/Z3/easy/easy_rule_local.yaml", help='YAML path to the rule definition.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log")
    parser.add_argument('--exp', type=str, default="rl_debug")
    parser.add_argument('--vis', type=bool, default=False, help='Visualize the city.')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum number of steps for the simulation.')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use.')
    parser.add_argument('--use_gym', type=bool, default=True, help='In gym mode, we can use RL alg. to control certain agents.')
    parser.add_argument('--debug', type=bool, default=False, help='In debug mode, the agents are in defined positions.')
    parser.add_argument('--eval', type=bool, default=False, help='In eval mode, we will load the PPO checkpoints.')

    return parser.parse_args()

def main(args, logger):
    logger.info("Starting city simulation with random seed {}... Debug mode: {}".format(args.seed, args.debug))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Create a city instance with a predefined grid
    city, cached_observation = CityLoader.from_yaml(args.map, args.agents, args.rules, args.rule_type, args.debug)
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



def main_gym(args, logger, train=True): 
    def make_envs(): 
        city, cached_observation = CityLoader.from_yaml(args.map, args.agents, args.rules, args.rule_type, True, args.debug)
        env = GymCityWrapper(city)
        return env
    logger.info("Starting city simulation with random seed {}... Debug mode: {}".format(args.seed, args.debug))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = SubprocVecEnv([make_envs for i in range(4)])
    
    # data rollouts
    if train: 
        env = make_envs()
        env.reset()
        # RL training mode
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/', name_prefix='ppo_model')

        # model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
        model = PPO("MlpPolicy", env, verbose=1)
        # Train the model
        model.learn(total_timesteps=1000_0000, callback=checkpoint_callback)
        # Save the model
        model.save("ppo_custommlp")
    
    else: 
        print(env.reset())
        t0 = time.time()
        for steps in range(1000):
            action = np.array([[0, 0, 0, 1, 0]]*4)
            o, r, d, i = env.step(action)
            print(o[0].shape)
            print(r, d)
            if d.any(): 
                input("done")
            cached_observation["Time_Obs"][steps] = o
        with open(os.path.join(args.log_dir, "{}.pkl".format(args.exp)), "wb") as f:
            pkl.dump(cached_observation, f)
    
    # Checkpoint evaluation
    
    rew_list = []
    for ts in range(1, 21): 
        city, cached_observation = CityLoader_Gym.from_yaml(args.map, args.agents, args.rules, args.rule_type, args.debug)
        env = GymCityWrapper(city)
        model = PPO.load("checkpoints/ppo_model_{}_steps".format(20*40000), env=env)
        o = env.reset()
        action = model.predict(o)[0]
        sys.stdout = open(os.devnull, 'w')
        ep_rew_list = []
        rew = 0        
        ep_rew = 0
        for steps in trange(500):
            o, r, d, i = env.step(action)
            action = model.predict(o)[0]
            ep_rew_list.append(r)
            rew += r
            cached_observation["Time_Obs"][steps] = i
            if d:
                print(ep_rew_list)
                np.save("log/rew_{}_{}.npy".format(args.exp, ts), np.array(ep_rew_list))
                # np.save('rew.npy', np.array(ep_rew_list))
                break
                # o = env.reset()
                # action = model.predict(o)[0]
        rew_list.append(rew)
        sys.stdout = sys.__stdout__   
        with open(os.path.join(args.log_dir, "{}_{}.pkl".format(args.exp, ts)), "wb") as f:
            pkl.dump(cached_observation, f)
        print(rew_list)
        # input("rew: {}".format(ts*40000))
        
    print(rew_list)
    # input("done")
if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    if args.use_gym:
        # RL mode, will use gym wrapper to learn and test an agent
        main_gym(args, logger, train=args.eval)
    else:
        # Sim mode, will use the logic-based simulator to run a simulation (no learning)
        main(args, logger)