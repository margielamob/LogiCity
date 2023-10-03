import os
import argparse
import pickle as pkl
from utils.load import CityLoader
from utils.logger import setup_logger
from utils.vis import visualize_city
from core.config import *
import torch
import numpy

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')

    # Add arguments for grid size, agent start and goal positions, etc.
    parser.add_argument('--map', type=str, default="TEST.yaml", help='YAML path to the map.')
    parser.add_argument('--agents', type=str, default="config/agents/v0.yaml", help='YAML path to the agent definition.')
    parser.add_argument('--rule_type', type=str, default="LNN", help='We support ["LNN"].')
    parser.add_argument('--rules', type=str, default="config/rules/LNN/stop_v0.yaml", help='YAML path to the rule definition.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log")
    parser.add_argument('--exp', type=str, default="debug")
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps for the simulation.')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use.')
    parser.add_argument('--debug', type=bool, default=False, help='In debug mode, the agents are in defined positions.')

    return parser.parse_args()

def main(args, logger):
    logger.info("Starting city simulation with random seed {}... Debug mode: {}".format(args.seed, args.debug))
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    # Create a city instance with a predefined grid
    city = CityLoader.from_yaml(args.map, args.agents, args.rules, args.rule_type, args.debug)
    visualize_city(city, 4*WORLD_SIZE, -1, "vis/init.png")

    # Main simulation loop
    steps = 0
    cached_observation = {0: city.city_grid}
    while steps < args.max_steps:
        logger.info("Simulating Step_{}...".format(steps))
        city.update()
        # Visualize the current state of the city (optional)
        visualize_city(city, 4*WORLD_SIZE, -1, "vis/step_{}.png".format(steps))
        steps += 1
        cached_observation[steps] = city.city_grid

    with open(os.path.join(args.log_dir, "{}.pkl".format(args.exp)), "wb") as f:
        pkl.dump(cached_observation, f)

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main(args, logger)
