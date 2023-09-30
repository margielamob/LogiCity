import os
import argparse
import pickle as pkl
from utils import CityLoader, setup_logger, visualize_city
import torch
import numpy

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')

    # Add arguments for grid size, agent start and goal positions, etc.
    parser.add_argument('--map', type=str, default="config/maps/v1.0.yaml", help='YAML path to the map.')
    parser.add_argument('--agents', type=str, default="config/agents/v0.yaml", help='YAML path to the agent definition.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log")
    parser.add_argument('--exp', type=str, default="debug")
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps for the simulation.')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use.')

    return parser.parse_args()

def main(args, logger):
    logger.info("Starting city simulation with random seed {}...".format(args.seed))
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    # Create a city instance with a predefined grid
    city = CityLoader.from_yaml(args.map, args.agents)
    visualize_city(city, 1000, -1, "vis/init.png")

    # Main simulation loop
    steps = 0
    cached_observation = {0: city.city_grid}
    while steps < args.max_steps:
        logger.info("Simulating Step_{}...".format(steps))
        city.update()
        # Visualize the current state of the city (optional)
        visualize_city(city, 1000, -1, "vis/step_{}.png".format(steps))
        steps += 1
        cached_observation[steps] = city.city_grid

    with open(os.path.join(args.log_dir, "{}.pkl".format(args.exp)), "wb") as f:
        pkl.dump(cached_observation, f)

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main(args, logger)
