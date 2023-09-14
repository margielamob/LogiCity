import argparse
import yaml
from core import City, Building
from utils import CityLoader, setup_logger, visualize_city
# from core.agent import Agent
# from planners.global_planner import GlobalPlanner
# from planners.local_planner import LocalPlanner
# from utils.visualization import visualize_city

def parse_arguments():
    parser = argparse.ArgumentParser(description='Logic-based city simulation.')

    # Add arguments for grid size, agent start and goal positions, etc.
    parser.add_argument('--map', type=str, default="config/maps/v0.yaml", help='YAML path to the map.')
    # logger
    parser.add_argument('--log_dir', type=str, default="./log")
    parser.add_argument('--exp', type=str, default="debug")
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps for the simulation.')

    return parser.parse_args()

def main(args, logger):
    logger.info("Starting city simulation...")
    # Create a city instance with a predefined grid
    city = CityLoader.from_yaml(args.map)
    visualize_city(city, 1000, 2, "Pedestrian", "vis/init.png")

    # Main simulation loop
    steps = 0
    cached_observation = {0: city.city_grid}
    while 1:
        city.update()
    #     # Visualize the current state of the city (optional)
        logger.info("Simulating Step_{}...".format(steps))
        # visualize_city(city, 1000, 2, "Pedestrian", "vis/step_{}.png".format(steps))
        steps += 1
        cached_observation[steps] = city.city_grid

    # if agent.at_goal():
    #     print("Agent reached its goal!")
    # else:
    #     print(f"Agent couldn't reach its goal in {args.max_steps} steps.")

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main(args, logger)
