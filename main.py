import argparse
import yaml
from core import City, Building
from utils import CityLoader, setup_logger
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
    city.visualize()
    
    # Create an agent instance with its goals and planners
    # global_planner = GlobalPlanner(city, tuple(args.start_position), tuple(args.goal_position))
    # local_planner = LocalPlanner(city)
    # agent = Agent(tuple(args.start_position), global_planner, local_planner)
    
    # Add the agent to the city
    # city.add_agent(agent)
    
    # Main simulation loop
    steps = 0
    # while not agent.at_goal() and steps < args.max_steps:
    #     # Get agent's next global plan
    #     global_plan = agent.global_planner.plan()

    #     # Let the agent decide its next move using local planner
    #     next_move = agent.local_planner.decide_move(global_plan)
    #     agent.move(next_move)

    #     # Visualize the current state of the city (optional)
    #     visualize_city(city)

    #     steps += 1

    # if agent.at_goal():
    #     print("Agent reached its goal!")
    # else:
    #     print(f"Agent couldn't reach its goal in {args.max_steps} steps.")

if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main(args, logger)
