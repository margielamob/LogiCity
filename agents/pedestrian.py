from .basic import Agent
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.find import find_nearest_building, find_building_mask
from planners import GPlanner_mapper, LPlanner_mapper
from utils.sample import sample_start_goal, sample_determine_start_goal
from core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

class Pedestrian(Agent):
    def __init__(self, type, size, id, world_state_matrix, global_planner, debug=False):
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner = GPlanner_mapper[global_planner]
        super().__init__(type, size, id, world_state_matrix)
        # pedestrian use A*, which is just a function
        self.action_mapping = {
            0: "Left", 
            1: "Right", 
            2: "Up", 
            3: "Down", 
            4: "Stop"
            }

    def init(self, world_state_matrix, debug=False):
        WALKING_STREET = TYPE_MAP['Walking Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        if debug:
            self.start, self.goal = sample_determine_start_goal(self.type, self.id)
        else:
            self.start = torch.tensor(self.get_start(world_state_matrix))
            self.goal = torch.tensor(self.get_goal(world_state_matrix, self.start))
            self.pos = self.start.clone()
        # specify the occupacy map
        self.movable_region = (world_state_matrix[STREET_ID] == WALKING_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET)
        # get global traj on the occupacy map
        self.global_traj = self.global_planner(self.movable_region, self.start, self.goal)
        logger.info("{}_{} initialization done!".format(self.type, self.id))

    def get_start(self, world_state_matrix):
        # Define the labels for different entities
        building = [TYPE_MAP[b] for b in PEDES_GOAL_START]
        # Find cells that are walking streets and have a house or office around them
        desired_locations = sample_start_goal(world_state_matrix, TYPE_MAP['Walking Street'], building, kernel_size=PED_GOAL_START_INCLUDE_KERNEL)
        self.start_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(self.start_point_list), (1,)).item()
        
        # Fetch the corresponding location
        start_point = self.start_point_list[random_index]

        # Return the indices of the desired locations
        return start_point

    def get_goal(self, world_state_matrix, start_point):
        # Define the labels for different entities
        building = [TYPE_MAP[b] for b in PEDES_GOAL_START]
        # Find cells that are walking streets and have a house or office around them
        self.desired_locations = sample_start_goal(world_state_matrix, TYPE_MAP['Walking Street'], building, kernel_size=PED_GOAL_START_INCLUDE_KERNEL)
        desired_locations = self.desired_locations.detach().clone()

        # Determine the nearest building to the start point
        nearest_building = find_nearest_building(world_state_matrix, start_point)
        start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

        # Get the mask for the building containing the nearest_building position
        # building_mask = find_building_mask(world_state_matrix, nearest_building)
        building_mask = world_state_matrix[BLOCK_ID] == start_block
        
        # Create a mask to exclude areas around the building. We'll dilate the building mask.
        exclusion_radius = PED_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 5 grids around the block
        expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
        
        desired_locations[expanded_mask[0, 0]] = False

        # Return the indices of the desired locations
        goal_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(goal_point_list), (1,)).item()
        
        # Fetch the corresponding location
        goal_point = goal_point_list[random_index]

        # Return the indices of the desired locations
        return goal_point

    def get_action(self, local_action_dist):
        if not torch.any(local_action_dist):
            return self.get_global_action()
        else:
            # sample from the local planner
            normalized_action_dist = local_action_dist / local_action_dist.sum()
            dist = Categorical(normalized_action_dist)
            # Sample an action index from the distribution
            action_index = dist.sample()
            # Get the actual action from the action space using the sampled index
            return self.action_space[action_index]

    def get_next_action(self, world_state_matrix, local_action_dist):
        # for now, just reckless take the global traj
        # reached goal
        if not self.reach_goal:
            if torch.all(self.pos == self.goal):
                # self.reach_goal = True
                logger.info("{}_{} reached goal! Will change goal in the next step!".format(self.type, self.id))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            else:
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
        else:
            logger.info("Generating new goal and gloabl plans for {}_{}...".format(self.type, self.id))
            self.start = self.goal.clone()
            desired_locations = self.desired_locations.detach().clone()

            # Determine the nearest building to the start point
            nearest_building = find_nearest_building(world_state_matrix, self.start)
            start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

            # Get the mask for the building containing the nearest_building position
            building_mask = world_state_matrix[BLOCK_ID] == start_block
            
            # Create a mask to exclude areas around the building. We'll dilate the building mask.
            exclusion_radius = PED_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 5 grids around the block
            expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
            
            desired_locations[expanded_mask[0, 0]] = False

            # Return the indices of the desired locations
            goal_point_list = torch.nonzero(desired_locations).tolist()
            random_index = torch.randint(0, len(goal_point_list), (1,)).item()
            
            # Fetch the corresponding location
            self.goal = torch.tensor(goal_point_list[random_index])
            self.global_traj = self.global_planner(self.movable_region, self.start, self.goal)
            logger.info("Generating new goal and gloabl plans for {}_{} done!".format(self.type, self.id))
            self.reach_goal = False
            # delete past traj
            world_state_matrix[self.layer_id] *= 0
            world_state_matrix[self.layer_id][self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
            for way_points in self.global_traj[1:-1]:
                world_state_matrix[self.layer_id][way_points[0], way_points[1]] \
                    = TYPE_MAP[self.type] + AGENT_GLOBAL_PATH_PLUS
            world_state_matrix[self.layer_id][self.start[0], self.start[1]] = TYPE_MAP[self.type]

            return self.get_action(local_action_dist), world_state_matrix[self.layer_id]

    def get_global_action(self):
        next_pos = self.global_traj[0]
        self.global_traj.pop(0)
        del_pos = next_pos - self.pos
        dis = torch.tensor(del_pos).float().norm().item()
        assert dis <= 1
        if del_pos[1] < 0:
            # left
            return self.action_space[0]
        elif del_pos[1] > 0:
            # right
            return self.action_space[1]
        elif del_pos[0] < 0:
            # up
            return self.action_space[2]
        elif del_pos[0] > 0:
            # down
            return self.action_space[3]
        else:
            return self.action_space[-1]

    def move(self, action, ped_layer):
        curr_pos = torch.nonzero((ped_layer==TYPE_MAP[self.type]).float())[0]
        assert torch.all(self.pos == curr_pos)
        next_pos = self.pos.clone()
        # becomes walked grid
        ped_layer[self.pos[0], self.pos[1]] += AGENT_WALKED_PATH_PLUS
        if action == self.action_space[0]:
            next_pos[1] -= 1
        elif action == self.action_space[1]:
            next_pos[1] += 1
        elif action == self.action_space[2]:
            next_pos[0] -= 1
        elif action == self.action_space[3]:
            next_pos[0] += 1
        else:
            next_pos = self.pos.clone()
        self.pos = next_pos.clone()
        # Update Agent Map
        ped_layer[self.start[0], self.start[1]] = TYPE_MAP[self.type] + AGENT_START_PLUS
        ped_layer[self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
        ped_layer[self.pos[0], self.pos[1]] = TYPE_MAP[self.type]
        return ped_layer