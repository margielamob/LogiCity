from .basic import Agent
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.find import find_nearest_building, find_building_mask
from utils.sample import sample_start_goal, sample_start_goal_vh, sample_determine_start_goal
from planners import GPlanner_mapper
from core.config import *
import logging
# import cv2
# import numpy as np
# # vis quick tool
# vis = np.zeros((250, 250, 3))
# vis[self.movable_region] = [255, 0, 0]
# vis[self.goal[0], self.goal[1]] = [0, 255, 255]
# vis[self.start[0], self.start[1]] = [0, 255, 0]
# vis[self.midline_matrix] = [0, 0, 255]
# cv2.imwrite("test.png", vis)

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

class Car(Agent):
    def __init__(self, type, size, id, world_state_matrix, global_planner, concepts=None, debug=False):
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner_type = global_planner
        super().__init__(type, size, id, world_state_matrix, concepts=concepts, debug=debug)
        # Actions: ["left_1", "right_1", "up_1", "down_1", "left_2", "right_2", "up_2", "down_2", "stop"]
        self.action_space = torch.tensor(range(9))
        self.action_to_move = {
            self.action_space[0].item(): torch.tensor((0, -1)),
            self.action_space[1].item(): torch.tensor((0, 1)),
            self.action_space[2].item(): torch.tensor((-1, 0)),
            self.action_space[3].item(): torch.tensor((1, 0)),
            self.action_space[4].item(): torch.tensor((0, -2)),
            self.action_space[5].item(): torch.tensor((0, 2)),
            self.action_space[6].item(): torch.tensor((-2, 0)),
            self.action_space[7].item(): torch.tensor((2, 0))
        }
        self.move_to_action = {
            tuple(self.action_to_move[k].tolist()): k for k in self.action_to_move
        }
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "Left_1", 
            1: "Right_1", 
            2: "Up_1", 
            3: "Down_1", 
            4: "Left_2", 
            5: "Right_2", 
            6: "Up_2", 
            7: "Down_2",
            8: "Stop"
            }

    def init(self, world_state_matrix, debug=False):
        Traffic_STREET = TYPE_MAP['Traffic Street']
        CROSSING_STREET = TYPE_MAP['Overlap']
        if debug:
            self.start, self.goal = sample_determine_start_goal(self.type, self.id)
            self.pos = self.start.clone()
        else:
            self.start = torch.tensor(self.get_start(world_state_matrix))
            self.pos = self.start.clone()
            self.goal = torch.tensor(self.get_goal(world_state_matrix, self.start))
        # specify the occupacy map
        self.movable_region = (world_state_matrix[STREET_ID] == Traffic_STREET) | (world_state_matrix[STREET_ID] == CROSSING_STREET)
        self.midline_matrix = (world_state_matrix[STREET_ID] == Traffic_STREET+MID_LINE_CODE_PLUS)
        self.global_planner = GPlanner_mapper[self.global_planner_type](self.movable_region, self.midline_matrix, CAR_STREET_OFFSET)
        # get global traj on the occupacy map
        self.global_traj = self.global_planner.plan(self.start, self.goal)
        logger.info("{}_{} initialization done!".format(self.type, self.id))

    def get_start(self, world_state_matrix):
        # Define the labels for different entities
        building = [TYPE_MAP[b] for b in CAR_GOAL_START]
        # Find cells that are walking streets and have a house or office around them
        desired_locations = sample_start_goal_vh(world_state_matrix, TYPE_MAP['Traffic Street'], building, kernel_size=CAR_GOAL_START_INCLUDE_KERNEL)
        self.start_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(self.start_point_list), (1,)).item()
        
        # Fetch the corresponding location
        start_point = self.start_point_list[random_index]

        # Return the indices of the desired locations
        return start_point

    def get_goal(self, world_state_matrix, start_point):
        # Define the labels for different entities
        # Define the labels for different entities
        building = [TYPE_MAP[b] for b in CAR_GOAL_START]

        # Find cells that are walking streets and have a house, office, or store around them
        self.desired_locations = desired_locations = sample_start_goal_vh(world_state_matrix, TYPE_MAP['Traffic Street'], building, kernel_size=CAR_GOAL_START_INCLUDE_KERNEL)
        desired_locations = self.desired_locations.detach().clone()

        # Determine the nearest building to the start point
        nearest_building = find_nearest_building(world_state_matrix, start_point)
        start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

        # Get the mask for the building containing the nearest_building position
        # building_mask = find_building_mask(world_state_matrix, nearest_building)
        building_mask = world_state_matrix[BLOCK_ID] == start_block
        
        # Create a mask to exclude areas around the building. We'll dilate the building mask.
        exclusion_radius = CAR_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 3 grids around the building
        expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
        
        desired_locations[expanded_mask[0, 0]] = False

        # Return the indices of the desired locations
        goal_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(goal_point_list), (1,)).item()
        
        # Fetch the corresponding location
        goal_point = goal_point_list[random_index]

        # Return the indices of the desired locations
        return goal_point

    def get_next_action(self, world_state_matrix, local_action_dist):
        # for now, just reckless take the global traj
        # reached goal
        if not self.reach_goal:
            if torch.all(self.pos == self.goal):
                if not self.debug:
                    self.reach_goal = True
                    self.reach_goal_buffer += REACH_GOAL_WAITING
                    logger.info("{}_{} reached goal! Will change goal in the next {} step!".format(self.type, self.id, self.reach_goal_buffer))
                else:
                    logger.info("{}_{} reached goal! In Debug, it will stop".format(self.type, self.id))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            else:
                return self.get_action(local_action_dist), world_state_matrix[self.layer_id]
        else:
            if self.reach_goal_buffer > 0:
                self.reach_goal_buffer -= 1
                logger.info("{}_{} reached goal! Will change goal in the next {} step!".format(self.type, self.id, self.reach_goal_buffer))
                return self.action_space[-1], world_state_matrix[self.layer_id]
            logger.info("Generating new goal and gloabl plans for {}_{}...".format(self.type, self.id))
            self.start = self.goal.clone()
            desired_locations = self.desired_locations.detach().clone()

            # Determine the nearest building to the start point
            nearest_building = find_nearest_building(world_state_matrix, self.start)
            start_block = world_state_matrix[BLOCK_ID][nearest_building[0], nearest_building[1]]

            # Get the mask for the building containing the nearest_building position
            building_mask = world_state_matrix[BLOCK_ID] == start_block
            
            # Create a mask to exclude areas around the building. We'll dilate the building mask.
            exclusion_radius = CAR_GOAL_START_EXCLUDE_KERNEL  # Excludes surrounding 3 grids around the building
            expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
            
            desired_locations[expanded_mask[0, 0]] = False

            # Return the indices of the desired locations
            goal_point_list = torch.nonzero(desired_locations).tolist()
            random_index = torch.randint(0, len(goal_point_list), (1,)).item()
            
            # Fetch the corresponding location
            self.goal = torch.tensor(goal_point_list[random_index])
            self.global_traj = self.global_planner.plan(self.start, self.goal)
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