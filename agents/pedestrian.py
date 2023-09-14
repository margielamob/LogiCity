from .basic import Agent
import torch
import torch.nn.functional as F
from utils.find import find_nearest_building, find_building_mask
import logging

logger = logging.getLogger(__name__)

class Pedestrian(Agent):
    def __init__(self, type, size, id, world_state_matrix, global_planner):
        self.start_point_list = None
        self.goal_point_list = None
        self.global_planner = global_planner
        super().__init__(type, size, id, world_state_matrix)

    def init(self, world_state_matrix):
        WALKING_STREET = 1
        CROSSING_STREET = -1
        self.start = torch.tensor(self.get_start(world_state_matrix))
        self.pos = self.start.clone()
        self.goal = torch.tensor(self.get_goal(world_state_matrix, self.start))
        # specify the occupacy map
        movable_region = (world_state_matrix[1] == WALKING_STREET) | (world_state_matrix[1] == CROSSING_STREET)
        # get global traj on the occupacy map
        self.global_traj = self.global_planner(movable_region, self.start, self.goal)
        logger.info("{}_{} initialization done!".format(self.type, self.id))

    def get_start(self, world_state_matrix):
        # Define the labels for different entities
        WALKING_STREET = 1
        HOUSE = 3
        OFFICE = 5

        # Slice the building and street layer
        street_layer = world_state_matrix[1]

        # Find all walking street cells
        walking_streets = (street_layer == WALKING_STREET)

        # Define a kernel that captures cells around a central cell.
        # This kernel will look for a house or office around the central cell.
        kernel = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Check for houses and offices around each cell
        building_layer = world_state_matrix[0]
        houses_offices = (building_layer == HOUSE) | (building_layer == OFFICE)
        conv_res = F.conv2d(houses_offices.float().unsqueeze(0).unsqueeze(0), kernel, padding=2)

        # Find cells that are walking streets and have a house or office around them
        desired_locations = (walking_streets & (conv_res.squeeze() > 0))
        self.start_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(self.start_point_list), (1,)).item()
        
        # Fetch the corresponding location
        start_point = self.start_point_list[random_index]

        # Return the indices of the desired locations
        return start_point

    def get_goal(self, world_state_matrix, start_point):
        # Define the labels for different entities
        WALKING_STREET = 1
        HOUSE = 3
        OFFICE = 5
        STORE = 7

        # Slice the building and street layer
        street_layer = world_state_matrix[1]

        # Find all walking street cells
        walking_streets = (street_layer == WALKING_STREET)

        # Define a kernel that captures cells around a central cell.
        kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Check for houses, offices, and stores around each cell
        building_layer = world_state_matrix[0]
        desired_cells = (building_layer == HOUSE) | (building_layer == OFFICE) | (building_layer == STORE)
        conv_res = F.conv2d(desired_cells.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)

        # Find cells that are walking streets and have a house, office, or store around them
        desired_locations = (walking_streets & (conv_res.squeeze() > 0))

        # Determine the nearest building to the start point
        nearest_building = find_nearest_building(world_state_matrix, start_point)

        # Get the mask for the building containing the nearest_building position
        building_mask = find_building_mask(world_state_matrix, nearest_building)
        
        # Create a mask to exclude areas around the building. We'll dilate the building mask.
        exclusion_radius = 3  # Excludes surrounding 3 grids around the building
        expanded_mask = F.max_pool2d(building_mask[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2) > 0
        
        desired_locations[expanded_mask[0, 0]] = False

        # Return the indices of the desired locations
        self.goal_point_list = torch.nonzero(desired_locations).tolist()
        random_index = torch.randint(0, len(self.goal_point_list), (1,)).item()
        
        # Fetch the corresponding location
        goal_point = self.goal_point_list[random_index]

        # Return the indices of the desired locations
        return goal_point

    def get_next_action(self, world_state_matrix):
        # for now, just reckless take the global traj
        # reached goal
        if torch.all(self.pos == self.goal):
            self.reach_goal = True
            logger.info("{}_{} reached goal!".format(self.type, self.id))
            return self.action_space[-1]
        else:
            # action = local_planner(world_state_matrix, self.layer_id)
            return self.get_global_action()

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
            # up
            return self.action_space[3]
        else:
            return self.action_space[-1]

    def move(self, action, ped_layer, curr_label):
        curr_pos = torch.nonzero((ped_layer==curr_label).float())[0]
        assert torch.all(self.pos == curr_pos)
        next_pos = self.pos.clone()
        # becomes walked grid
        ped_layer[self.pos[0], self.pos[1]] -= 0.1
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
        ped_layer[self.start[0], self.start[1]] = curr_label - 0.2
        ped_layer[self.pos[0], self.pos[1]] = curr_label
        return ped_layer