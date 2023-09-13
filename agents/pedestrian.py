from .basic import Agent
import torch
import torch.nn.functional as F
from utils.find import find_nearest_building, find_building_mask

class Pedestrian(Agent):
    def __init__(self, type, size, id, world_state_martix):
        super().__init__(type, size, id, world_state_martix)
        self.start_point_list = None
        self.goal_point_list = None

    def init(self, world_state_martix, global_planner=None, local_planner=None):
        self.start = self.get_start(world_state_martix)
        self.pos = self.start
        self.goal = self.get_goal(world_state_martix, self.start)
        # specify the occupacy map
        movable_region = self.get_movable_area(world_state_martix)
        # get global traj on the occupacy map
        self.global_traj = global_planner(self.start, self.goal, movable_region)
        return super().init(global_planner, local_planner)

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
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Check for houses and offices around each cell
        building_layer = world_state_matrix[0]
        houses_offices = (building_layer == HOUSE) | (building_layer == OFFICE)
        conv_res = F.conv2d(houses_offices.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)

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
