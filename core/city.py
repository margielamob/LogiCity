from .building import Building
from planners import LPlanner_mapper
import numpy as np
import random
from skimage.draw import line, polygon
from scipy.ndimage import label
import torch
import torch.nn.functional as F
from core.config import *
from utils.vis import visualize_intersections

class City:
    def __init__(self, grid_size, local_planner, rule_file=None):
        self.grid_size = grid_size
        self.layers = BASIC_LAYER
        # 0 for blocks
        # 1 for buildings
        # 2 for streets
        # Initialize the grid with 0 placeholders
        # world_start_matrix
        self.city_grid = torch.zeros((self.layers, grid_size[0], grid_size[1]))
        self.buildings = []
        self.streets = []
        self.agents = []
        self.label2type = LABEL_MAP
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        # city rule defines local decision of all the agents
        self.local_planner = LPlanner_mapper[local_planner](rule_file)
        self.logic_grounds = {}
        # vis color map
        self.color_map = COLOR_MAP

    def update(self):
        current_obs = {}
        # state at time t
        current_obs["World"] = self.city_grid
        current_obs["Agent_actions"] = {}

        new_matrix = torch.zeros_like(self.city_grid)
        # first do local planning based on city rules
        agent_action_dist = self.local_planner.plan(self.city_grid, self.intersection_matrix, self.agents)
        pred_grounds = self.local_planner.get_current_lnn_state(self.logic_grounds)
        current_obs["LNN_state"] = pred_grounds
        # Then do global action taking acording to the local planning results
        for agent in self.agents:
            # re-initialized agents may update city matrix as well
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            empty_action = agent.action_dist.clone()
            # local reasoning-based action distribution
            local_action_dist = agent_action_dist[agent_name]
            # global trajectory-based action or sampling from local action distribution
            local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid, local_action_dist)
            # save the current action in the action
            empty_action[local_action] = 1.0
            current_obs["Agent_actions"][agent_name] = empty_action
            if agent.reach_goal:
                continue
            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[BASIC_LAYER:] = new_matrix[BASIC_LAYER:]
        return current_obs

    def add_building(self, building):
        """Add a building to the city and mark its position on the grid."""
        self.buildings.append(building)
        building_code = self.type2label[building.type]
        self.city_grid[0][building.position[0]:building.position[0] + building.size[0], \
            building.position[1]:building.position[1] + building.size[1]] = building.block
        self.city_grid[1][building.position[0]:building.position[0] + building.size[0], \
            building.position[1]:building.position[1] + building.size[1]] = building_code

    def add_street(self, street):
        """Add a street to the city and mark its position on the grid."""
        self.streets.append(street)
        street_code = self.type2label[street.type]
        if street.orientation == 'horizontal':
            for i in range(street.position[0], street.position[0] + street.width):
                for j in range(street.position[1], street.position[1] + street.length):
                    if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:  # Check boundaries
                        if self.city_grid[2][i][j] == 0 or self.city_grid[2][i][j] == street_code:
                            self.city_grid[2][i][j] = street_code
                        else:
                            self.city_grid[2][i][j] = INTERSECTION_CODE
        else:  # vertical street
            for i in range(street.position[0], street.position[0] + street.length):
                for j in range(street.position[1], street.position[1] + street.width):
                    if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:  # Check boundaries
                        if self.city_grid[2][i][j] == 0 or self.city_grid[2][i][j] == street_code:
                            self.city_grid[2][i][j] = street_code
                        else:
                            self.city_grid[2][i][j] = INTERSECTION_CODE

    def add_mid(self):
        """Add a mid lanes to traffic to the city and mark its position on the grid."""
        assert len(self.buildings) > 0
        street_code = self.type2label['Traffic Street'] + MID_LINE_CODE_PLUS
        for i in range(1, NUM_OF_BLOCKS+1):
            current_block = self.city_grid[BLOCK_ID] == i
            pixels = torch.nonzero(current_block.float())
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            left = torch.min(cols).item()
            right = torch.max(cols).item()
            top = torch.min(rows).item()
            bottom = torch.max(rows).item()
            # top
            self.city_grid[STREET_ID][left:(right+1), top-TRAFFIC_STREET_WID] = street_code
            self.city_grid[STREET_ID][left:(right+1), bottom+TRAFFIC_STREET_WID] = street_code
            self.city_grid[STREET_ID][left-TRAFFIC_STREET_WID, top:(bottom+1)] = street_code
            self.city_grid[STREET_ID][right+TRAFFIC_STREET_WID, top:(bottom+1)] = street_code
    
    def add_intersections(self):
        # Extract the 0-th layer of the world matrix
        world_layer = self.city_grid[BLOCK_ID, :, :]
        
        # Extract the unique block IDs from the 0-th layer
        unique_blocks = set(world_layer.flatten().tolist())
        unique_blocks.remove(0)  # Assuming 0 is the ID for non-block pixels
        
        # Find the corners of the blocks
        corners = {}
        for block_id in unique_blocks:
            block_positions = (world_layer == block_id).nonzero()
            xmin, xmax = min(block_positions[:, 1])-1, max(block_positions[:, 1])+1
            ymin, ymax = min(block_positions[:, 0])-1, max(block_positions[:, 0])+1
            corners[block_id] = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

        intersection_matrix = np.zeros((2, world_layer.shape[0], world_layer.shape[1]), dtype=bool)
        intersection_line_len = TRAFFIC_STREET_WID + 2*WALKING_STREET_WID - 1
        for block_id, block_corners in corners.items():
            for other_block_id, other_block_corners in corners.items():
                if block_id != other_block_id:
                    for corner in block_corners:
                        for other_corner in other_block_corners:
                            if np.linalg.norm(np.array(corner) - np.array(other_corner)) == intersection_line_len:
                                rr, cc = line(corner[0], corner[1], other_corner[0], other_corner[1])
                                # first layer is for check "At intersection, they are lines"
                                intersection_matrix[0, rr, cc] = True
                            elif intersection_line_len < np.linalg.norm(np.array(corner) - np.array(other_corner)) < 2 * intersection_line_len:
                                # second layer is for check "In intersection, they are blocks"
                                rr, cc = line(corner[0], corner[1], other_corner[0], other_corner[1])
                                # Gather the vertices of the polygon
                                assert rr.max()!=rr.min() and cc.max()!=cc.min()
                                intersection_matrix[1, rr.min():rr.max(), cc.min():cc.max()] = True
                                

        # Label connected regions in the intersection matrix
        labeled_matrix_line, num_line = label(intersection_matrix[0])
        assert num_line == NUM_INTERSECTIONS_LINES, "Number of intersection lines is not {}".format(NUM_INTERSECTIONS_LINES)
        labeled_matrix_block, num_block = label(intersection_matrix[1])
        assert num_block == NUM_INTERSECTIONS_BLOCKS, "Number of intersection blocks is not {}".format(NUM_INTERSECTIONS_BLOCKS)
        self.intersection_matrix = torch.tensor([labeled_matrix_line, labeled_matrix_block])
        # exclusion_radius = 2*AT_INTERSECTION_E+1
        # self.intersection_matrix = F.max_pool2d(intersection_matrix[None, None].float(), exclusion_radius, stride=1, padding=(exclusion_radius - 1) // 2)
        # self.intersection_matrix = self.intersection_matrix.squeeze(0).squeeze(0)

    def add_agent(self, agent):
        """Add a agents to the city and mark its position on the grid. Label correspons
            to the current position, Label+0.1 denotes planned global path, Label+0.3 denotes goal point,
            Label+0.2 denoets next position and, Label-0.1 means walked position, Label-0.2 means start position,
            see config.py for details.
        """
        self.agents.append(agent)
        agent_layer = torch.zeros((1, self.grid_size[0], self.grid_size[1]))
        agent_code = self.type2label[agent.type]
        # draw agent
        agent_layer[0][agent.start[0], agent.start[1]] = agent_code
        agent_layer[0][agent.goal[0], agent.goal[1]] = agent_code + AGENT_GOAL_PLUS
        for way_points in agent.global_traj[1:-1]:
            if torch.all(way_points==agent.start) or torch.all(way_points==agent.goal):
                continue
            agent_layer[0][way_points[0], way_points[1]] = agent_code + AGENT_GLOBAL_PATH_PLUS
        agent.layer_id = self.city_grid.shape[0]
        self.city_grid = torch.concat([self.city_grid, agent_layer], dim=0)
        self.layers += 1
        agent_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        if "{}_{}".format(agent.type, agent.id) not in self.color_map.keys():
                self.color_map["{}_{}".format(agent.type, agent.id)] = agent_color