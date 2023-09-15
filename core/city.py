from .building import Building
import numpy as np
import random
import cv2
import torch

LABEL_MAP = {
    -1: 'Overlap',
    0: 'Under Construction',
    1: 'Walking Street',
    2: 'Traffic Street',
    3: 'House',
    4: 'Gas Station',
    5: 'Office',
    6: 'Garage',
    7: 'Store',
    8: 'Pedestrian',
    9: 'Car'
}

class City:
    def __init__(self, grid_size=(10, 10)):
        self.grid_size = grid_size
        self.layers = 2
        # Initialize the grid with 0 placeholders
        # world_start_matrix
        self.city_grid = torch.zeros((self.layers, grid_size[0], grid_size[1]))
        self.buildings = []
        self.streets = []
        self.agents = []
        self.label2type = LABEL_MAP
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        # vis color map
        self.color_map = {
            -1: [100, 100, 100],
            0: [200, 200, 200],       # Grey for empty
            1: [152, 216, 170],           # Green for walking street
            2: [168, 161, 150],        # Red for traffic street
            3: [255, 204, 112],       # house
            4: [34, 102, 141],        # gas station
            5: [255, 250, 221],       # office
            6: [142, 205, 221],       # garage
            7: [255, 63, 164]         # store
        }

    def update(self):
        new_matrix = torch.zeros_like(self.city_grid)
        for agent in self.agents:
            # we use the current map for update, i.e., the agents don't know other's behavior
            # re-initialized agents may update city matrix as well
            local_action, new_matrix[agent.layer_id] = agent.get_next_action(self.city_grid)
            next_layer = agent.move(local_action, new_matrix[agent.layer_id])
            new_matrix[agent.layer_id] = next_layer
        # Update city grid after all the agents make decisions
        self.city_grid[2:] = new_matrix[2:]

    def add_building(self, building):
        """Add a building to the city and mark its position on the grid."""
        self.buildings.append(building)
        building_code = self.type2label[building.type]
        self.city_grid[0][building.position[0]:building.position[0] + building.size[0], \
            building.position[1]:building.position[1] + building.size[1]] = building_code

    def add_street(self, street):
        """Add a street to the city and mark its position on the grid."""
        self.streets.append(street)
        street_code = self.type2label[street.type]
        if street.orientation == 'horizontal':
            for i in range(street.position[0], street.position[0] + street.width):
                for j in range(street.position[1], street.position[1] + street.length):
                    if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:  # Check boundaries
                        if self.city_grid[1][i][j] == 0 or self.city_grid[1][i][j] == street_code:
                            self.city_grid[1][i][j] = street_code
                        else:
                            self.city_grid[1][i][j] = -1
        else:  # vertical street
            for i in range(street.position[0], street.position[0] + street.length):
                for j in range(street.position[1], street.position[1] + street.width):
                    if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:  # Check boundaries
                        if self.city_grid[1][i][j] == 0 or self.city_grid[1][i][j] == street_code:
                            self.city_grid[1][i][j] = street_code
                        else:
                            self.city_grid[1][i][j] = -1

    def add_agent(self, agent):
        """Add a agents to the city and mark its position on the grid. Label correspons
            to the current position, Label+0.1 denotes planned global path, Label+0.3 denotes goal point,
            Label+0.2 denoets next position and, Label-0.1 means walked position, Label-0.2 means start position
        """
        self.agents.append(agent)
        agent_layer = torch.zeros((1, self.grid_size[0], self.grid_size[1]))
        agent_code = self.type2label[agent.type]
        # draw agent
        agent_layer[0][agent.start[0], agent.start[1]] = agent_code
        agent_layer[0][agent.goal[0], agent.goal[1]] = agent_code + 0.3
        for way_points in agent.global_traj[1:-1]:
            if torch.all(way_points==agent.start) or torch.all(way_points==agent.goal):
                continue
            agent_layer[0][way_points[0], way_points[1]] = agent_code + 0.1
        agent.layer_id = self.city_grid.shape[0]
        self.city_grid = torch.concat([self.city_grid, agent_layer], dim=0)
        self.layers += 1
        agent_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        if "{}_{}".format(agent.type, agent.id) not in self.color_map.keys():
                self.color_map["{}_{}".format(agent.type, agent.id)] = agent_color