from .building import Building
import numpy as np
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
    8: 'Pedestrian'
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
            Label+0.2 denoets next position and, Label-0.1 means walked position
        """
        self.agents.append(agent)
        agent_layer = torch.zeros((1, self.grid_size[0], self.grid_size[1]))
        agent_code = self.type2label[agent.type]
        # draw agent
        agent_layer[0][agent.start[0]:agent.start[0]+agent.size, agent.start[1]:agent.start[1]+agent.size] = agent_code
        agent_layer[0][agent.goal[0]:agent.goal[0]+agent.size, agent.goal[1]:agent.goal[1]+agent.size] = agent_code + 0.3
        for way_points in agent.global_traj[1:-1]:
            agent_layer[0][way_points[0]:way_points[0]+agent.size, way_points[1]:way_points[1]+agent.size] = agent_code + 0.1
        agent.layer_id = self.city_grid.shape[0]
        self.city_grid = torch.concat([self.city_grid, agent_layer], dim=0)
        self.layers += 1

    def visualize(self, resolution, agent_layer=None, agent_type=None):
        # Define a color for each entity
        # only visualize a static map with buildings, streets, and one layer agents
        color_map = {
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

        # Create a visual grid of the city, buildings and streets
        visual_grid = np.ones((resolution, resolution, 3), dtype=np.uint8)*200
        np_grid = self.city_grid.numpy().astype(np.int)
        scale_factor = resolution/self.grid_size[0]
        for k in range(2):
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if np_grid[k][i][j] != 0:
                        color = color_map[np_grid[k][i][j]]
                        visual_grid[int(i*scale_factor):int((i+1)*scale_factor), int(j*scale_factor):int((j+1)*scale_factor)] = color

        # draw agent's if provided layer
        if agent_layer != None:
            assert agent_type != None
            agent_layer = self.city_grid[agent_layer]
            cur_agent_pos = torch.nonzero((agent_layer == self.type2label[agent_type]).float()).tolist()
            planned_traj = torch.nonzero((agent_layer == self.type2label[agent_type]+0.1).float()).tolist()
            goal_agent_pos = torch.nonzero((agent_layer == self.type2label[agent_type]+0.3).float()).tolist()
            cv2.drawMarker(visual_grid, (int(cur_agent_pos[0][1]*scale_factor), int(cur_agent_pos[0][0]*scale_factor)), \
                (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=2, line_type=cv2.LINE_AA)
            cv2.drawMarker(visual_grid, (int(goal_agent_pos[0][1]*scale_factor), int(goal_agent_pos[0][0]*scale_factor)), \
                (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=2, line_type=cv2.LINE_AA)
            for way_point in planned_traj:
                visual_grid[int(way_point[0]*scale_factor):int((way_point[0]+1)*scale_factor), \
                    int(way_point[1]*scale_factor):int((way_point[1]+1)*scale_factor)] = [0, 255, 0]
        
        
        # Add the legend
        padding = int(10*scale_factor)
        legend_width = resolution
        legend_height = resolution
        legend_item_height = int(20*scale_factor)
        legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255  # white background

        for idx, (key, value) in enumerate(color_map.items()):
            y_offset = idx * legend_item_height + padding
            if y_offset + legend_item_height > legend_height:  # Ensure we don't render beyond the legend image
                break
            cv2.rectangle(legend_img, (padding, y_offset), (padding + legend_item_height, y_offset + legend_item_height), color_map[key], -1)
            cv2.putText(legend_img, str(self.label2type[key]), (padding + int(30*scale_factor), y_offset + legend_item_height - int(5*scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*scale_factor, \
                (0, 0, 0), int(scale_factor), lineType=cv2.LINE_AA)

        # Combine the visual grid and the legend side by side
        combined_img = np.hstack((visual_grid, legend_img))

        # Use OpenCV to display the city
        cv2.imwrite('City_agent.png', combined_img)
