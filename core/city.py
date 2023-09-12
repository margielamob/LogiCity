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
    7: 'Store'
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
        for i in range(building.position[0], building.position[0] + building.size[0]):
            for j in range(building.position[1], building.position[1] + building.size[1]):
                if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:  # Check boundaries
                    self.city_grid[0][i][j] = building_code

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

    def visualize(self, resolution):
        # Define a color for each entity
        color_map = {
            -1: [100, 100, 100],
            0: [200, 200, 200],       # White for empty
            1: [0, 255, 0],           # Green for walking street
            2: [0, 0, 255],           # Red for traffic street
            3: [230, 230, 250],       # Lavender for house
            4: [255, 99, 71],         # Tomato for gas station
            5: [135, 206, 235],       # SkyBlue for office
            6: [139, 69, 19],         # SaddleBrown for garage
            7: [255, 215, 0]          # Gold for store
        }

        # Create a visual grid of the city
        visual_grid = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        scale_factor = resolution/self.grid_size[0]
        for k in range(self.layers):
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    color = color_map[self.grid[i][j]]
                    visual_grid[int(i*scale_factor):int((i+1)*scale_factor), int(j*scale_factor):int((j+1)*scale_factor)] = color

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
        cv2.imwrite('City.png', combined_img)
