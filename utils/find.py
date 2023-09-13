import torch
import torch.nn.functional as F

def find_nearest_building(world_state_matrix, start_point):
    start_point = torch.tensor(start_point)
    buildings_layer = world_state_matrix[0]
    
    building_positions = torch.nonzero(buildings_layer)
    distances = torch.abs(building_positions - start_point).sum(dim=1)
    
    min_index = distances.argmin().item()
    nearest_building = building_positions[min_index]
    
    return nearest_building

def find_building_mask(city_grid, start_point):
    # The value at the starting point, i.e., the building type
    start_point = start_point.numpy().tolist()
    city_grid = city_grid[0]
    building_type = city_grid[start_point[0], start_point[1]]
    
    # Expand boundaries iteratively to find the extent of the building
    top, bottom, left, right = start_point[0], start_point[0], start_point[1], start_point[1]

    while top > 0 and city_grid[top - 1, start_point[1]] == building_type:
        top -= 1

    while bottom < city_grid.shape[0] - 1 and city_grid[bottom + 1, start_point[1]] == building_type:
        bottom += 1

    while left > 0 and city_grid[start_point[0], left - 1] == building_type:
        left -= 1

    while right < city_grid.shape[1] - 1 and city_grid[start_point[0], right + 1] == building_type:
        right += 1

    # Create the mask
    mask = torch.zeros_like(city_grid, dtype=torch.bool)
    mask[top:bottom + 1, left:right + 1] = True

    return mask
