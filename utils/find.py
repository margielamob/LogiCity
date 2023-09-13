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

def find_building_mask(world_state_matrix, nearest_building):
    buildings_layer = world_state_matrix[0]
    
    # Create a mask of zeros with the same shape as buildings_layer
    mask = torch.zeros_like(buildings_layer, dtype=torch.float)
    
    # Set the position of the nearest_building in the mask to 1
    mask[nearest_building[0], nearest_building[1]] = 1
    
    # Use dilation to expand the mask and get the entire building
    kernel_size = 3  # We assume buildings can be isolated by a 3x3 kernel. You might adjust this.
    building_mask = F.max_pool2d(mask[None, None], kernel_size, stride=1, padding=(kernel_size - 1) // 2) > 0
    
    return building_mask[0, 0]