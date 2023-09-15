import torch
import torch.nn.functional as F

def sample_start_goal(world_matrix, available_street_id, availabe_building_id, kernel_size):
    # Slice the building and street layer
    street_layer = world_matrix[1]

    # Find all walking street cells
    walking_streets = (street_layer == available_street_id)

    # Define a kernel that captures cells around a central cell.
    # This kernel will look for a house or office around the central cell.
    assert kernel_size %2 != 0
    kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32)
    center = kernel_size // 2
    kernel[center, center] = 0
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = center

    # Check for houses and offices around each cell
    building_layer = world_matrix[0]
    houses_offices = torch.any(torch.stack([(building_layer == val) for val in availabe_building_id]), dim=0)
    conv_res = F.conv2d(houses_offices.float().unsqueeze(0).unsqueeze(0), kernel, padding=padding)

    # Find cells that are walking streets and have a house or office around them
    desired_locations = (walking_streets & (conv_res.squeeze() > 0))

    return desired_locations