import torch
import torch.nn.functional as F
from core.config import *

def sample_start_goal(world_matrix, available_street_id, availabe_building_id, kernel_size):
    # Slice the building and street layer
    street_layer = world_matrix[STREET_ID]

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
    building_layer = world_matrix[BUILDING_ID]
    houses_offices = torch.any(torch.stack([(building_layer == val) for val in availabe_building_id]), dim=0)
    conv_res = F.conv2d(houses_offices.float().unsqueeze(0).unsqueeze(0), kernel, padding=padding)

    # Find cells that are walking streets and have a house or office around them
    desired_locations = (walking_streets & (conv_res.squeeze() > 0))

    return desired_locations

def sample_start_goal_vh(world_matrix, available_street_id, availabe_building_id, kernel_size):
    # exclude corner points
    # Slice the building and street layer
    street_layer = world_matrix[STREET_ID]

    # Find all walking street cells
    walking_streets = (street_layer == available_street_id)

    # Define a kernel that captures cells around a central cell.
    # This kernel will look for a house or office around the central cell.
    assert kernel_size %2 != 0
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    center = kernel_size // 2
    kernel[center, :] = 1
    kernel[:, center] = 1
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = center

    # Check for houses and offices around each cell
    building_layer = world_matrix[BUILDING_ID]
    houses_offices = torch.any(torch.stack([(building_layer == val) for val in availabe_building_id]), dim=0)
    conv_res = F.conv2d(houses_offices.float().unsqueeze(0).unsqueeze(0), kernel, padding=padding)

    # Find cells that are walking streets and have a house or office around them
    desired_locations = (walking_streets & (conv_res.squeeze() > 0))

    return desired_locations

def sample_determine_start_goal(agent_type, id):
    start_goal_dict = {
        'Pedestrian': {
            1: (torch.tensor([30, 45]), torch.tensor([80, 56])),
            2: (torch.tensor([80, 56]), torch.tensor([30, 45])),
            3: (torch.tensor([45, 30]), torch.tensor([56, 80])),
            4: (torch.tensor([56, 80]), torch.tensor([45, 30]))
        },
        'Car': {
            1: (torch.tensor([20, 48]), torch.tensor([91, 48])),
            2: (torch.tensor([91, 54]), torch.tensor([54, 91])),
            3: (torch.tensor([54, 14]), torch.tensor([14, 54])),
            4: (torch.tensor([48, 91]), torch.tensor([48, 14]))
        }
    }
    return start_goal_dict[agent_type][id]