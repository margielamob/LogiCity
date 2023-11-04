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
    # Pedestrian
    p1_s_x = TRAFFIC_STREET_WID+WALKING_STREET_LENGTH//2
    p1_s_y = TRAFFIC_STREET_WID+2*WALKING_STREET_WID+2*BUILDING_SIZE+2
    p1_g_x = p1_s_x + WALKING_STREET_LENGTH + TRAFFIC_STREET_WID*2
    p1_g_y = p1_s_y + TRAFFIC_STREET_WID + WALKING_STREET_WID
    # Vehicles
    c1_s_x = TRAFFIC_STREET_WID + WALKING_STREET_WID + 1
    c1_s_y = TRAFFIC_STREET_WID+3*WALKING_STREET_WID+2*BUILDING_SIZE+2
    c1_g_x = c1_s_x + WALKING_STREET_LENGTH + TRAFFIC_STREET_WID*3
    c1_g_y = c1_s_y
    c2_s_x = c1_s_y + TRAFFIC_STREET_WID//2
    c2_s_y = c1_s_x
    c2_g_x = c1_g_x
    c2_g_y = c1_g_y 
    c3_s_x = c1_g_y
    c3_s_y = c1_g_x
    c3_g_x = c1_s_x
    c3_g_y = c2_s_x
    c4_s_x = c2_g_x
    c4_s_y = c2_g_y
    c4_g_x = c3_s_x
    c4_g_y = c3_s_y
    # to debug intersection rules
    # start_goal_dict = {
    #     'Pedestrian': {
    #         1: (torch.tensor([p1_s_x, p1_s_y]), torch.tensor([p1_g_x, p1_g_y])),
    #         2: (torch.tensor([p1_g_x, p1_g_y]), torch.tensor([p1_s_y, p1_s_x])),
    #         3: (torch.tensor([p1_g_y, p1_g_x]), torch.tensor([p1_s_x, p1_s_y])),
    #         4: (torch.tensor([p1_s_y, p1_s_x]), torch.tensor([p1_g_y, p1_g_x]))
    #     },
    #     'Car': {
    #         1: (torch.tensor([c1_s_x, c1_s_y]), torch.tensor([c1_g_x, c1_g_y])),
    #         2: (torch.tensor([c2_s_x, c2_s_y]), torch.tensor([c2_g_x-10, c2_g_y])),
    #         3: (torch.tensor([c3_s_x, c3_s_y]), torch.tensor([c3_g_x, c3_g_y])),
    #         4: (torch.tensor([c4_s_x, c4_s_y]), torch.tensor([c4_g_x, c4_g_y]))
    #     }
    # }
    # to debug bus stop and reckless driver rules
    start_goal_dict = {
        'Pedestrian': {
            1: (torch.tensor([p1_s_x, p1_s_y]), torch.tensor([p1_g_x, p1_g_y])),
            2: (torch.tensor([p1_g_x, p1_g_y]), torch.tensor([p1_s_y, p1_s_x])),
            3: (torch.tensor([p1_g_y, p1_g_x]), torch.tensor([p1_s_x, p1_s_y])),
            4: (torch.tensor([p1_s_y, p1_s_x]), torch.tensor([p1_g_y, p1_g_x])),
            5: (torch.tensor([p1_g_y, p1_g_x+10]), torch.tensor([p1_g_y, p1_g_x+100])),
            6: (torch.tensor([p1_g_y, p1_g_x+10]), torch.tensor([p1_g_y, p1_g_x+100])),
            7: (torch.tensor([p1_g_y, p1_g_x+10]), torch.tensor([p1_g_y, p1_g_x+100])),
            8: (torch.tensor([p1_g_x+10, p1_g_y]), torch.tensor([p1_g_x+11, p1_g_y])),
            9: (torch.tensor([p1_g_x+11, p1_g_y]), torch.tensor([p1_g_x+12, p1_g_y])),
            10: (torch.tensor([p1_g_x+12, p1_g_y]), torch.tensor([p1_g_x+13, p1_g_y])),
        },
        'Car': {
            1: (torch.tensor([c1_s_x, c1_s_y]), torch.tensor([c1_g_x, c1_g_y])),
            2: (torch.tensor([c2_s_x, c2_s_y]), torch.tensor([c2_g_x-10, c2_g_y])),
            3: (torch.tensor([c3_s_x, c3_s_y]), torch.tensor([c3_g_x, c3_g_y])),
            4: (torch.tensor([c4_s_x, c4_s_y]), torch.tensor([c4_g_x, c4_g_y]))
        }
    }
    return start_goal_dict[agent_type][id]