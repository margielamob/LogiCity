import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def check_is_at_intersection(world, agent_id, agent_type, intersect_matrix):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if intersect_matrix[agent_position[0], agent_position[1]]:
        return torch.tensor([1.0, 1.0])
    else:
        return torch.tensor([0.0, 0.0])

def is_car(world, agent_id, agent_type, intersect_matrix):
    if agent_type == "Car":
        return torch.tensor([1.0, 1.0])
    else:
        return torch.tensor([0.0, 0.0])

def pedestrians_near_intersection(world, agent_id, agent_type, intersect_matrix):
    return torch.tensor([0.0, 0.0])

def intersection_empty(world, agent_id, agent_type, intersect_matrix):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if not intersect_matrix[agent_position[0], agent_position[1]]:
        return torch.tensor([1.0, 1.0])
    else:
        local_intersection = intersect_matrix == intersect_matrix[agent_position[0], agent_position[1]]
        intersection_positions = (local_intersection).nonzero()
        xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
        ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
        partial_world = world[BASIC_LAYER:, ymin-AT_INTERSECTION_E:ymax+AT_INTERSECTION_E, xmin-AT_INTERSECTION_E:xmax+AT_INTERSECTION_E]
        # EXCLUDE myself
        partial_world = torch.cat([partial_world[:agent_id-BASIC_LAYER], partial_world[agent_id-BASIC_LAYER+1:]], dim=0)
        # Create mask for integer values (except 0)
        int_mask = (partial_world % 1 == 0) & (partial_world != 0)

        # Create mask for float values and zero
        float_mask = (partial_world % 1 != 0) | (partial_world == 0)

        # Update values using masks
        partial_world[int_mask] = 1
        partial_world[float_mask] = 0

        if partial_world.any():
            return torch.tensor([0.0, 0.0])
        else:
            return torch.tensor([1.0, 1.0])

def intersection_empty_cars(world, agent_id, agent_type, intersect_matrix):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if not intersect_matrix[agent_position[0], agent_position[1]]:
        return torch.tensor([1.0, 1.0])
    else:
        local_intersection = intersect_matrix == intersect_matrix[agent_position[0], agent_position[1]]
        intersection_positions = (local_intersection).nonzero()
        xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
        ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
        partial_world = world[BASIC_LAYER:, ymin-AT_INTERSECTION_E:ymax+AT_INTERSECTION_E, xmin-AT_INTERSECTION_E:xmax+AT_INTERSECTION_E]
        # EXCLUDE myself
        partial_world = torch.cat([partial_world[:agent_id-BASIC_LAYER], partial_world[agent_id-BASIC_LAYER+1:]], dim=0)
        # Create mask for integer values (except 0)
        int_mask = partial_world == TYPE_MAP["Car"]

        # Create mask for float values and zero
        float_mask = partial_world != TYPE_MAP["Car"]

        # Update values using masks
        partial_world[int_mask] = 1
        partial_world[float_mask] = 0

        if partial_world.any():
            return torch.tensor([0.0, 0.0])
        else:
            return torch.tensor([1.0, 1.0])

def previous_cars(world, agent_id, agent_type, intersect_matrix):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if not intersect_matrix[agent_position[0], agent_position[1]]:
        return torch.tensor([0.0, 0.0])
    else:
        local_intersection = intersect_matrix == intersect_matrix[agent_position[0], agent_position[1]]
        intersection_positions = (local_intersection).nonzero()
        xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
        ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
        priority_lines = torch.tensor([[[ymin, xmax], [ymax, xmax]], [[ymax, xmin], [ymax, xmax]], [[ymin, xmin], [ymax, xmin]], [[ymin, xmin], [ymin, xmax]]])
        pos_id = (torch.all((priority_lines[:, 0] <= agent_position) & (agent_position <= priority_lines[:, 1]), dim=1)).nonzero(as_tuple=True)[0].item()
        bounds = torch.tensor([0.0, 0.0])
        for line_id in range(pos_id+1, priority_lines.shape[0]):
            line = priority_lines[line_id]
            partial_world = world[BASIC_LAYER:, line[0, 0]:line[1, 0]+1, line[0, 1]:line[1, 1]+1]
            # EXCLUDE myself
            partial_world = torch.cat([partial_world[:agent_id-BASIC_LAYER], partial_world[agent_id-BASIC_LAYER+1:]], dim=0)
            # Create mask for integer values (except 0)
            int_mask = partial_world == TYPE_MAP["Car"]
            if torch.any(int_mask):
                bounds = torch.tensor([1.0, 1.0])
    return bounds

def is_pedestrian(world, agent_id, agent_type, intersect_matrix):
    if agent_type == "Pedestrian":
        return torch.tensor([1.0, 1.0])
    else:
        return torch.tensor([0.0, 0.0])

def visualize_intersections(intersection_matrix):
    # Get unique intersection IDs (excluding 0)
    unique_intersections = np.unique(intersection_matrix)
    unique_intersections = unique_intersections[unique_intersections != 0]
    
    # Create a color map for each intersection ID
    colors = list(mcolors.CSS4_COLORS.values())
    intersection_colors = {uid: colors[i % len(colors)] for i, uid in enumerate(unique_intersections)}

    # Create an RGB visualization matrix
    vis_matrix = np.zeros((*intersection_matrix.shape, 3), dtype=np.uint8)

    for uid, color in intersection_colors.items():
        r, g, b = mcolors.hex2color(color)
        mask = (intersection_matrix == uid)
        vis_matrix[mask] = (np.array([r, g, b]) * 255).astype(np.uint8)

    # Plot
    plt.imshow(vis_matrix)
    plt.title("Intersections Visualization")
    plt.axis('off')
    plt.imsave("test.png", vis_matrix)

