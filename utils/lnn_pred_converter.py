import torch
import numpy as np
from skimage.draw import line
from scipy.ndimage import label
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
        is_integer = (partial_world == partial_world.long())
        if is_integer.any():
            return torch.tensor([0.0, 0.0])
        else:
            return torch.tensor([1.0, 1.0])

def high_priority_agents_near(world, agent_id, agent_type, intersect_matrix):
    return torch.tensor([0.0, 0.0])

def is_pedestrian(world, agent_id, agent_type, intersect_matrix):
    if agent_type == "Pedestrian":
        return torch.tensor([1.0, 1.0])
    else:
        return torch.tensor([0.0, 0.0])

def generate_intersection_matrix(world):
    # Extract the 0-th layer of the world matrix
    world_layer = world[BLOCK_ID, :, :]
    
    # Extract the unique block IDs from the 0-th layer
    unique_blocks = set(world_layer.flatten().tolist())
    unique_blocks.remove(0)  # Assuming 0 is the ID for non-block pixels
    
    # Find the corners of the blocks
    corners = {}
    for block_id in unique_blocks:
        block_positions = (world_layer == block_id).nonzero()
        xmin, xmax = min(block_positions[:, 1]), max(block_positions[:, 1])
        ymin, ymax = min(block_positions[:, 0]), max(block_positions[:, 0])
        corners[block_id] = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

    intersection_matrix = np.zeros_like(world_layer, dtype=bool)

    for block_id, block_corners in corners.items():
        for other_block_id, other_block_corners in corners.items():
            if block_id != other_block_id:
                for corner in block_corners:
                    for other_corner in other_block_corners:
                        if np.linalg.norm(np.array(corner) - np.array(other_corner)) == 14:
                            rr, cc = line(corner[0], corner[1], other_corner[0], other_corner[1])
                            intersection_matrix[rr, cc] = True

    # Label connected regions in the intersection matrix
    labeled_matrix, num = label(intersection_matrix)
    assert num == NUM_INTERSECTIONS, "Number of intersections is not 32"
    return torch.tensor(labeled_matrix)

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

