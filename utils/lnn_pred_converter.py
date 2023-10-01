import torch
import numpy as np
from skimage.draw import line
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def check_is_at_intersection(world, agent_id, agent_type, intersect_matrix):
    return torch.tensor([0.0, 0.0])

def is_car(world, agent_id, agent_type):
    return torch.tensor([0.0, 0.0])

def pedestrians_near_intersection(world, agent_id, agent_type, intersect_matrix):
    return torch.tensor([0.0, 0.0])

def intersection_empty(world, agent_id, agent_type, intersect_matrix):
    return torch.tensor([0.0, 0.0])

def high_priority_agents_near(world, agent_id, agent_type, intersect_matrix):
    return torch.tensor([0.0, 0.0])

def is_pedestrian(world, agent_id, agent_type, intersect_matrix):
    if agent_type == "Pedestrian":
        return torch.tensor([1.0, 1.0])
    else:
        return torch.tensor([0.0, 0.0])

def generate_intersection_matrix(world):
    # Extract the 0-th layer of the world matrix
    world_layer = world[0, :, :]
    
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
    assert num == 32, "Number of intersections is not 32"
    return labeled_matrix

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

