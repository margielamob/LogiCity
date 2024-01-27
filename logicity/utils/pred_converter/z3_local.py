import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ...core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def IsAt(world_matrix, intersect_matrix, agents, entity1, entity2):
    # Must be "Agents" at "Intersections"
    assert "Agent" in entity1
    assert "Intersection" in entity2
    if "PH" in entity1 or "PH" in entity2:
        return 0

    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    # at intersection needs to care if the car is "entering" or "leaving", so use intersect_matrix[0]
    _, inter_id = entity2.split("_")
    if agent_type == "Car":
        if intersect_matrix[0, agent_position[0], agent_position[1]] == int(inter_id):
            return 1
        else:
            return 0
    else:
        if intersect_matrix[1, agent_position[0], agent_position[1]] == int(inter_id):
            return 1
        else:
            return 0

def CollidingClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    assert "Agent" in entity1
    assert "Agent" in entity2
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    # TODO: Colliding close checker
    # 1. Get the position of the two agents
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    agent_layer1 = world_matrix[int(layer_id1)]
    agent_layer2 = world_matrix[int(layer_id2)]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    # 2. Get the moving direction of the first agent
    if layer_id1 in agents.keys():
        agent1_dire = agents[layer_id1].moving_direction
    else:
        assert "ego_{}".format(layer_id1) in agents.keys()
        agent1_dire = agents["ego_{}".format(layer_id1)].moving_direction
    if agent1_dire == None:
        return 0
    else:
        dist = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
        if dist > OCC_CHECK_RANGE[agent_type1]:
            return 0
        else:
            agent1_dire_vec = torch.tensor(DIRECTION_VECTOR[agent1_dire])
            angle = torch.acos(torch.dot(agent1_dire_vec, (agent_position2 - agent_position1)) / dist)
            if angle < OCC_CHECK_ANGEL:
                if agent_type1 == "Car":
                    # Cars will definitely stop to avoid collide
                    return 1
                else:
                    # Pedestrians will probably stop to avoid collide
                    sample = np.random.rand()
                    if sample < PED_AGGR:
                        return 1
                    else:
                        return 0
    return 0

def IsInterCarEmpty(world_matrix, intersect_matrix, agents, entity):
    assert "Intersection" in entity
    if "PH" in entity:
        return 0
    _, inter_id = entity.split("_")
    inter_id = int(inter_id)
    # check if there is a car in the intersection, use block
    local_intersection = intersect_matrix[2] == inter_id
    intersection_positions = (local_intersection).nonzero()
    if intersection_positions.shape[0] == 0:
        return 1
    xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
    ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
    partial_world = world_matrix[BASIC_LAYER:, ymin:ymax+1, xmin:xmax+1]
    # Create mask for integer values (except 0)
    int_mask = partial_world == TYPE_MAP["Car"]

    if int_mask.any():
        return 0
    else:
        return 1

def IsInterEmpty(world_matrix, intersect_matrix, agents, entity):
    assert "Intersection" in entity
    if "PH" in entity:
        return 0
    _, inter_id = entity.split("_")
    inter_id = int(inter_id)
    # check if there is an agent in the intersection, use block
    local_intersection = intersect_matrix[2] == inter_id
    intersection_positions = (local_intersection).nonzero()
    if intersection_positions.shape[0] == 0:
        return 1
    xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
    ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
    partial_world = world_matrix[BASIC_LAYER:, ymin:ymax+1, xmin:xmax+1]
    # Create mask for integer values (except 0)
    int_mask = torch.logical_or(partial_world == TYPE_MAP["Car"], partial_world == TYPE_MAP["Pedestrian"])

    if int_mask.any():
        return 0
    else:
        return 1

def IsInInter(world_matrix, intersect_matrix, agents, entity1, entity2):
    # Must be "Agents" in "Intersections"
    assert "Agent" in entity1
    assert "Intersection" in entity2
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type, layer_id = entity1.split("_")
    layer_id = int(layer_id)
    agent_layer = world_matrix[layer_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    # at intersection needs to care if the car is "entering" or "leaving", so use intersect_matrix[0]
    _, inter_id = entity2.split("_")

    if intersect_matrix[2, agent_position[0], agent_position[1]] == int(inter_id):
        return 1
    else:
        return 0

def IsClose(world_matrix, intersect_matrix, agents, entity1, entity2):
    # Must be "Agents" close to "Agents"
    assert "Agent" in entity1
    assert "Agent" in entity2
    if entity1 == entity2:
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    _, agent_type1, layer_id1 = entity1.split("_")
    _, agent_type2, layer_id2 = entity2.split("_")
    layer_id1 = int(layer_id1)
    layer_id2 = int(layer_id2)
    agent_layer1 = world_matrix[layer_id1]
    agent_layer2 = world_matrix[layer_id2]
    agent_position1 = (agent_layer1 == TYPE_MAP[agent_type1]).nonzero()[0]
    agent_position2 = (agent_layer2 == TYPE_MAP[agent_type2]).nonzero()[0]
    eudis = torch.sqrt(torch.sum((agent_position1 - agent_position2)**2))
    if eudis <= CLOSE_RANGE:
        return 1
    else:
        return 0

def IsCar(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "PH" in entity:
        return 0
    if "Car" in entity:
        return 1
    else:
        return 0


def inter2priority_list(intersection_positions):
    '''
    intersection_positions: tensor of shape (n, 2) where n is the number of points belonging to intersections
    Returns a list of lists containing the points of the intersection in order of priority, 
    [[left side], [bottom side], [right side], [top side]]
    '''
    priority_list = [[], [], [], []]
    xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
    ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])

    for point in intersection_positions:
        y, x = point
        
        if x == xmin and ymin <= y < ymax:  # Left side
            priority_list[0].append(point.tolist())
        elif y == ymin and xmin < x <= xmax:  # Top side
            priority_list[1].append(point.tolist())
        elif x == xmax and ymin < y <= ymax:  # Right side
            priority_list[2].append(point.tolist())
        elif y == ymax and xmin <= x < xmax:  # Bottom side
            priority_list[3].append(point.tolist())

    # Find the length of the longest list
    max_len = max(len(lst) for lst in priority_list)

    # Pad shorter lists with a placeholder (e.g., [-1, -1])
    padded_priority_list = [lst + [[-1, -1]] * (max_len - len(lst)) for lst in priority_list]

    return torch.tensor(padded_priority_list)

def HigherPri(world_matrix, intersect_matrix, agents, entity1, entity2):
    assert "Agent" in entity1
    assert "Agent" in entity2
    if entity1 == entity2:
        return 0
    if ("Car" not in entity1) or ("Car" not in entity2):
        return 0
    if "PH" in entity1 or "PH" in entity2:
        return 0
    if entity1 == entity2:
        return 0
    
    _, _, other_agent_layer = entity1.split("_")
    _, _, ego_agent_layer = entity2.split("_")
    ego_agent_layer = int(ego_agent_layer)
    other_agent_layer = int(other_agent_layer)

    ego_agent_layer = world_matrix[ego_agent_layer]
    ego_agent_position = (ego_agent_layer == TYPE_MAP["Car"]).nonzero()[0]
    if not intersect_matrix[0, ego_agent_position[0], ego_agent_position[1]]:
        return 0
    else:
        other_agent_layer = world_matrix[other_agent_layer]
        other_agent_position = (other_agent_layer == TYPE_MAP["Car"]).nonzero()[0]
        if not intersect_matrix[0, other_agent_position[0], other_agent_position[1]] == \
            intersect_matrix[0, ego_agent_position[0], ego_agent_position[1]]:
            return 0
        
        local_intersection = intersect_matrix[0] == intersect_matrix[0, ego_agent_position[0], ego_agent_position[1]]
        intersection_positions = (local_intersection).nonzero()
        if intersection_positions.shape[0] == 0:
            return 0
        xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
        ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
        # T junctions, no previous cars
        if ymax==ymin or xmax==xmin:
            return 0

        priority_list = inter2priority_list(intersection_positions)
        # higher priority lines
        my_id = torch.all(((priority_list == ego_agent_position).sum(dim=1)) > 0, dim=1).nonzero()[0].item()
        other_groups = priority_list[my_id+1:, :, :].reshape(-1, 2)
        higher = torch.any(torch.all(other_groups == other_agent_position, dim=1))

        if higher:
            return 1
        else:
            return 0


def IsPed(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "PH" in entity:
        return 0
    if "Pedestrian" in entity:
        return 1
    else:
        return 0

def IsAmb(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "ambulance" in agent_concept:
        return 1
    else:
        return 0

def IsBus(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "bus" in agent_concept:
        return 1
    else:
        return 0

def IsTiro(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "Pedestrian" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "tiro" in agent_concept:
        return 1
    else:
        return 0

def IsOld(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "Car" in entity:
        return 0
    if "PH" in entity:
        return 0
    _, _, layer_id = entity.split("_")
    if layer_id in agents.keys():
        agent_concept = agents[layer_id].concepts
    else:
        assert "ego_{}".format(layer_id) in agents.keys()
        agent_concept = agents["ego_{}".format(layer_id)].concepts
    if "old" in agent_concept:
        return 1
    else:
        return 0
