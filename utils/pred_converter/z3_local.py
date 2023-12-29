import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from core.config import *
import logging

logger = logging.getLogger(__name__)

TYPE_MAP = {v: k for k, v in LABEL_MAP.items()}

def IsAt(world_matrix, intersect_matrix, agents, entity1, entity2):
    # Must be "Agents" at "Intersections"
    assert "Agent" in entity1
    assert "Intersection" in entity2
    if ("dummy" in entity1) or ("dummy" in entity2):
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

def IsInterCarEmpty(world_matrix, intersect_matrix, agents, entity):
    assert "Intersection" in entity
    if "dummy" in entity:
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
    if "dummy" in entity:
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

def check_is_in_intersection(world, agent_id, agent_type, intersect_matrix, agents):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if intersect_matrix[1, agent_position[0], agent_position[1]]:
        return torch.tensor([1.0, 1.0])
    else:
        return torch.tensor([0.0, 0.0])

def check_is_ambulance_in_intersection(world, agent_id, agent_type, intersect_matrix, agents):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    if not intersect_matrix[1, agent_position[0], agent_position[1]]:
        return torch.tensor([0.0, 0.0])
    else:
        local_intersection = intersect_matrix[1] == intersect_matrix[1, agent_position[0], agent_position[1]]
        intersection_positions = (local_intersection).nonzero()
        xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
        ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
        partial_world = world[BASIC_LAYER:, ymin:ymax+1, xmin:xmax+1]
        amb = torch.tensor([0.0, 0.0])
        for i, agent in enumerate(agents):
            if "ambulance" in agent.concepts.keys():
                if agent.concepts["ambulance"] == 1.0:
                    # Create mask for integer values (except 0)
                    int_mask = partial_world[i] == TYPE_MAP["Car"]
                    if int_mask.any():
                        amb = torch.tensor([1.0, 1.0])
            else:
                continue
        return amb

def IsCar(world_matrix, intersect_matrix, agents, entity):
    assert "Agent" in entity
    if "dummy" in entity:
        return 0
    if "Car" in entity:
        return 1
    else:
        return 0

def intersection_empty_ped(world, agent_id, agent_type, intersect_matrix, agents):
    agent_layer = world[agent_id]
    agent_position = (agent_layer == TYPE_MAP[agent_type]).nonzero()[0]
    # empty need to check the squares in intersetcion[2]
    if not intersect_matrix[2, agent_position[0], agent_position[1]]:
        return torch.tensor([1.0, 1.0])
    else:
        local_intersection = intersect_matrix[2] == intersect_matrix[2, agent_position[0], agent_position[1]]
        intersection_positions = (local_intersection).nonzero()
        xmin, xmax = min(intersection_positions[:, 1]), max(intersection_positions[:, 1])
        ymin, ymax = min(intersection_positions[:, 0]), max(intersection_positions[:, 0])
        # T junctions
        if ymax==ymin:
            if ymax < TRAFFIC_STREET_WID + 3*WALKING_STREET_WID + 2*BUILDING_SIZE:
                ymin = 0
            else:
                ymax = world.shape[1] - 1
        elif xmax==xmin:
            if xmax < TRAFFIC_STREET_WID + 3*WALKING_STREET_WID + 2*BUILDING_SIZE:
                xmin = 0
            else:
                xmax = world.shape[2] - 1
        partial_world = world[BASIC_LAYER:, ymin:ymax+1, xmin:xmax+1]
        # EXCLUDE myself
        partial_world = torch.cat([partial_world[:agent_id-BASIC_LAYER], partial_world[agent_id-BASIC_LAYER+1:]], dim=0)
        # Create mask for integer values (except 0)
        int_mask = partial_world == TYPE_MAP["Pedestrian"]

        # Create mask for float values and zero
        float_mask = partial_world != TYPE_MAP["Pedestrian"]

        # Update values using masks
        partial_world[int_mask] = 1
        partial_world[float_mask] = 0

        if partial_world.any():
            return torch.tensor([0.0, 0.0])
        else:
            return torch.tensor([1.0, 1.0])

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
    if ("Car" not in entity1) or ("Car" not in entity2):
        return 0
    if "dummy" in entity1 or "dummy" in entity2:
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
    if "dummy" in entity:
        return 0
    if "Pedestrian" in entity:
        return 1
    else:
        return 0

def is_ambulance(world, agent_id, agent_type, intersect_matrix, agents):
    if agent_type == "Car":
        # note that agent_id is the layer_id, not the id in agents
        if "ambulance" in agents[agent_id-BASIC_LAYER].concepts.keys():
            if agents[agent_id-BASIC_LAYER].concepts["ambulance"] == 1.0:
                return torch.tensor([1.0, 1.0])
    return torch.tensor([0.0, 0.0])

def is_tiro(world, agent_id, agent_type, intersect_matrix, agents):
    if agent_type == "Car":
        # note that agent_id is the layer_id, not the id in agents
        if "tiro" in agents[agent_id-BASIC_LAYER].concepts.keys():
            if agents[agent_id-BASIC_LAYER].concepts["tiro"] == 1.0:
                return torch.tensor([1.0, 1.0])
    return torch.tensor([0.0, 0.0])

def is_bus(world, agent_id, agent_type, intersect_matrix, agents):
    if agent_type == "Car":
        # note that agent_id is the layer_id, not the id in agents
        if "bus" in agents[agent_id-BASIC_LAYER].concepts.keys():
            if agents[agent_id-BASIC_LAYER].concepts["bus"] == 1.0:
                return torch.tensor([1.0, 1.0])
    return torch.tensor([0.0, 0.0])

def is_many_ped_around(world, agent_id, agent_type, intersect_matrix, agents):
    if agent_type == "Car":
        # note that agent_id is the layer_id, not the id in agents
        if "bus" in agents[agent_id-BASIC_LAYER].concepts.keys():
            if agents[agent_id-BASIC_LAYER].concepts["bus"] == 1.0:
                ped_layers = []
                for agent in agents:
                    if agent.concepts["type"] == "Pedestrian":
                        # note that agent_id is the layer_id, not the id in agents
                            ped_layers.append(world[agent.layer_id].unsqueeze(0))
                ped_world = torch.cat(ped_layers, dim=0)
                ego_center = (world[agent_id] == TYPE_MAP[agent_type]).nonzero()[0].tolist()
                partial_world = ped_world[:, ego_center[0]-BUS_SEEK_RANGE:ego_center[0]+BUS_SEEK_RANGE+1, ego_center[1]-BUS_SEEK_RANGE:ego_center[1]+BUS_SEEK_RANGE+1]
                # Create mask for integer values (except 0)
                int_mask = partial_world == TYPE_MAP["Pedestrian"]
                if int_mask.sum() >= BUS_PASSENGER_NUM:
                    return torch.tensor([1.0, 1.0])
        elif "tiro" in agents[agent_id-BASIC_LAYER].concepts.keys():
            if agents[agent_id-BASIC_LAYER].concepts["tiro"] == 1.0:
                ped_layers = []
                for agent in agents:
                    if agent.concepts["type"] == "Pedestrian":
                        # note that agent_id is the layer_id, not the id in agents
                            ped_layers.append(world[agent.layer_id].unsqueeze(0))
                ped_world = torch.cat(ped_layers, dim=0)
                ego_center = (world[agent_id] == TYPE_MAP[agent_type]).nonzero()[0].tolist()
                partial_world = ped_world[:, ego_center[0]-TIRO_SEEK_RANGE:ego_center[0]+TIRO_SEEK_RANGE+1, ego_center[1]-TIRO_SEEK_RANGE:ego_center[1]+TIRO_SEEK_RANGE+1]
                # Create mask for integer values (except 0)
                int_mask = partial_world == TYPE_MAP["Pedestrian"]
                if int_mask.sum() >= TIRO_PED_NUM:
                    return torch.tensor([1.0, 1.0])            
    return torch.tensor([0.0, 0.0])

def is_mayor(world, agent_id, agent_type, intersect_matrix, agents):
    if agent_type == "Pedestrian":
        # note that agent_id is the layer_id, not the id in agents
        if "mayor" in agents[agent_id-BASIC_LAYER].concepts.keys():
            if agents[agent_id-BASIC_LAYER].concepts["mayor"] == 1.0:
                return torch.tensor([1.0, 1.0])
    return torch.tensor([0.0, 0.0])

def is_mayor_around(world, agent_id, agent_type, intersect_matrix, agents):
    if agent_type == "Car":
        return torch.tensor([0.0, 0.0])
    else:
        mayor_layers = []
        for agent in agents:
            if agent.concepts["type"] == "Pedestrian":
                # note that agent_id is the layer_id, not the id in agents
                if "mayor" in agent.concepts.keys():
                    if agent.concepts["mayor"] == 1.0:
                        mayor_layers.append(world[agent.layer_id].unsqueeze(0))
        mayor_world = torch.cat(mayor_layers, dim=0)
        ego_center = (world[agent_id] == TYPE_MAP[agent_type]).nonzero()[0].tolist()
        partial_world = mayor_world[:, ego_center[0]-MAYOR_AFFECT_RANGE:ego_center[0]+MAYOR_AFFECT_RANGE+1, ego_center[1]-MAYOR_AFFECT_RANGE:ego_center[1]+MAYOR_AFFECT_RANGE+1]
        # Create mask for integer values (except 0)
        int_mask = partial_world == TYPE_MAP["Pedestrian"]
        if int_mask.any():
            return torch.tensor([1.0, 1.0])
        else:
            return torch.tensor([0.0, 0.0])
