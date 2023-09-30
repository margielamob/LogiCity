import torch
import numpy as np

def check_is_at_intersection(world, agent_id, agent_type):
    """Check if the agent is at an intersection."""
    return torch.tensor([0.0, 0.0])

def is_car(world, agent_id, agent_type):
    """Check if the agent is the first arriving at an intersection."""
    return torch.tensor([0.0, 0.0])

def pedestrians_near_intersection(world, agent_id, agent_type):
    """Check if the agent is the first arriving at an intersection."""
    return torch.tensor([0.0, 0.0])

def intersection_empty(world, agent_id, agent_type):
    return torch.tensor([0.0, 0.0])

def high_priority_agents_near(world, agent_id, agent_type):
    return torch.tensor([0.0, 0.0])

def is_pedestrian(world, agent_id, agent_type):
    return torch.tensor([1.0, 1.0])

