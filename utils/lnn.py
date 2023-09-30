import torch
import numpy as np

def check_is_at_intersection(world, agent_id, agent_type):
    """Check if the agent is at an intersection."""
    return torch.tensor([1.0, 1.0])

def arrived_first(world, agent_id, agent_type):
    """Check if the agent is the first arriving at an intersection."""
    return torch.tensor([0.0, 0.0])