import torch
from torch.distributions import Categorical
from core.config import *

class Agent:
    def __init__(self, type, size, id, world_state_matrix, concepts=None, debug=False):
        self.size = size
        self.type = type
        self.id = id
        # -1 is always the stop action
        # Actions: ["left", "right", "up", "down", "stop"]
        self.action_space = torch.tensor(range(5))
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "Left_1", 
            1: "Right_1", 
            2: "Up_1", 
            3: "Down_1", 
            4: "Stop"
            }
        self.start = None
        self.goal = None
        self.pos = None
        self.layer_id = 0
        self.reach_goal = False
        self.reach_goal_buffer = 0
        self.debug = debug
        # concepts
        if concepts == None:
            self.concepts = {}
        else:
            self.concepts = concepts
        self.init(world_state_matrix, debug)

    def init(self, world_state_matrix, debug=False):
        # init global planner, global traj, local planner, start and goal point
        pass

    def get_start(self, world_state_matrix):
        pass

    def get_goal(self, world_state_matrix):
        pass
        
    def move(self, action, ped_layer, curr_label):
        pass
        
    def get_next_action(self, world_state_matrix):
        pass

    def get_movable_area(self, world_state_matrix):
        pass

    def get_action(self, local_action_dist):
        if not torch.any(local_action_dist):
            return self.get_global_action()
        else:
            # sample from the local planner
            normalized_action_dist = local_action_dist / local_action_dist.sum()
            dist = Categorical(normalized_action_dist)
            # Sample an action index from the distribution
            action_index = dist.sample()
            # Get the actual action from the action space using the sampled index
            return self.action_space[action_index]

    def move(self, action, ped_layer):
        curr_pos = torch.nonzero((ped_layer==TYPE_MAP[self.type]).float())[0]
        assert torch.all(self.pos == curr_pos)
        next_pos = self.pos.clone()
        # becomes walked grid
        ped_layer[self.pos[0], self.pos[1]] += AGENT_WALKED_PATH_PLUS
        next_pos += self.action_to_move.get(action, torch.tensor((0, 0)))
        self.pos = next_pos.clone()
        # Update Agent Map
        ped_layer[self.start[0], self.start[1]] = TYPE_MAP[self.type] + AGENT_START_PLUS
        ped_layer[self.goal[0], self.goal[1]] = TYPE_MAP[self.type] + AGENT_GOAL_PLUS
        ped_layer[self.pos[0], self.pos[1]] = TYPE_MAP[self.type]
        return ped_layer

    def get_global_action(self):
        next_pos = self.global_traj[0]
        self.global_traj.pop(0)
        del_pos = tuple((next_pos - self.pos).tolist())
        return self.move_to_action.get(del_pos, self.action_space[-1].item())