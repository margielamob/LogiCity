import torch

class Agent:
    def __init__(self, type, size, id, world_state_matrix, debug=False):
        self.size = size
        self.type = type
        self.id = id
        # -1 is always the stop action
        # Actions: ["left", "right", "up", "down", "stop"]
        self.action_space = torch.tensor(range(5))
        self.action_dist = torch.zeros_like(self.action_space).float()
        self.action_mapping = {
            0: "left_1", 
            1: "right_1", 
            2: "up_1", 
            3: "down_1", 
            4: "stop"
            }
        self.start = None
        self.goal = None
        self.pos = None
        self.layer_id = 0
        self.reach_goal = False
        self.debug = debug
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