class Agent:
    def __init__(self, type, size):
        self.size = size
        self.type = type
        # Actions: ["left", "right", "up", "down", "stop"]
        self.action_space = 5

    def init(self, global_planner, local_planner=None):
        # 
        pass
        
    def move(self, action):
        # Define how the agent moves based on action
        pass
        
    def get_next_action(self, world_state_martix):
        pass