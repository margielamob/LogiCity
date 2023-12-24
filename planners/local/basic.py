from yaml import load, FullLoader

class LocalPlanner:
    def __init__(self, yaml_path):        
        # Load the yaml file, create the predicates and rules
        with open(yaml_path, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        
        self._create_entities()
        self._create_predicates()
        self._create_rules()
    
    def _create_entities(self):
        # Create the entities
        raise NotImplementedError
    
    def _create_predicates(self):
        # Create the predicates
        raise NotImplementedError
    
    def _create_rules(self):
        # Create the rules
        raise NotImplementedError
    
    def plan(self, map_data, start, goal):
        # Plan a path from start to goal
        pass

