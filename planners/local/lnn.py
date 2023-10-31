from lnn import Model, Predicates, Variables, Implies, And, Or, Not, Fact, World, Direction
from yaml import load, FullLoader
import importlib
import torch

class LNNPlanner:
    def __init__(self, yaml_path):        
        # Load the yaml file
        with open(yaml_path, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        
        self._create_predicates()
        self.model = Model()
        self._create_rules()
        
    def _create_predicates(self):
        # Using a dictionary to store the arity as well
        self.predicates = {}
        for p in self.data["predicates"]:
            predicate, info = list(p.items())[0]
            self.predicates[predicate] = {
                "instance": Predicates(predicate),
                "method": info["method"],
                "arity": info["arity"],
                "description": info["description"]
            }
        
    def _create_rules(self):
        # For the sake of simplicity, we'll only handle basic logical constructs here (And, Or, Implies)
        # More complex rules might require further adjustments
        logical_mapping = {
            'And': And,
            'Or': Or,
            'Implies': Implies,
            'Not': Not
        }
        
        x = Variables('x')  # For demonstration, considering only one variable for now
        
        for r in self.data["rules"]:
            rule_name, rule_info = list(r.items())[0]
            formula_str = rule_info["formula"]
            
            # Replace formula's string content to make it Python executable
            for predicate in self.predicates:
                formula_str = formula_str.replace(predicate, "self.predicates['{}']['instance']".format(str(predicate)))
            
            for key, func in logical_mapping.items():
                formula_str = formula_str.replace(key, f'{func.__name__}')
            
            rule_instance = eval(formula_str)  # This dynamically evaluates the Python equivalent formula
            # All the rules are considered as axioms for now
            self.model.add_knowledge(rule_instance, world=World.AXIOM)
    
    # Example method to process world matrix for a specific predicate
    def add_world_data(self, world_matrix, intersect_matrix, agent_id, agent_type, agents):
        # Convert the world matrix to the format expected by LNN
        data_dict = {}
        agent_name = "{}_{}".format(agent_type, agent_id)
        
        for p in self.predicates.keys():
            data_dict[self.predicates[p]["instance"]] = {}

            if self.predicates[p]["method"]!='None':

                method_full_name = self.predicates[p]["method"]
                # Split the string to separate module name and method name
                module_name, method_name = method_full_name.rsplit('.', 1)

                # Dynamically import the module
                module = importlib.import_module(module_name)

                # Get the method from the module
                method = getattr(module, method_name)

                # Call the method
                values = method(world_matrix, agent_id, agent_type, intersect_matrix, agents)
                
                # Now only supporting one arity
                data_dict[self.predicates[p]["instance"]][agent_name] = values

        self.model.add_data(data_dict)

    def plan(self, world_matrix, intersect_matrix, agents):
        self.model.reset_bounds()
        for p in self.predicates.keys():
            self.predicates[p]["instance"].flush()
        agents_actions = {}
        for agent in agents:
            # ego id
            agent_id = agent.layer_id
            agent_type = agent.type
            agent_name = "{}_{}".format(agent_type, agent_id)
            self.add_world_data(world_matrix, intersect_matrix, agent_id, agent_type, agents)
        self.model.infer(Direction.UPWARD)
        self.model.infer(Direction.DOWNWARD)
        # use LNN to get the action distribution
        for agent in agents:
            agent_id = agent.layer_id
            agent_type = agent.type
            agent_name = "{}_{}".format(agent_type, agent_id)
            action_mapping = agent.action_mapping
            action_dist = torch.zeros_like(agent.action_dist)
            for key in self.predicates.keys():
                action = []
                for action_id, action_name in action_mapping.items():
                    if key in action_name:
                        action.append(action_id)
                if len(action)>0:
                    for a in action:
                        action_dist[a] = self.convert(key, agent_name)
            agents_actions[agent_name] = action_dist
        return agents_actions

    def convert(self, key_name, agent_name):
        pred_list = []
        pred_list.append(self.predicates[key_name]["instance"].get_data(agent_name))
        LU_bound = torch.cat(pred_list, dim=0)

        value = torch.avg_pool1d(LU_bound, kernel_size=2)
        value[value==0.5] = 0.0
        value = torch.clip(value.sum(), 0.0, 1.0)
        return value

    def get_current_lnn_state(self, logic_groundings, agents):
        all_grounding = []
        for agent in agents:
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            agent_grounding = []
            for pred in logic_groundings.keys():
                if pred == 'Stop' or pred == "Normal":
                    continue
                agent_grounding.append(self.predicates[pred]["instance"].get_data(agent_name))
            all_grounding.append(torch.cat(agent_grounding, dim=0))
        return torch.stack(all_grounding, dim=0)