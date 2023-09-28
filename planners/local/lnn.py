from lnn import Model, Predicates, Variables, Implies, And, Or, Not, Fact, World
from yaml import load, FullLoader
import importlib
import torch

class LNNPlanner:
    def __init__(self, yaml_path):
        self.model = Model()
        
        # Load the yaml file
        with open(yaml_path, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        
        self._create_predicates()
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
    def add_world_data(self, world_matrix, agent_id, agent_type):
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
                values = method(world_matrix, agent_id, agent_type)
                
                # Now only supporting one arity
                data_dict[self.predicates[p]["instance"]][agent_name] = values
            else:
                # By defaul False for actions
                data_dict[self.predicates[p]["instance"]][agent_name] = torch.tensor([0.0, 0.0])

        self.model.add_data(data_dict)

    def plan(self, world_matrix, agent_id, agent_type):
        self.add_world_data(world_matrix, agent_id, agent_type)
        self.model.infer()
        action_dist = self.predicates["Stop"]["instance"].get_data('Pedestrian_3')

        return self.model.plan()

    def infer(self, **kwargs):
        return self.model.infer()
