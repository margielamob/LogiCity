from lnn import Model, Predicates, Variables, Implies, And, Or, Not, Fact, World
from yaml import load, FullLoader
import utils.lnn_pred_converter as converter

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
                "arity": info["arity"]
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
        
        for key, rule_info in self.data["rules"].items():
            formula_str = rule_info["formula"]
            
            # Replace formula's string content to make it Python executable
            for predicate in self.data["predicates"]:
                formula_str = formula_str.replace(predicate, f'self.predicates.{predicate}(x)')
            
            for key, func in logical_mapping.items():
                formula_str = formula_str.replace(key, f'{func.__name__}')
            
            rule_instance = eval(formula_str)  # This dynamically evaluates the Python equivalent formula
            self.model.add_knowledge(rule_instance, world=World.AXIOM)
            
    def add_world_data(self, world_matrix):
        # Convert the world matrix to the format expected by LNN
        data_dict = {}
        
        for predicate_name, details in self.data["predicates"].items():
            method_name = details["method"]
            # Dynamically call the method to process the world matrix
            values = getattr(converter, method_name)(world_matrix, details)
            data_dict[self.predicates[predicate_name]["instance"].name] = values

        self.model.add_data(data_dict)
    
    # Example method to process world matrix for a specific predicate
    def add_world_data(self, world_matrix):
        # Convert the world matrix to the format expected by LNN
        data_dict = {}
        
        for predicate_name, details in self.data["predicates"].items():
            method_name = details["method"]
            # Dynamically call the method to process the world matrix
            values = getattr(converter, method_name)(world_matrix, details)
            data_dict[self.predicates[predicate_name]["instance"].name] = values

        self.model.add_data(data_dict)

    def infer(self, **kwargs):
        return self.model.infer()
