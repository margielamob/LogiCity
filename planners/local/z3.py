import re
import logging
import importlib
import numpy as np
from z3 import *
from yaml import load, FullLoader
from core.config import *


logger = logging.getLogger(__name__)

class Z3Planner:
    def __init__(self, yaml_path):        
        # Load the yaml file, create the predicates and rules
        logger.info("Using Z3 solver as local planner")
        with open(yaml_path, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        
        self._create_entities()
        self._create_predicates()
        self._create_rules()

    def _create_entities(self):
        # Create Z3 sorts for each entity type
        self.entity_types = {}
        for entity_type in self.data["EntityTypes"]:
            # Create a Z3 sort (type) for each entity
            self.entity_types[entity_type] = DeclareSort(entity_type)
        # Print the entity types
        entity_types_info = "\n".join(["- {}: {}".format(entity, sort) for entity, sort in self.entity_types.items()])
        logger.info("Number of Entity Types: {}\nEntity Types:\n{}".format(len(self.entity_types), entity_types_info))

    def _create_predicates(self):
        self.predicates = {}
        for pred_dict in self.data["Predicates"]:
            (pred_name, info), = pred_dict.items()
            method_name = info["method"].split('(')[0]
            arity = info["arity"]
            z3_func = None
            if arity == 1:
                # Unary predicate
                entity_type = info["method"].split('(')[1].split(')')[0]
                z3_func = Function(method_name, self.entity_types[entity_type], BoolSort())
            elif arity == 2:
                # Binary predicate
                types = info["method"].split('(')[1].split(')')[0].split(', ')
                z3_func = Function(method_name, self.entity_types[types[0]], self.entity_types[types[1]], BoolSort())

            # Store complete predicate information
            self.predicates[pred_name] = {
                "instance": z3_func,
                "arity": arity,
                "type": info["type"],
                "method": info["method"],
                "function": info.get("function", None),  # Optional, may be used for dynamic grounding
            }
        # Print the predicates
        predicates_info = "\n".join(["- {}: {}".format(predicate, details) for predicate, details in self.predicates.items()])
        logger.info("Number of Predicates: {}\nPredicates:\n{}".format(len(self.predicates), predicates_info))

    def _create_rules(self):
        self.rules = []
        for rule_dict in self.data["Rules"]:
            (rule_name, rule_info), = rule_dict.items()
            # Check if the rule is valid
            formula = rule_info["formula"]
            formatted_string = self.format_rule_string(rule_info["formula"])
            logger.info("Rule: {} -> \n {}".format(rule_name, formatted_string))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)
            z3_vars = {var_name: Const(var_name, self.entity_types[var_name.replace('dummy', '')]) \
                       for var_name in var_names}

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'self.predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name, z3_var in z3_vars.items():
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')

            # Evaluate the modified formula string to create the Z3 expression
            z3_formula = eval(formula)  # Still using eval(), but with a controlled environment
            self.rules.append(z3_formula)

    def _extract_variables(self, formula):
        # Find the variable declaration part of the formula (after 'Forall([' or 'Exists([')
        match = re.search(r'Forall\(\[([^\]]*)\]', formula)
        if not match:
            match = re.search(r'Exists\(\[([^\]]*)\]', formula)
        
        if match:
            # Extract variable names from the matched string
            var_section = match.group(1)
            # Remove whitespace and split by commas
            variables = [var.strip() for var in var_section.split(',')]
            return variables
        else:
            # If no match, return an empty list
            return []
    
    # Process world matrix to ground the world state predicates
    def add_world_data(self, world_matrix, intersect_matrix, agents):
        # Check if static predicates have been grounded, if not, ground them
        if not hasattr(self, 'static_groundings'):
            self.static_groundings = {}
            self.ground_static_predicates(world_matrix, intersect_matrix, agents)

        # Ground dynamic predicates based on the current world state
        dynamic_groundings = self.ground_dynamic_predicates(world_matrix, intersect_matrix, agents)

        # Add the grounded predicates as facts to the solver
        for predicate, groundings in {**self.static_groundings, **dynamic_groundings}.items():
            for entities, value in groundings.items():
                if value:
                    if isinstance(entities, tuple):
                        # Binary predicate
                        self.solver.add(predicate(*entities))
                    else:
                        # Unary predicate
                        self.solver.add(predicate(entities))
    
    def ground_static_predicates(self, world_matrix, intersect_matrix, agents):
        # Ground static predicates
        for pred_name, pred_info in self.predicates.items():
            if pred_info["type"] == "S":
                self.static_groundings[pred_info["instance"]] = \
                    self.ground_predicate(pred_info, world_matrix, intersect_matrix, agents)

    def ground_dynamic_predicates(self, world_matrix, intersect_matrix, agents):
        # Ground dynamic predicates
        dynamic_groundings = {}
        for pred_name, pred_info in self.predicates.items():
            if pred_info["type"] == "D":
                dynamic_groundings[pred_info["instance"]] = \
                    self.ground_predicate(pred_info, world_matrix, intersect_matrix, agents)
        return dynamic_groundings

    def ground_predicate(self, pred_info, world_matrix, intersect_matrix, agents):
        # Generic method to ground a predicate
        groundings = {}
        predicate_function = pred_info["instance"]
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]
        if method_full_name == "None":
            return groundings
        module_name, method_name = method_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        method = getattr(module, method_name)

        if arity == 1:
            # Unary predicate grounding
            for entity in self.entities[predicate_function.domain().name()]:
                value = method(world_matrix, intersect_matrix, agents, entity)
                groundings[entity] = value
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in self.entities[predicate_function.domain(0).name()]:
                for entity2 in self.entities[predicate_function.domain(1).name()]:
                    value = method(world_matrix, intersect_matrix, agents, entity1, entity2)
                    groundings[(entity1, entity2)] = value
        return groundings

    def plan(self, world_matrix, intersect_matrix, agents):
        # 1. Init solver and entities
        s = Solver()
        # Check if entities have not been initialized or if they need to be updated
        if not self.entities:
            self.world2entity(world_matrix, intersect_matrix, agents)

        # 2. Add the grounded predicates as facts to the model
        self.add_world_data(s, world_matrix, intersect_matrix, agents)

        # 3. Solve the FOL problem
        if s.check() == sat:
            m = s.model()
        else:
            raise ValueError("No solution found!")

        # 4. Convert the solution to actions
        agents_actions = self.interpret_solution(m, agents)
        return agents_actions
    
    def world2entity(self, world_matrix, intersect_matrix, agents):
        self.entities = {}
        for entity_type in self.entity_types.keys():
            self.entities[entity_type] = []
            # For Agents
            if entity_type == "Agent":
                for agent in agents:
                    agent_id = agent.layer_id
                    agent_type = agent.type
                    agent_name = f"Agent_{agent_type}_{agent_id}"
                    # Create a Z3 constant for the agent
                    agent_entity = Const(agent_name, self.entity_types['Agent'])
                    self.entities[entity_type].append(agent_entity)
            elif entity_type == "Intersection":
                # For Intersections
                unique_intersections = np.unique(intersect_matrix[0])
                unique_intersections = unique_intersections[unique_intersections != 0]
                for intersection_id in unique_intersections:
                    intersection_name = f"intersection_{intersection_id}"
                    # Create a Z3 constant for the intersection
                    intersection_entity = Const(intersection_name, self.entity_types['Intersection'])
                    self.entities[entity_type].append(intersection_entity)
                assert len(unique_intersections) == NUM_INTERSECTIONS_BLOCKS
        assert "Agent" in self.entities.keys() and "Intersection" in self.entities.keys()

    def format_rule_string(self, rule_str):
        indent_level = 0
        formatted_str = ""
        for char in rule_str:
            if char == ',':
                formatted_str += ',\n' + ' ' * 4 * indent_level
            elif char == '(':
                formatted_str += '(\n' + ' ' * 4 * (indent_level + 1)
                indent_level += 1
            elif char == ')':
                indent_level -= 1
                formatted_str += '\n' + ' ' * 4 * indent_level + ')'
            else:
                formatted_str += char
        return formatted_str
