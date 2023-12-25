import re
import torch
import logging
import importlib
import numpy as np
from z3 import *
from core.config import *
from utils.find import find_agent
from planners.local.basic import LocalPlanner


logger = logging.getLogger(__name__)

class Z3Planner(LocalPlanner):
    def __init__(self, yaml_path):        
        super().__init__(yaml_path)

    def _create_entities(self):
        # Create Z3 sorts for each entity type
        self.entity_types = {}
        for entity_type in self.data["EntityTypes"]:
            # Create a Z3 sort (type) for each entity
            self.entity_types[entity_type] = DeclareSort(entity_type)
        # Print the entity types
        entity_types_info = "\n".join(["- {}: {}".format(entity, sort) for entity, sort in self.entity_types.items()])
        self.entities = {}
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
        self.rules = {}
        self.rule_tem = {}
        for rule_dict in self.data["Rules"]:
            (rule_name, rule_info), = rule_dict.items()
            # Check if the rule is valid
            formula = rule_info["formula"]
            formatted_string = self.format_rule_string(rule_info["formula"])
            logger.info("Rule: {} -> \n {}".format(rule_name, formatted_string))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)
            self.z3_vars = {var_name: Const(var_name, self.entity_types[var_name.replace('dummy', '')]) \
                       for var_name in var_names}

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'self.predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name, z3_var in self.z3_vars.items():
                formula = formula.replace(var_name, f'self.z3_vars["{var_name}"]')

            # Evaluate the modified formula string to create the Z3 expression
            self.rule_tem[rule_name] = formula
        rule_info = "\n".join(["- {}: {}".format(rule, details) for rule, details in self.rule_tem.items()])
        logger.info("Number of Rules: {}\nRules:\n{}".format(len(self.rule_tem), rule_info))
        logger.info("Rules will be grounded later...")

    def _extract_variables(self, formula):
        # Find the variable declaration part of the formula (after 'Forall([' or 'Exists([')
        match = re.search(r'ForAll\(\[([^\]]*)\]', formula)
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
    
    def instaniate_rules(self):
        for rule_name, rule_template in self.rule_tem.items():
            self.rules[rule_name] = []
            for agent in self.entities["Agent"]:
                # Replace placeholder in the rule template with the actual agent entity
                instantiated_rule = eval(rule_template)
                self.rules[rule_name].append(instantiated_rule)

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
            for true_entity in groundings["True"]:
                if isinstance(true_entity, tuple):
                    # Binary predicate, true case
                    self.solver.add(predicate(*true_entity))
                else:
                    # Unary predicate, true case
                    self.solver.add(predicate(true_entity))

            # If we used assertion default, then may do not need the negtive facts
            for false_entity in groundings["False"]:
                if isinstance(false_entity, tuple):
                    # Binary predicate, false case
                    self.solver.add(Not(predicate(*false_entity)))
                else:
                    # Unary predicate, false case
                    self.solver.add(Not(predicate(false_entity)))
    
    def ground_static_predicates(self, world_matrix, intersect_matrix, agents):
        # Ground static predicates
        for pred_name, pred_info in self.predicates.items():
            world_matrix_clone = world_matrix.clone()
            if pred_info["type"] == "S":
                self.static_groundings[pred_info["instance"]] = \
                    self.ground_predicate(pred_info, world_matrix_clone, intersect_matrix, agents)

    def ground_dynamic_predicates(self, world_matrix, intersect_matrix, agents):
        # Ground dynamic predicates
        dynamic_groundings = {}
        for pred_name, pred_info in self.predicates.items():
            world_matrix_clone = world_matrix.clone()
            if pred_info["type"] == "D":
                dynamic_groundings[pred_info["instance"]] = \
                    self.ground_predicate(pred_info, world_matrix_clone, intersect_matrix, agents)
        return dynamic_groundings

    def ground_predicate(self, pred_info, world_matrix, intersect_matrix, agents):
        # Return the groundings that satisfy the predicate
        # Generic method to ground a predicate
        groundings = {
            "True": [],
            "False": []
        }
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
            for entity in self.entities[predicate_function.domain(0).name()]:
                entity_name = entity.decl().name()
                value = method(world_matrix, intersect_matrix, agents, entity_name)
                if value:
                    groundings["True"].append(entity)
                else:
                    groundings["False"].append(entity)
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in self.entities[predicate_function.domain(0).name()]:
                for entity2 in self.entities[predicate_function.domain(1).name()]:
                    entity1_name = entity1.decl().name()
                    entity2_name = entity2.decl().name()
                    value = method(world_matrix, intersect_matrix, agents, entity1_name, entity2_name)
                    if value:
                        groundings["True"].append((entity1, entity2))
                    else:
                        groundings["False"].append((entity1, entity2))
        return groundings

    def plan(self, world_matrix, intersect_matrix, agents):
        # 1. Init solver and entities
        self.solver = Solver()
        # Check if entities have not been initialized or if they need to be updated
        if not self.entities:
            self.world2entity(world_matrix, intersect_matrix, agents)
        # Check if rules have not been initialized or if they need to be updated
        if not self.rules:
            self.instaniate_rules()

        # 2. Add the grounded predicates as facts to the model
        world_matrix_clone = world_matrix.clone()
        self.add_world_data(world_matrix_clone, intersect_matrix, agents)
        # 3. Add Rule as known truth to the model
        for rule_name, rule_list in self.rules.items():
            for rule in rule_list:
                self.solver.add(rule)

        # 3. Solve the FOL problem
        if self.solver.check() == sat:
            m = self.solver.model()
            # 4. Interpret the solution
            return self.interpret_solution(m, agents)
        else:
            raise ValueError("No solution found!")
    
    def interpret_solution(self, model, agents, skip=False):
        # Interpret the solution to the FOL problem
        agents_actions = {}
        for agent_entity in self.entities["Agent"]:
            if "dummy" in agent_entity.decl().name():
                continue
            entity_name = agent_entity.decl().name()
            agent = find_agent(agents, entity_name)
            agent_name = "{}_{}".format(agent.type, agent.layer_id)
            action_mapping = agent.action_mapping
            action_dist = torch.zeros_like(agent.action_dist)

            if not skip:
                for key in self.predicates.keys():
                    action = []
                    for action_id, action_name in action_mapping.items():
                        if key in action_name:
                            action.append(action_id)
                    if len(action)>0:
                        for a in action:
                            if is_true(model.evaluate(self.predicates[key]["instance"](agent_entity))):
                                action_dist[a] = 1.0
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0

            agents_actions[agent_name] = action_dist
        return agents_actions
    
    def world2entity(self, world_matrix, intersect_matrix, agents):
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
                for intersection_id in unique_intersections[:7]:
                    intersection_name = f"Intersection_{intersection_id}"
                    # Create a Z3 constant for the intersection
                    intersection_entity = Const(intersection_name, self.entity_types['Intersection'])
                    self.entities[entity_type].append(intersection_entity)
                assert len(unique_intersections) == NUM_INTERSECTIONS_BLOCKS
        assert "Agent" in self.entities.keys() and "Intersection" in self.entities.keys()
        # dummy is also part of entity
        for var_name, z3_var in self.z3_vars.items():
            self.entities[var_name.replace('dummy', '')].append(z3_var)

    def format_rule_string(self, rule_str):
        indent_level = 0
        formatted_str = ""
        bracket_stack = []  # Stack to keep track of brackets

        for char in rule_str:
            if char == ',':
                formatted_str += ',\n' + ' ' * 4 * indent_level
            elif char == '(':
                bracket_stack.append('(')
                formatted_str += '(\n' + ' ' * 4 * (indent_level + 1)
                indent_level += 1
            elif char == ')':
                if not bracket_stack or bracket_stack[-1] != '(':
                    raise ValueError("Unmatched closing bracket detected.")
                bracket_stack.pop()
                indent_level -= 1
                formatted_str += '\n' + ' ' * 4 * indent_level + ')'
            else:
                formatted_str += char

        if bracket_stack:
            raise ValueError("Unmatched opening bracket detected.")

        return formatted_str
