import re
import time
import torch
import logging
import importlib
import numpy as np
from z3 import *
from core.config import *
from multiprocessing import Pool
from utils.find import find_agent
from utils.sample import split_into_subsets
from planners.local.basic import LocalPlanner

logger = logging.getLogger(__name__)

class Z3PlannerLocal(LocalPlanner):
    def __init__(self, yaml_path):        
        super().__init__(yaml_path)

    def _create_entities(self):
        # Create Z3 sorts for each entity type
        self.entity_types = []
        for entity_type in self.data["EntityTypes"]:
            # Create a Z3 sort (type) for each entity
            self.entity_types.append(entity_type)
        # Print the entity types
        entity_types_info = "\n".join(["- {}".format(entity) for entity in self.entity_types])
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
                z3_func = "Function({method_name}, {entity_type}, BoolSort())"
            elif arity == 2:
                # Binary predicate
                types = info["method"].split('(')[1].split(')')[0].split(', ')
                z3_func = "Function({method_name}, {types[0]}, {types[1]}, BoolSort())"

            # Store complete predicate information
            self.predicates[pred_name] = {
                "instance": z3_func,
                "arity": arity,
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
                formula = formula.replace(method_name, f'predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name, z3_var in self.z3_vars.items():
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')

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

    def plan(self, world_matrix, intersect_matrix, agents):
        # 1. Break the global world matrix into local world matrix and split the agents and intersections
        local_world_matrix = world_matrix.clone()
        partial_agents, partial_world, partial_intersections = self.break_world_matrix(local_world_matrix, agents, intersect_matrix)
        # 2. multi-processing to solve each sub-problem
        with Pool(processes=NUM_PROCESS) as pool:
            results = pool.starmap(solve_sub_problem, 
                                [(ego_name, self.rule_tem, self.entity_types, self.predicates, \
                                  partial_agents[ego_name], partial_world[ego_name], partial_intersections[ego_name]) \
                                    for ego_name in partial_agents.keys()])

        return results
    
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

def solve_sub_problem(ego_name, 
                      rule_tem, 
                      entity_types, 
                      predicates, 
                      partial_agents, 
                      partial_world, 
                      partial_intersections):
    # 1. create solver
    local_solver = Solver()
    # 2. create sorts
    entity_sorts = {}
    for entity_type in entity_types:
        entity_sorts[entity_type] = DeclareSort(entity_type)
    # 3. partial world to entities
    local_entities = world2entity(entity_sorts, partial_world, partial_intersections)
    # 4. create, ground predicates and add to solver
    for pred_name, pred_info in predicates.items():
        eval_pred = eval(pred_info["instance"])
        pred_info["instance"] = eval_pred
        arity = pred_info["arity"]

        # Import the grounding method
        method_full_name = pred_info["function"]
        if method_full_name == "None":
            continue
        module_name, method_name = method_full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        method = getattr(module, method_name)

        if arity == 1:
            # Unary predicate grounding
            for entity in local_entities[eval_pred.domain(0).name()]:
                entity_name = entity.decl().name()
                value = method(partial_world, partial_intersections, partial_agents, entity_name)
                if value:
                    local_solver.add(eval_pred(entity))
                else:
                    local_solver.add(Not(eval_pred(entity)))
        elif arity == 2:
            # Binary predicate grounding
            for entity1 in local_entities[eval_pred.domain(0).name()]:
                entity1_name = entity1.decl().name()
                for entity2 in local_entities[eval_pred.domain(1).name()]:
                    entity2_name = entity2.decl().name()
                    value = method(partial_world, partial_intersections, partial_agents, entity1_name, entity2_name)
                    if value:
                        local_solver.add(eval_pred(entity1, entity2))
                    else:
                        local_solver.add(Not(eval_pred(entity1, entity2)))

    # 5. create, ground rules and add to solver
    for rule_name, rule_template in rule_tem.items():
        # the first entity is the ego agent
        agent = local_entities[0]
        # Replace placeholder in the rule template with the actual agent entity
        instantiated_rule = eval(rule_template)
    # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
    self.rules["ClosedWorld"] = []
    for var_name, z3_var in self.z3_vars.items():
        entity_list = self.entities[var_name.replace('dummy', '')]
        constraint = Or([z3_var == entity for entity in entity_list])
        self.rules["ClosedWorld"].append(ForAll([z3_var], constraint))


def world2entity(entity_sorts, partial_intersect, partial_agents):
    assert "Agent" in entity_sorts.keys() and "Intersection" in entity_sorts.keys()
    # all the enitities are stored in self.entities
    entities = {}
    for entity_type in entity_sorts.keys():
        entities[entity_type] = []
        # For Agents
        if entity_type == "Agent":
            for agent in partial_agents:
                agent_id = agent.layer_id
                agent_type = agent.type
                agent_name = f"Agent_{agent_type}_{agent_id}"
                # Create a Z3 constant for the agent
                agent_entity = Const(agent_name, entity_sorts['Agent'])
                entities[entity_type].append(agent_entity)
        elif entity_type == "Intersection":
            # For Intersections
            unique_intersections = np.unique(partial_intersect[0])
            unique_intersections = unique_intersections[unique_intersections != 0]
            for intersection_id in unique_intersections:
                intersection_name = f"Intersection_{intersection_id}"
                # Create a Z3 constant for the intersection
                intersection_entity = Const(intersection_name, entity_sorts['Intersection'])
                entities[entity_type].append(intersection_entity)
    return entities
