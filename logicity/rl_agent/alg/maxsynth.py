import re
import copy
import torch
from z3 import *
from yaml import load, FullLoader

import logging
logger = logging.getLogger(__name__)

class MaxSynth:
    def __init__(self, env, 
                 logic_engine_file
                ):
        """
        Rules from MaxSynth algorithm are used as a rule-based policy.
        """
        self.env = env
        onotology_yaml, rule_yaml = logic_engine_file['ontology'], logic_engine_file['rule']
        with open(onotology_yaml, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        with open(rule_yaml, 'r') as file:
            self.data.update(load(file, Loader=FullLoader))
        
        self._create_predicates()
        self._create_rules()
        self.pred_grounding_index = env.pred_grounding_index
        self.num_ents = env.env.rl_agent["fov_entities"]["Entity"]
        self.ego_action_mapping = env.agent.action_mapping
        self.ego_action_dist = env.agent.action_dist

    def obs2grounding(self, observation):
        # TODO: Input is a 205 dim binary vector for all ontology, convert to domainData
        grounding_dict = {}
        for k, v in self.pred_grounding_index.items():
            original = observation[v[0]:v[1]]
            grounding_dict[k] = original
        return grounding_dict

    def predict(self, observation, deterministic=True):
        # 0. get grounding dic from obs
        grounding_dict = self.obs2grounding(observation)
        # 1. create sorts and variables
        entity_sorts = DeclareSort('Entity')
        z3_vars = {var_name: Const(var_name, entity_sorts) \
                       for var_name in self.z3_vars}
        # 2. entities
        entities = [Const(f"Entity_{i}", entity_sorts) for i in range(self.num_ents)]
        # 3. create, ground predicates and add to solver
        local_predicates = copy.deepcopy(self.predicates)
        # 4. create, ground predicates and add to solver
        local_solver = Solver()
        for pred_name, pred_info in local_predicates.items():
            eval_pred = eval(pred_info["instance"])
            pred_info["instance"] = eval_pred
            arity = pred_info["arity"]
            # Import the grounding method
            method_full_name = pred_info["function"]
            if method_full_name == "None":
                continue
            assert pred_name in grounding_dict, f"Predicate {pred_name} not found in the grounding dictionary"
            if arity == 1:
                # Unary predicate grounding
                grounding = grounding_dict[pred_name]
                assert len(grounding) == self.num_ents, f"Grounding for {pred_name} has incorrect length"
                for i, entity in enumerate(entities):
                    value = grounding[i]
                    if value:
                        local_solver.add(eval_pred(entity))
                    else:
                        local_solver.add(Not(eval_pred(entity)))
            elif arity == 2:
                # Binary predicate grounding
                grounding = grounding_dict[pred_name]
                assert len(grounding) == self.num_ents ** 2, f"Grounding for {pred_name} has incorrect length"
                for i, entity1 in enumerate(entities):
                    for j, entity2 in enumerate(entities):
                        value = grounding[i * self.num_ents + j]
                        if value:
                            local_solver.add(eval_pred(entity1, entity2))
                        else:
                            local_solver.add(Not(eval_pred(entity1, entity2)))
        # 5. create, ground rules and add to solver
        local_rule_tem = copy.deepcopy(self.rule_tem)
        for rule_name, rule_template in local_rule_tem.items():
            # the first entity is the ego agent
            entity = entities[0]
            # Replace placeholder in the rule template with the actual agent entity
            instantiated_rule = eval(rule_template)
            local_solver.add(instantiated_rule)
        
        # **Important: Closed world quantifier rule, to ensure z3 do not add new entity to satisfy the rule and "dummy" is not part of the world**
        for var_name, z3_var in z3_vars.items():
            constraint = Or([z3_var == entity for entity in entities])
            local_solver.add(ForAll([z3_var], constraint))

        # 6. solve
        if local_solver.check() == sat:
            model = local_solver.model()
            # Interpret the solution to the FOL problem
            action_mapping = self.ego_action_mapping
            action_dist = torch.zeros_like(self.ego_action_dist)

            for key in local_predicates.keys():
                action = []
                for action_id, action_name in action_mapping.items():
                    if key in action_name:
                        action.append(action_id)
                if len(action)>0:
                    for a in action:
                        if is_true(model.evaluate(local_predicates[key]["instance"](entities[0]))):
                            action_dist[a] = 1.0
            # No action specified, use the default action, Normal
            if action_dist.sum() == 0:
                for action_id, action_name in action_mapping.items():
                    if "Normal" in action_name:
                        action_dist[action_id] = 1.0

        # convert to action id
        a = self.full_action2index(action_dist)
        return a, None

    def _create_rules(self):
        self.rules = {}
        self.rule_tem = {}
        for rule_dict in self.data["Rules"]:
            (rule_name, rule_info), = rule_dict.items()
            # Check if the rule is valid
            formula = rule_info["formula"]
            logger.info("Maxsynth Rule: {} -> \n {}".format(rule_name, formula))

            # Create Z3 variables based on the formula
            var_names = self._extract_variables(formula)
            self.z3_vars = var_names

            # Substitute predicate names in the formula with Z3 function instances
            for method_name, pred_info in self.predicates.items():
                formula = formula.replace(method_name, f'local_predicates["{method_name}"]["instance"]')

            # Now replace the variable names in the formula with their Z3 counterparts
            for var_name in var_names:
                formula = formula.replace(var_name, f'z3_vars["{var_name}"]')

            # Evaluate the modified formula string to create the Z3 expression
            self.rule_tem[rule_name] = formula
        rule_info = "\n".join(["- {}: {}".format(rule, details) for rule, details in self.rule_tem.items()])
        logger.info("Number of Rules: {}\nRules:\n{}".format(len(self.rule_tem), rule_info))
        logger.info("Rules will be grounded later...")

    def _extract_variables(self, formula):
        # Regular expression to find words that start with 'dummy'
        pattern = re.compile(r'\bdummy\w*\b')

        # Find all matches in the formula
        matches = pattern.findall(formula)

        # Remove duplicates by converting the list to a set, then back to a list
        unique_variables = list(set(matches))

        return unique_variables

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
                z3_func = "Function('{}', entity_sorts, BoolSort())".format(method_name)
            elif arity == 2:
                # Binary predicate
                types = info["method"].split('(')[1].split(')')[0].split(', ')
                z3_func = "Function('{}', entity_sorts, entity_sorts, BoolSort())".format(method_name)

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

    def full_action2index(self, action):
        # see agents/car.py
        if action[0] == 1:
            return 0
        elif action[4] == 1:
            return 1
        elif action[8] == 1:
            return 2
        else:
            return 3