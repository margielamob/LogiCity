import re
from z3 import *
from yaml import load, FullLoader

import logging
logger = logging.getLogger(__name__)

class MaxSynth:
    def __init__(self, env, 
                 logic_engine_file
                ):
        """
        MaxSynth algorithm for synthesizing expert demonstrations.
        """
        self.env = env
        onotology_yaml, rule_yaml = logic_engine_file['ontology'], logic_engine_file['rule']
        with open(onotology_yaml, 'r') as file:
            self.data = load(file, Loader=FullLoader)
        with open(rule_yaml, 'r') as file:
            self.data.update(load(file, Loader=FullLoader))
        
        self._create_rules()
        self.pred_grounding_index = env.pred_grounding_index
        self.num_ents = env.env.rl_agent["fov_entities"]["Entity"]

    def obs2grounding(self, observation):
        # TODO: Input is a 205 dim binary vector for all ontology, convert to domainData
        grounding_dict = {}
        for k, v in self.pred_grounding_index.items():
            original = observation[v[0]:v[1]]
            grounding_dict[k] = original
        return grounding_dict

    def predict(self, observation, deterministic=True):
    # 1. create sorts and variables
        entity_sorts = {}
        # for entity_type in entity_types:
        #     entity_sorts[entity_type] = DeclareSort(entity_type)
        # z3_vars = {var_name: Const(var_name, entity_sorts["Entity"]) \
        #                 for var_name in var_names}
        action = self.env.expert_action
        return action, None

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