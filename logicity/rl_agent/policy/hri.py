import os
import torch
import torch.nn as nn
#from line_profiler import LineProfiler

from .hri_helper.Infer import infer_tgt_vectorise, infer_one_step_vectorise_neo
from .hri_helper.utils.Initialise import init_rules_embeddings, init_predicates_embeddings_plain, init_aux_valuation, init_rule_templates
from .hri_helper.utils.Masks import init_mask, get_hierarchical_mask

# ------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class HriPolicy(nn.Module):

    def __init__(self, env,
                 num_background,
                 num_features,
                 max_depth=None,
                 tgt_arity=None,
                 predicates_labels=None,
                 ):
        """
            Main model
            
            Args:
                num_background: num background predicates
                num_features: num features, by default would be set to number of predicates.
                max_depth: needed for unified + hierarchical model
                tgt_arity: needed for unified models notably
                depth_aux_predicates  # for model hierarchical, say depth for each aux predicate
                recursive_predicates for model_hierarchical if template is campero; by default extended rules are recursive for other model
                predicates_labels For visualisation, else would be automatically generated
                rules_str: here give the list of templates we want to use, if do not use unified templates
                templates_set: for unified model, this would serve to construct all rule templates used.
                    It is a dictionary, per arity. And in case of hierarchical model, it is a base set which would be added in each depth too. 
                pred_two_rules: if use campero templates, beware, some predicates may have two rules
        """
        super().__init__()
        
        # 1----parameters given as input
        # TODO: Put the parameters of the model below in a config file

        self.predicates_labels = predicates_labels
        self.num_feat = num_features
        self.num_background = num_background
        self.max_depth = max_depth
        self.tgt_arity = tgt_arity
        self.templates_set = {'unary': ['A00+'], 'binary': ['C00+', 'B00+', 'Inv']}

        # NOTE: adding p0, then True & False as special bg predicates
        self.num_background += 2

        # 2----initialisation further parameters
        self.initialise()
        print(
            "Initialised model. num features {} num predicates {} aux pred idx {} predicates to rule{} rule str {}".format(
                self.num_feat, self.num_predicates, self.idx_aux, self.PREDICATES_TO_RULES, self.rules_str
            ))

        # 3---- Init Parameters to learn for both embeddings and rules
        init_predicates = init_predicates_embeddings_plain(self)
        init_body = init_rules_embeddings(self)
        self.rules = nn.Parameter(init_body, requires_grad=True)
        self.embeddings = nn.Parameter(init_predicates, requires_grad=True)

        # NOTE: TEMPORARY here to later merge w/ Progressive Model and use same procedure inference
        self.num_soft_predicates = self.num_rules
        self.idx_soft_predicates = self.idx_aux
        # here no other symb predicate
        self.num_all_symbolic_predicates = self.num_background


    def initialise(self):
        """
        Initialise some model parameters.
        """

        # --1--default param:
        self.num_body = 3
        self.two_rules = False  # default one predicate one rule
        # ----2-create template and number rules etc
        self.idx_background, self.idx_aux, self.rules_str, \
        self.predicates_labels, self.rules_arity, self.depth_predicates = init_rule_templates(num_background=self.num_background,
                                                                                            max_depth=self.max_depth,
                                                                                            tgt_arity=self.tgt_arity,
                                                                                            templates_unary=self.templates_set["unary"],
                                                                                            templates_binary=self.templates_set["binary"],
                                                                                            predicates_labels=self.predicates_labels
                                                                                            )
        # Add one to tgt depth if unified and hierarchical
        self.depth_predicates[-1] = self.depth_predicates[-2]+1
        self.num_aux = len(self.idx_aux)  # include tgt
        self.num_rules = self.num_aux

        if not self.two_rules:  # 1 o 1 mapping
            self.RULES_TO_PREDICATES = [i for i in range(self.num_rules)]
            self.PREDICATES_TO_RULES = [[i] for i in range(self.num_rules)]

        # NOTE: here, T & F are added in num_backgroud
        self.num_predicates = self.num_background + self.num_aux
        # ---mask hierarchical for unifications
        init_mask(self)

        self.hierarchical_mask = get_hierarchical_mask(
            self.depth_predicates, self.num_rules, self.num_predicates, self.num_body, self.rules_str, recursivity='none')

    def infer(self, valuation, num_constants, unifs, steps=1, permute_masks=None, task_idx=None, num_predicates=None, numFixedVal=None):
        """
        Main inference procedure, running for a certain number of steps.

        Args:
            valuation: valuations of the predicates, being updated, tensor of shape (num_predicates, num_constants, num_constants) 
            num_constants: int, number constant considered
            unifs: unification scores, tensor of shape (num_predicates, num_body, num_rules)
            steps: int. number inference steps
            permute_masks: float mask with 0 and 1, in case use permutation parameters, to know for which rule we have to permute the first resp. the second body.
            task_idx: int, index of the task considered.
            num_predicates: int, number predicates.
            numFixedVal: int, num predicates whose valuation is unchanged (e.g. initial predicates, True, False)

        Output:
            valuation: valuations of the predicates, being updated, tensor of shape (num_predicates, num_constants, num_constants) 
            valuation_tgt: valuation of the target predicate, tensor of shape valuations of the predicates, being updated, tensor of shape (num_constants, num_constants) 

        """
        if num_predicates is None:
            num_predicates = self.num_predicates
        # -1--preparation if vectorise as tensor size (pred, num cst, num_csts)
        if self.args.vectorise:
            unifs_ = unifs.view(num_predicates, self.num_body, self.num_rules)
            # shape p-1, p-1, r-1 -- removed tgt from predicates
            unifs_duo = torch.einsum(
                'pr,qr->pqr', unifs_[:-1, 0, :-1], unifs_[:-1, 1, :-1]).view(-1, self.num_rules-1)
            if self.args.normalise_unifs_duo:  # TODO: Which unifs ?
                unifs_duo = unifs_duo / \
                    torch.sum(unifs_duo, keepdim=True, dim=0)[0]
            unifs_duo = unifs_duo.view(
                num_predicates-1, num_predicates-1, self.num_rules-1)

        # 2----run inference steps, depending on template chosen
        for step in range(steps):
            valuation = infer_one_step_vectorise_neo(self, valuation, num_constants, unifs_, unifs_duo,
                                                             num_predicates=num_predicates, numFixedVal=numFixedVal)

        # 3---- tgt valuation computed at very end here
        valuation_tgt = infer_tgt_vectorise(self.args, valuation, unifs.view(
                num_predicates, self.num_body, self.num_rules), tgt_arity=self.rules_arity[-1])

        return valuation, valuation_tgt