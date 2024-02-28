import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from logicity.rl_agent.policy import build_policy
from logicity.rl_agent.policy.hri_helper.Utils import gumbel_softmax_sample
from logicity.rl_agent.policy.hri_helper.Symbolic import extract_symbolic_path
from logicity.rl_agent.policy.hri_helper.utils.Initialise import init_aux_valuation

import logging
logger = logging.getLogger(__name__)

class HRI():
    def __init__(self, policy, env, tgt_action, default_action, \
                 threshold, action2idx, pred2ind, if_un_pred, \
                 policy_kwargs, device="cuda:0"):
        self.tgt_action = tgt_action
        self.default_action = default_action
        self.action2idx = action2idx
        self.threshold = threshold
        self.policy_class = policy
        self.policy_kwargs = policy_kwargs
        self.predicates_labels = policy_kwargs['predicates_labels']
        self.policy = build_policy[policy](env, **policy_kwargs)
        self.device = device
        self.pred2ind = pred2ind
        self.if_un_pred = if_un_pred
        self.pred_grounding_index = env.pred_grounding_index
        self.num_ents = env.env.rl_agent["fov_entities"]["Entity"]
        self.policy.to(self.device)

    def obs2domainArray(self, observation):
        # TODO: Input is a 205 dim binary vector for all ontology, convert to domainData
        # 1. convert 205 to predicate groundings
        unp_arr_ls = []
        bip_arr_ls = []
        unp_name_ls = []
        bip_name_ls = []
        for k, v in self.pred_grounding_index.items():
            original = observation[v[0]:v[1]]
            if len(original) == self.num_ents:
                if np.sum(original) > 0:
                    unp_arr_ls.append(torch.tensor(original).unsqueeze(1))
                    unp_name_ls.append(k)
            elif len(original) == self.num_ents**2:
                if np.sum(original) > 0:
                    bip_arr_ls.append(torch.tensor(original).reshape(self.num_ents, self.num_ents))
                    bip_name_ls.append(k)
        unp_ind_ls = [self.pred2ind[pn] for pn in unp_name_ls]
        bip_ind_ls = [self.pred2ind[pn] for pn in bip_name_ls]
        valuation_init = [Variable(arr) for arr in unp_arr_ls] + [Variable(arr) for arr in bip_arr_ls]
        pred_ind_ls = unp_ind_ls + bip_ind_ls
        return valuation_init, pred_ind_ls, self.num_ents
    
    def predict(self, observation, deterministic=False):
        valuation_eval_temp, bg_pred_ind_ls_noTF, num_constants = self.obs2domainArray(observation)
        valuation_eval = [torch.zeros(num_constants).view(-1, 1) if tp else torch.zeros(
            (num_constants, num_constants)) for tp in self.if_un_pred]
        for idx, idp in enumerate(bg_pred_ind_ls_noTF):
            assert valuation_eval[idp].shape == valuation_eval_temp[idx].shape
            valuation_eval[idp] = valuation_eval_temp[idx]
        assert max(bg_pred_ind_ls_noTF) < self.policy.num_background - 2
        # ----2--add valuation other aux predicates
        valuation_eval = init_aux_valuation(self.policy, valuation_eval, num_constants, steps=4)
        # --3---inference steps
        valuation_eval = valuation_eval.cuda()
        
        valuation_eval, valuation_tgt = self.policy.infer(
            valuation_eval, num_constants, unifs=self.unifs, steps=4)
        
        action = self.get_action(valuation_tgt)
        
        return action, None
    
    def get_action(self, valuation_tgt):
        prob = valuation_tgt[0]
        if prob > self.threshold:
            action_id = self.action2idx[self.tgt_action]
        else:
            action_id = self.action2idx[self.default_action]
        return action_id

    def load(
        self,
        path
    ):
        self.policy.load_state_dict(torch.load(path))
        # soft model
        unifs = self.get_unifs(temperature=0.01, gumbel_noise=0.)
        self.unifs = unifs.view(self.policy.num_predicates, self.policy.num_body, self.policy.num_rules)
        # symbolic model
        full_rules_str=self.policy.rules_str
        #--3-extract symbolic path
        #TODO: More efficient unification with symbolic rule instead of this.
        _, symbolic_formula, symbolic_unifs, _ = extract_symbolic_path(self.unifs, full_rules_str, predicates_labels=self.predicates_labels)
        symbolic_unifs=symbolic_unifs.double() #1 where max value, else 0
        assert list(symbolic_unifs.size())==[self.policy.num_predicates, self.policy.num_body, self.policy.num_rules]
        self.symbolic_unifs = symbolic_unifs
        logger.info(f"Symbolic model (Argmax backtracked path formulae): {symbolic_formula}")

    def get_unifs(self, temperature, gumbel_noise):
        """
        Compute unifications score of (soft) rules with embeddings (all:background+ symbolic+soft)

        Inputs:
            embeddings: tensor size (p,f) where p number of predicates, f is feature dim, (and currently p=f)
            rules: tensor size (r, num_body*f) where f is feature dim,  r nb rules, and num_body may be 2 or 3 depending on model considered
        Outputs:        
        """
        # -- 00---init
        rules = self.policy.rules.clone()
        embeddings = self.policy.embeddings.clone()
        num_rules, d= rules.shape
        num_predicates, num_feat= embeddings.shape
            
        # 0: ---add True and False
        num_predicates += 2
        num_feat += 2
        # NOTE: add False which is (0,1,0...) in second position
        row = torch.zeros(1, embeddings.size(1))
        col = torch.zeros(embeddings.size(0)+1, 1)
        col[0][0] = 1
        row = row.cuda()
        col = col.cuda()
        embeddings = torch.cat((row, embeddings), 0)
        embeddings = torch.cat((col, embeddings), 1)
        # NOTE: add True which is (1,0,...) in first position
        row_F = torch.zeros(1, embeddings.size(1))
        col_F = torch.zeros(embeddings.size(0)+1, 1)
        col_F[0][0]=1
        row_F = row_F.cuda()
        col_F = col_F.cuda()
        embeddings = torch.cat((row_F, embeddings), 0)
        embeddings = torch.cat((col_F, embeddings), 1)
        
        assert d % num_feat == 0
        num_body=d//num_feat

        # -- 1---prepare rules and embedding in good format for computation below
        if num_body == 2:
            rules_aux = torch.cat(
                (rules[:, : num_feat],
                    rules[:, num_feat: 2 * num_feat]),
                0)
        elif num_body == 3:
            rules_aux = torch.cat(
                (rules[:, : num_feat],
                    rules[:, num_feat: 2 * num_feat],
                    rules[:, 2 * num_feat: 3 * num_feat]),
                0)
        else:
            raise NotImplementedError

        rules_aux = rules_aux.repeat(num_predicates, 1)#size (p*3*r, f)
        
        embeddings_aux = embeddings.repeat(1, num_rules * num_body).view(-1, num_feat)
        # -2-- compute similarity score between predicates and rules body
        sim = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates, num_body, num_rules)

        self.policy.hierarchical_mask = self.policy.hierarchical_mask.double()
        sim[self.policy.hierarchical_mask==0] = -10000
        cancel_out = -10000
        #-5-----tgt mask: other rule body not being matched to tgt
        sim[-1,:,:]= cancel_out*torch.ones(num_body, num_rules)

        # --3-- possibly apply softmax to normalise or gumbel softmax to normalise + explore
        unifs = gumbel_softmax_sample(
            sim, temperature, gumbel_noise, use_gpu=True).view(-1)

        if not ((torch.max(unifs.view(-1)).item()<=1) and (torch.min(unifs.view(-1)).item()>=0)):
            print("ERROR UNIFS not in BOUNDARY", torch.max(unifs.view(-1)).item(), torch.min(unifs.view(-1)).item())
            pass

        return unifs.view(num_predicates, -1)