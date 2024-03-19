import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from logicity.rl_agent.policy import build_policy

import logging
logger = logging.getLogger(__name__)

class NLM():
    def __init__(self, policy, env, index2action, \
                 action2idx, policy_kwargs, device="cuda:0"):
        self.policy_class = policy
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.policy = build_policy[policy](env, **policy_kwargs)
        self.pred_grounding_index = env.pred_grounding_index
        self.num_ents = env.env.rl_agent["fov_entities"]["Entity"]
        self.index2action = index2action
        self.action2idx = action2idx

    def obs2domainArray(self, observation):
        # TODO: Input is a 205 dim binary vector for all ontology, convert to domainData
        unp_arr_ls = []
        bip_arr_ls = []
        for k, v in self.pred_grounding_index.items():
            original = observation[v[0]:v[1]]
            if original.shape[0] == self.num_ents:
                unp_arr_ls.append(torch.tensor(original).unsqueeze(1))
            elif original.shape[0] == self.num_ents**2:
                bip_arr_ls.append(torch.tensor(original).reshape(self.num_ents, self.num_ents).unsqueeze(2))
        # convert a to target
        unp_arr_ls = torch.cat(unp_arr_ls, dim=1).unsqueeze(0)
        bip_arr_ls = torch.cat(bip_arr_ls, dim=2).unsqueeze(0)
        return dict(n=torch.tensor([self.num_ents]), states=unp_arr_ls, relations=bip_arr_ls)
    
    def predict(self, observation, deterministic=False):
        feed_dict = self.obs2domainArray(observation)
        action_prob = self.policy(feed_dict)
        action = self.get_action(action_prob)
        return action, None
    
    def get_action(self, action_prob):
        max_idx = np.argmax(action_prob)
        max_action = self.index2action[max_idx]
        return self.action2idx[max_action]

    def load(
        self,
        path
    ):
        self.policy.load_state_dict(torch.load(path)['model'])
        self.policy.eval()