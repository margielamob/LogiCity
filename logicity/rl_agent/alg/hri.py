import torch
import torch.nn.functional as F
from logicity.rl_agent.policy import build_policy
from logicity.rl_agent.policy.hri_helper.Utils import gumbel_softmax_sample
from logicity.rl_agent.policy.hri_helper.Symbolic import extract_symbolic_path

import logging
logger = logging.getLogger(__name__)

class HRI():
    def __init__(self, policy, env, policy_kwargs, device="cuda:0"):
        self.policy_class = policy
        self.policy_kwargs = policy_kwargs
        self.predicates_labels = policy_kwargs['predicates_labels']
        self.policy = build_policy[policy](env, **policy_kwargs)
        self.device = device
        self.policy.to(self.device)

    
    def predict(self, observation, deterministic=False):
        if self.policy.training:
            self.policy.eval()
        observation = torch.tensor(observation).to(self.device).float()
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, _ = self.policy(observation)
            action = F.softmax(action_logits, dim=-1)
            action = action.argmax(dim=-1)
        
        return action.cpu().numpy(), None
    
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