import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from logicity.rl_agent.policy.dreamer_helper.utils.module import get_parameters, FreezeParameters
from logicity.rl_agent.policy.dreamer_helper.utils.algorithm import compute_return

from logicity.rl_agent.policy.dreamer_helper.models.actor import DiscreteActionModel
from logicity.rl_agent.policy.dreamer_helper.models.dense import DenseModel
from logicity.rl_agent.policy.dreamer_helper.models.rssm import RSSM
from logicity.rl_agent.policy.dreamer_helper.models.pixel import ObsDecoder, ObsEncoder
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from gymnasium import spaces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DreamerPolicy(BasePolicy):

    def __init__(
        self,
        env,
        config,
        device,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        # init vars
        # MPC policy needs env.get_reward for planning
        self.env = env
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.config = config
        self._model_initialize(config, device)
        self._optim_initialize(config)

    def _model_initialize(self, config, device):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size 
    
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, device, config.rssm_type, config.rssm_info).to(self.device)
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(device)
        self.ValueModel = DenseModel((1,), modelstate_size, config.critic).to(device)
        self.TargetValueModel = DenseModel((1,), modelstate_size, config.critic).to(device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        
        if config.discount['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(device)
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = self.optimizer_class(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = self.optimizer_class(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = self.optimizer_class(get_parameters(self.value_list), value_lr)

        self.kl_info = config.kl
        self.lambda_ = config.lambda_
        self.discount = config.discount_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))    
        _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, True, prev_rssmstate)
        model_state = self.RSSM.get_model_state(posterior_rssm_state)
        action, _ = self.ActionModel(model_state)
        prev_rssmstate = posterior_rssm_state
        prev_action = action

        return action

    def actorcritc_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        
        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]+[self.DiscountModel]):
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)     

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
        }

        return actor_loss, value_loss, target_info

    def representation_loss(self, obs, actions, rewards, nonterms, reward_weighing=False):

        embed = self.ObsEncoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)   
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardDecoder(post_modelstate[:-1])               #t to t+seq_len-1  
        pcont_dist = self.DiscountModel(post_modelstate[:-1])                #t to t+seq_len-1   
        
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:], reward_weighing)
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount']*pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)
        
        if self.config.actor_grad == 'reinforce':
            advantage = (lambda_returns-imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.ValueModel(value_modelstates) 
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss
            
    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards, reward_weighing):
        if reward_weighing:
            weights = self.calculate_weights(rewards)
            reward_loss = -torch.mean(reward_dist.log_prob(rewards) * weights.squeeze())
        else:
            reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def calculate_weights(self, rewards, scale=3):
        # Weights increase as rewards decrease. The scale factor adjusts sensitivity.
        weights = torch.exp(-scale * rewards)
        return weights / weights.mean()