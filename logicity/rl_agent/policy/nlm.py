import torch
import torch.nn as nn
import torch.nn.functional as F

from .nlm_helper.nn.neural_logic import LogicMachine, LogitsInference

class NLMPolicy(nn.Module):
  """The model for family tree or general graphs path tasks."""

  def __init__(self, env, tgt_arity, nlm_args, \
               target_dim):
    super().__init__()
    # inputs
    self.feature_axis = tgt_arity
    self.nlm_args = nlm_args
    self.features = LogicMachine(**nlm_args)
    output_dim = self.features.output_dims[self.feature_axis]
    # Do not sigmoid as we will use CrossEntropyLoss
    self.pred = LogitsInference(output_dim, target_dim, [])

  def forward(self, feed_dict):
    # import ipdb; ipdb.set_trace()

    # relations
    states = feed_dict['states']
    relations = feed_dict['relations']
    batch_size, nr = relations.size()[:2]

    inp = [None for _ in range(self.nlm_args['breadth'] + 1)]
    # import ipdb; ipdb.set_trace()
    inp[1] = states
    inp[2] = relations
    depth = None
    feature = self.features(inp, depth=depth)[self.feature_axis]

    # import ipdb; ipdb.set_trace()
    pred = self.pred(feature)
    pred = pred[:, 0]
    pred = F.softmax(pred, dim=-1)
    return pred