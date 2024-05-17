import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNNFeatureExtractor(nn.Module):
    def __init__(self, observation_space, num_ents, pred_grounding_index, gnn_args):
        super(GNNFeatureExtractor, self).__init__()
        self.pred_grounding_index = pred_grounding_index
        self.num_ents = num_ents
        self.edge_index = torch.tensor([(i, j) for i in range(num_ents) for j in range(num_ents)], dtype=torch.long).t().to(device)
        self.gnn_hidden = gnn_args["gnn_hidden"]
        node_dim = gnn_args["node_dim"]
        edge_dim = gnn_args["edge_dim"]      
        self.conv = GINEConv(nn=nn.Sequential(
            nn.Linear(node_dim, self.gnn_hidden),
            nn.ReLU(),
            nn.Linear(self.gnn_hidden, self.gnn_hidden),
            nn.ReLU()), 
            edge_dim=edge_dim)

    def obs2domainArray(self, observation):
        unp_arr_ls = []
        bip_arr_ls = []
        B = observation.size(0)
        for k, v in self.pred_grounding_index.items():
            original = observation[:, v[0]:v[1]]
            if original.shape[1] == self.num_ents:
                unp_arr_ls.append(torch.tensor(original).unsqueeze(2))
            elif original.shape[1] == self.num_ents**2:
                bip_arr_ls.append(torch.tensor(original).reshape(-1, self.num_ents, self.num_ents).unsqueeze(3))
        # convert a to target
        unp_arr_ls = torch.cat(unp_arr_ls, dim=-1) # B x 5 x C_node
        bip_arr_ls = torch.cat(bip_arr_ls, dim=-1).reshape(B, self.num_ents*self.num_ents, -1) # B x (5 x 5) x C_edge
        return unp_arr_ls, bip_arr_ls

    def create_batch_graph(self, batch_nodes, batch_edges):
        data_list = []
        num_graphs = batch_nodes.size(0)
        for i in range(num_graphs):
            graph_data = Data(x=batch_nodes[i], edge_index=self.edge_index, edge_attr=batch_edges[i])
            data_list.append(graph_data)
        batch_graph = Batch.from_data_list(data_list)
        return batch_graph

    def forward(self, observations):
        batch_nodes, batch_edges = self.obs2domainArray(observations)
        batch_graph = self.create_batch_graph(batch_nodes, batch_edges)
        x, edge_index, edge_attr = batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr
        B = x.size(0) // self.num_ents
        features = self.conv(x, edge_index, edge_attr).reshape(B, self.num_ents, -1)[:, 0, :].reshape(B, -1)
        return features

class GNNPolicy(nn.Module):
    def __init__(self, gym_env, features_extractor_class, features_extractor_kwargs):
        super(GNNPolicy, self).__init__()

        self.features_extractor = features_extractor_class(gym_env.observation_space, **features_extractor_kwargs)
        # Adjust for discrete action spaces
        if isinstance(gym_env.action_space, gym.spaces.Discrete):
            n_output = gym_env.action_space.n  # Number of discrete actions
        else:
            # This is just a fallback for continuous spaces; adjust as necessary
            n_output = gym_env.action_space.shape[0]

        self.action_pred_layer = nn.Linear(self.features_extractor.gnn_hidden, n_output)
        self.value_pred_layer = nn.Linear(self.features_extractor.gnn_hidden, 1)

    def forward(self, observations):
        # Extract features
        features = self.features_extractor(observations)
        # Get the action logits
        action_logits = self.action_pred_layer(features)
        # Get the value
        values = self.value_pred_layer(features)
        return action_logits, values
