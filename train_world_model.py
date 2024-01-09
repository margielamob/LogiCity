import joblib

import numpy as np
import torch
import torch.nn as nn
import argparse

from utils.dataset import WMDataset

def CPU(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda()

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--state_dim", type=int, default=33)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--data_path", type=str, default='expert_all')
    
    return parser


# Neural Network for the World Model
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim):
        super(WorldModel, self).__init__()
        self.fc_hidden = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    
        self.fc_state = nn.Linear(hidden_dim, state_dim)
        self.fc_reward = nn.Linear(hidden_dim, reward_dim)
        
    def forward(self, state, action):        
        x = torch.cat([state, action], dim=-1)
        x = self.fc_hidden(x)
        state_pred = self.fc_state(x)
        reward_pred = self.fc_reward(x).softmax(dim=-1)
        
        return state_pred, reward_pred

def compute_acc(pred, label):
    pred = CPU(pred)
    label = CPU(label)
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    acc = np.sum(pred == label) / len(label)
    idx = np.where(label != 2)
    prec = np.sum(pred[idx] == label[idx]) / (len(idx[0])+1e-12)
    # print(pred, label, acc)
    return acc, prec

if __name__ == "__main__":
    args = get_parser().parse_args()
    reward_val = np.array([-1, 0, 1, 10, -10])
    reward_dim = len(reward_val)
    state_dim = args.state_dim
    action_dim = args.action_dim
    hidden_dim = args.hidden
    
    data = joblib.load("log/{}.pkl".format(args.data_path))
    model = WorldModel(state_dim, action_dim, reward_dim, hidden_dim).cuda()
    
    dataset = WMDataset(data, batch_size=64, shuffle=True)
    # split into train and test set
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    for epoch in range(1000):
        loss_train, loss_test = 0., 0.
        loss_state_train, loss_state_test = 0., 0.
        loss_pos_train, loss_pos_test = 0., 0.
        acc_train, acc_test = 0., 0.
        prec_train, prec_test = 0., 0.
        
        for batch in train_dataset:
            obs, acts, rews, dones, next_obs = batch
            obs = CUDA(torch.from_numpy(obs))
            acts = CUDA(torch.from_numpy(acts)).float()
            rews = CUDA(torch.from_numpy(rews))
            next_obs = CUDA(torch.from_numpy(next_obs))
            pred_obs, pred_rew = model(obs, acts)
            loss_rew = loss_ce(pred_rew, rews)
            loss_state = loss_mse(pred_obs, next_obs)
            loss_pos = loss_mse(pred_obs[:, 2:4], next_obs[:, 2:4])
            
            loss = loss_pos * 100 + loss_rew # loss_state + loss_rew
            acc, prec = compute_acc(pred_rew, rews)
            acc_train += acc
            prec_train += prec
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += loss.item()
            loss_state_train += loss_state.item()
            loss_pos_train += loss_pos.item()
            
            
        loss_train /= len(train_dataset)
        loss_state_train /= len(train_dataset)
        loss_pos_train /= len(train_dataset)
        acc_train /= len(train_dataset)
        prec_train /= len(train_dataset)
        print("Training: {}, Loss: {:.4f}, Loss Pos: {:.4f}, Loss State: {:.4f}, Acc: {:.4f}, Prec: {:.4f}".format(
            epoch, loss_train, loss_pos_train, loss_state_train, acc_train, prec_train))
        
        # evaluate the accuracy and loss on test set
        with torch.no_grad():
            for batch in test_dataset:
                obs, acts, rews, dones, next_obs = batch
                obs = CUDA(torch.from_numpy(obs))
                acts = CUDA(torch.from_numpy(acts)).float()
                rews = CUDA(torch.from_numpy(rews))
                next_obs = CUDA(torch.from_numpy(next_obs))
                pred_obs, pred_rew = model(obs, acts)

                loss_rew = loss_ce(pred_rew, rews)
                loss_state = loss_mse(pred_obs, next_obs)
                loss_pos = loss_mse(pred_obs[:, 2:4], next_obs[:, 2:4])
                
                loss = loss_pos * 100 + loss_rew
                acc, prec = compute_acc(pred_rew, rews)
                acc_test += acc
                prec_test += prec
                loss_test += loss.item()
                loss_state_test += loss_state.item()
                loss_pos_test += loss_pos.item()
                
        loss_test /= len(test_dataset)
        loss_state_test /= len(test_dataset)
        loss_pos_test /= len(test_dataset)

        acc_test /= len(test_dataset)
        prec_test /= len(test_dataset)
        print("Testing : {}, Loss: {:.4f}, Loss Pos: {:.4f}, Loss State: {:.4f}, Acc: {:.4f}, Prec: {:.4f}".format(
            epoch, loss_test, loss_pos_test, loss_state_test, acc_test, prec_test))
            