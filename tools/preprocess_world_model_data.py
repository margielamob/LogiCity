import joblib
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import argparse

def CPU(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

def CUDA(x):
    return x.cuda() if isinstance(x, torch.Tensor) else x

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=str, default='expert_10k')
    parser.add_argument("--output", type=str, default='expert_all')
    
    return parser


def get_easy_obs(obs, start_pos, goal_pos):
    arr = torch.where(obs[3] == 8.0)
    cur_pos = CPU(torch.tensor([arr[0][0], arr[1][0]]))
    neighborhood_obs = obs[0:3, cur_pos[0]-1:cur_pos[0]+2, cur_pos[1]-1:cur_pos[1]+2].reshape(-1)
    
    # print(neighborhood_obs.shape, obs_dict["World"].shape)
    start_pos = np.asarray(start_pos, dtype=np.float32) / 240.
    cur_pos = np.asarray(cur_pos, dtype=np.float32) / 240.
    goal_pos = np.asarray(goal_pos, dtype=np.float32) / 240.
    obs = np.concatenate([start_pos, cur_pos, goal_pos, neighborhood_obs])
    # print(obs.shape)
    
    return obs

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    data_buffer = {"acts": [], "obs": [], "rews": [], "dones": [], "next_obs": []}
    for i in range(1, 18):
        data = joblib.load("log/{}_{}.pkl".format(args.prefix, i))
        obs = data["Time_Obs"]
        rew = np.load("log/rew__{}_{}.npy".format(args.prefix, i))
        # print(np.where(obs[4]["World"][3] == 8.0), np.where(obs[5]["World"][3] == 8.0))
        T = len(obs)
        
        arr = torch.where(obs[0]["World"][3] == 8.0)
        start_pos = CPU(torch.tensor([arr[0][0], arr[1][0]]))
        arr = torch.where(obs[T-1]["World"][3] == 8.0)
        goal_pos = CPU(torch.tensor([arr[0][0], arr[1][0]]))

        for t in range(T-1):
            data_buffer["obs"].append(get_easy_obs(obs[t]["World"], start_pos, goal_pos))
            data_buffer["acts"].append(CPU(obs[t]["Agent_actions"][0]))
            data_buffer["rews"].append(rew[t])
            if t == len(obs) - 2:
                data_buffer["dones"].append(True)
            else: 
                data_buffer["dones"].append(False)
            data_buffer["next_obs"].append(get_easy_obs(obs[t+1]["World"], start_pos, goal_pos))

        print(T, rew.shape)
    
    print("============ File Saving ===========")
    joblib.dump(data_buffer, "log/{}.pkl".format(args.output))
    print("============ File Saved ===========")

    