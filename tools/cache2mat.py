"""
Convert the cahce file to the format expected by symbolic (decision tree) learning
"""
from tqdm import tqdm
import numpy as np
import torch
import pickle as pkl

def main(pkl_path, out_path):
    with open(pkl_path, "rb") as f:
        cached_observation = pkl.load(f)

    symbolic_observation = {}
    for agent_id in tqdm(range(len(cached_observation["Time_Obs"][1]["Agent_actions"]))):
        Xs = []
        Ys = []
        for t in tqdm(range(1, len(cached_observation["Time_Obs"])+1)):
            preds = cached_observation["Time_Obs"][t]["LNN_state"][agent_id]
            acts = cached_observation["Time_Obs"][t]["Agent_actions"][agent_id]
            Xs.append(preds)
            Ys.append(acts)

        Xs = torch.stack(Xs, dim=0)
        Ys = torch.stack(Ys, dim=0)
        symbolic_observation[agent_id] = {"Xs": Xs, "Ys": Ys}

    with open(out_path, "wb") as f:
        pkl.dump(symbolic_observation, f)


if __name__ == '__main__':
    pkl_path = "log/train_100.pkl"
    out_path = "log/train_100_symbolic.pkl"
    main(pkl_path, out_path)