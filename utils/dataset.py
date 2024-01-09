import numpy as np
import torch

# data loader class for data buffer inherited from torch.utils.data.Dataset
class WMDataset(torch.utils.data.Dataset):
    def __init__(self, data_buffer, batch_size=256, shuffle=True):
        self.data_buffer = data_buffer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(self.data_buffer["obs"])
        self.indices = np.arange(self.data_size)
        self.num_batches = self.data_size // self.batch_size
        self.batch_idx = 0
        self.reward_val = np.array([-1, 0, 1, 10, -10])
        self.reset()
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        obs = np.stack(self.data_buffer["obs"])[idx]
        acts = np.stack(self.data_buffer["acts"])[idx]
        # rews = np.array([np.where(self.reward_val == self.data_buffer["rews"][i]) for i in idx], dtype=np.int32).reshape(-1, 1)
        # transform rews to one-hot vector
        rews = np.zeros((len(idx), len(self.reward_val)))
        # print([self.data_buffer["rews"][i] for i in idx])
        try: 
            rews[np.arange(len(idx)), np.array([np.where(self.reward_val == self.data_buffer["rews"][i]) for i in idx]).reshape(-1)] = 1
        except: 
            pass
            # print(len([self.data_buffer["rews"][i] for i in idx]))
            # input()
        # rews = np.array(np.stack(self.data_buffer["rews"])[idx])
        dones = np.array(np.stack(self.data_buffer["dones"])[idx], dtype=np.int32).reshape(-1, 1)
        next_obs = np.stack(self.data_buffer["next_obs"])[idx]
        return obs, acts, rews, dones, next_obs
    
    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch_idx < self.num_batches:
            batch = self.__getitem__(self.batch_idx)
            self.batch_idx += 1
            return batch
        else:
            self.reset()
            raise StopIteration

if __name__ == '__main__': 
    import joblib
    data = joblib.load("log/expert_all.pkl")
    
    dataset = WMDataset(data)
    for obs, acts, rews, dones, next_obs in dataset:
        print(obs.shape, acts.shape, rews.shape, dones.shape, next_obs.shape)
        print(rews[0])
        break