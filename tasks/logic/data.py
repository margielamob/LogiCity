import torch
import pickle
from torch.utils.data.dataset import Dataset

class LogicDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.data = torch.load(data_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    