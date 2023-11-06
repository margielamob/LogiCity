import torch
import pickle
from torch.utils.data.dataset import Dataset

class LogicDataset(Dataset):
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY
    def __len__(self):
        return self.dataX.shape[0]
    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]
    