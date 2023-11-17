import torch
import numpy as np
from torch.utils.data import Dataset
from prettytable import PrettyTable

class LogicDataset(Dataset):
    def __init__(self, dataX, dataY, Xname, Yname, logger, noise_std = 0.0):
        self.dataX = dataX
        self.dataY = dataY
        self.Yname = Yname  # Store the Yname
        self.logger = logger
        self.log_distribution("Dataset distribution")
        self.add_gaussian_noise(noise_std)
        logger.info(f"Noise std: {noise_std}")

    def log_distribution(self, message):
        unique_rows, counts = np.unique(self.dataY, axis=0, return_counts=True)

        # Create a PrettyTable instance
        table = PrettyTable()
        # Use Yname for the field names
        table.field_names = self.Yname + ["Count"]
        table.title = message

        # Populate the table with data
        for row, count in zip(unique_rows, counts):
            # Convert row to labels using Yname
            label_row = [self.Yname[i] if x == 1 else "" for i, x in enumerate(row)]
            table.add_row(label_row + [count])

        # Sort the table by the 'Count' column
        table.sortby = "Count"
        table.reversesort = True

        # Log the table using the provided logger
        self.logger.info(f"\n{table}")

    def add_gaussian_noise(self, std_dev=0.0):
        # Adding Gaussian noise
        noise = torch.randn(self.dataX.size()) * std_dev
        noisy_tensor = self.dataX.to(torch.float32) + noise
        # Clamping to ensure values are within 0 and 1
        self.dataX = noisy_tensor.clamp(0, 1)

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]

class LogicDatasetSAT(LogicDataset):
    def __init__(self, dataX, dataY, Xname, Yname, logger, adjust=False):
        super().__init__(dataX, dataY, Xname, Yname, logger, adjust)
        self.dataY = torch.tensor(np.where(dataY == 0.5, 0, 1)).float()
        self.input_data = torch.cat((self.dataX, torch.zeros_like(self.dataY)), 1)
        mask_list = [1]*self.dataX.shape[1] + [0]*self.dataY.shape[1]
        self.input_mask = torch.IntTensor(mask_list).repeat(self.dataX.shape[0], 1)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx], self.input_mask[idx], self.dataY[idx]