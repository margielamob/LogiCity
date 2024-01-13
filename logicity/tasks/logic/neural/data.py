import torch
import numpy as np
from torch.utils.data import Dataset
from prettytable import PrettyTable

class LogicDataset(Dataset):
    def __init__(self, dataX, dataY, Xname, Yname, logger, test=False, uni_boundary=0.5, w_bernoulli = False, irr_c=0):
        self.dataX = dataX
        self.dataY = dataY
        self.Yname = Yname  # Store the Yname
        self.logger = logger
        self.test = test
        self.log_distribution("Dataset distribution")
        self.add_noise(uni_boundary, w_bernoulli, irr_c)
        logger.info(f"Sample from bernoulli: {w_bernoulli}")

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

    def add_noise(self, uni_boundary, w_bernoulli=False, irr_c=0):
        # Adding irr concepts
        if irr_c > 0:
            noise_irr_c = torch.rand((self.dataX.shape[0], irr_c))
            bernoulli_c = torch.bernoulli(noise_irr_c)
            self.dataX = torch.cat((self.dataX, bernoulli_c), dim=1)
        # Adding uniform noise
        noise_0 = torch.rand(self.dataX.size()) * uni_boundary
        noise_1 = 1 - torch.rand(self.dataX.size()) * uni_boundary
        noisy_tensor = torch.where(self.dataX == 0, noise_0, noise_1)
        if not w_bernoulli:
            self.dataX = noisy_tensor
            return
        # Adding bernoulli noise
        else:
            if self.test:
                return
            bernoulli_tensor = torch.bernoulli(noisy_tensor)
            self.dataX = bernoulli_tensor
            return

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]

class LogicDatasetSAT(LogicDataset):
    def __init__(self, dataX, dataY, Xname, Yname, logger, test=False, uni_boundary=0.5, w_bernoulli = False, irr_c=0):
        super().__init__(dataX, dataY, Xname, Yname, logger, test=False, uni_boundary=uni_boundary, w_bernoulli = w_bernoulli, irr_c=irr_c)
        self.dataY = torch.tensor(np.where(dataY == 0.5, 0, 1)).float()
        self.input_data = torch.cat((self.dataX, torch.zeros_like(self.dataY)), 1)
        mask_list = [1]*self.dataX.shape[1] + [0]*self.dataY.shape[1]
        self.input_mask = torch.IntTensor(mask_list).repeat(self.dataX.shape[0], 1)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx], self.input_mask[idx], self.dataY[idx]