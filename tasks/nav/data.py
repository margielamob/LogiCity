import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class NaviDatasetPOV(Dataset):
    def __init__(self, global_map, data, logger, fov, train=True, transform=None):
        self.fov = fov
        self.train = train
        self.logger = logger
        self.global_map = global_map
        self.traj = data['traj']
        self.centers = data['centers']
        self.goals = data['goals']
        self.labels = data['labels']
        self.transform = transform
        self.logger.info(f"Loaded {len(self.traj)} trajectories")
        self.logger.info(f"Loaded {len(self.goals)} state_action pairs")
        return

    def __len__(self):
        # if train return num of centers (all the pairs), else return the num as traj
        if self.train:
            return len(self.centers)
        else:
            return len(self.traj)

    def __getitem__(self, idx):
        if self.train:
            # return a pair of (img, goal)
            fov_img = np.ones((self.fov, self.fov, 3), dtype=np.uint8) * 255
            center = self.centers[idx]["center"]
            goal = self.goals[idx]["goal"]
            label = self.labels[idx]["action"]
            # get the fov image
            # Calculate the region of the city image that falls within the FOV
            x_start = max(center[0] - self.fov//2, 0)
            y_start = max(center[1] - self.fov//2, 0)
            x_end = min(center[0] + self.fov//2, self.global_map.shape[0])
            y_end = min(center[1] + self.fov//2, self.global_map.shape[1])

            # Calculate where this region should be placed in the fov_img
            new_x_start = max(self.fov//2 - center[0], 0)
            new_y_start = max(self.fov//2 - center[1], 0)
            new_x_end = new_x_start + (x_end - x_start)
            new_y_end = new_y_start + (y_end - y_start)

            fov_img[new_x_start:new_x_end, new_y_start:new_y_end] = self.global_map[x_start:x_end, y_start:y_end]
            # get the normalized goal
            goal = np.array([goal[0]-center[0], goal[1]-center[1]])
            goal = goal/(self.fov // 2)
            assert np.abs(goal).max() <= 1, f"goal is not normalized: {goal}"
            # Apply the transformation to the fov image if specified
            if self.transform:
                fov_img = self.transform(fov_img)
            else:
                # Convert image to PyTorch tensor if no other transform is applied
                fov_img = transforms.ToTensor()(fov_img)
            # Convert the numpy arrays to torch tensors
            goal = torch.from_numpy(goal).to(torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            return fov_img, goal, label
        else:
            # return a traj
            traj_id = self.traj[idx]
            traj_centers = []
            traj_goals = []
            for i in range(len(self.centers)):
                if self.centers[i]["traj_id"] == traj_id:
                    traj_centers.append(self.centers[i])
                    traj_goals.append(self.goals[i])
                    traj_labels.append(self.labels[i])
            traj_centers = sorted(traj_centers, key=lambda x: x["step"])
            traj_goals = sorted(traj_goals, key=lambda x: x["step"])
            traj_labels = sorted(traj_labels, key=lambda x: x["step"])
            # get the first step
            # return a pair of (img, goal)
            fov_img = np.ones((self.fov, self.fov, 3), dtype=np.uint8) * 255
            fov_imgs = []
            goals = []
            labels = []
            for i in range(len(traj_centers)):
                center = traj_centers[0]
                goal = traj_goals[0]
                # get the fov image
                fov_img = self.global_map[center[0]-self.fov//2:center[0]+self.fov//2, 
                                        center[1]-self.fov//2:center[1]+self.fov//2, :]
                # get the normalized goal
                goal = np.array([goal[0]-center[0], goal[1]-center[1]])
                goal = np.clip(goal, -self.fov//2, self.fov//2)
                goal = goal/(self.fov // 2)
                # Apply the transformation to the fov image if specified
                if self.transform:
                    fov_img = self.transform(fov_img)
                else:
                    # Convert image to PyTorch tensor if no other transform is applied
                    fov_img = transforms.ToTensor()(fov_img)
                # Convert the numpy arrays to torch tensors
                goal = torch.from_numpy(goal).to(torch.float32)
                label = torch.tensor(label, dtype=torch.long)
                fov_imgs.append(fov_img)
                goals.append(goal)
                labels.append(label)
            return fov_imgs, goals, labels