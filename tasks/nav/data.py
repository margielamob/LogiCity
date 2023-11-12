import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tasks.nav.pkl_parser import get_fov

class NaviDatasetPOV(Dataset):
    def __init__(self, global_map, data, logger, fov, train=True, transform=None):
        self.fov = fov
        self.train = train
        self.logger = logger
        self.global_map = global_map
        self.traj = data['traj']
        self.centers = data['centers']
        self.local_goals = data['local_goals']
        self.final_goals = data['final_goals']
        self.labels = data['labels']
        self.transform = transform
        self.logger.info(f"Loaded {len(self.traj)} trajectories")
        self.logger.info(f"Loaded {len(self.local_goals)} state_action pairs")
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
            local_goals = self.local_goals[idx]["goal"]
            label = self.labels[idx]["action"]
            # get the fov image
            # Calculate the region of the city image that falls within the FOV
            fov_img = get_fov(self.global_map, center, self.fov)
            # get the normalized goal
            local_goals = np.array([local_goals[0]-center[0], local_goals[1]-center[1]])
            local_goals = local_goals/(self.fov // 2)
            assert np.abs(local_goals).max() <= 1, f"goal is not normalized: {local_goals}"
            # Apply the transformation to the fov image if specified
            if self.transform:
                fov_img = self.transform(fov_img)
            else:
                # Convert image to PyTorch tensor if no other transform is applied
                fov_img = transforms.ToTensor()(fov_img)
            # Convert the numpy arrays to torch tensors
            local_goals = torch.from_numpy(local_goals).to(torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            return fov_img, local_goals, label
        else:
            # return a traj
            traj_id = self.traj[idx]
            traj_centers = []
            traj_goals = []
            traj_labels = []
            traj_final_goal = []
            for i in range(len(self.centers)):
                if self.centers[i]["traj_id"] == traj_id:
                    traj_centers.append(self.centers[i])
                    traj_goals.append(self.local_goals[i])
                    traj_labels.append(self.labels[i])
                    traj_final_goal.append(self.final_goals[i])
            traj_centers = sorted(traj_centers, key=lambda x: x["step"])
            traj_goals = sorted(traj_goals, key=lambda x: x["step"])
            traj_labels = sorted(traj_labels, key=lambda x: x["step"])
            traj_final_goal = sorted(traj_final_goal, key=lambda x: x["step"])
            # get the first step
            # return a pair of (img, goal)
            fov_imgs = []
            goals = []
            labels = []
            for i in range(len(traj_centers)):
                center = traj_centers[i]['center']
                goal = traj_goals[i]["goal"]
                label = traj_labels[i]["action"]
                fov_img = get_fov(self.global_map, center, self.fov)
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
                fov_imgs.append(fov_img)
                goals.append(goal)
                labels.append(label)
            final_goal = torch.from_numpy(traj_final_goal[-1]["goal"]).to(torch.float32)
            init_center = torch.from_numpy(traj_centers[0]["center"]).to(torch.float32)
            return fov_imgs, goals, labels, init_center, final_goal