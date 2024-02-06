import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

class BCAgent:
    def __init__(self, observation_space, action_space, model_architecture='mlp', learning_rate=1e-3):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self._build_model(model_architecture)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _build_model(self, architecture):
        # Example for MLP architecture
        if architecture == 'mlp':
            model = nn.Sequential(
                nn.Linear(self.observation_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_space.n)  # Assuming discrete action space
            )
        else:
            raise NotImplementedError(f"Architecture '{architecture}' is not implemented.")
        return model
    
    def learn(self, demonstrations, batch_size=64, epochs=10):
        # Convert demonstrations to PyTorch datasets
        states, actions = zip(*demonstrations)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        dataset = TensorDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                action_logits = self.model(batch_states)
                loss = nn.CrossEntropyLoss()(action_logits, batch_actions)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    def predict(self, observation, deterministic=True):
        self.model.eval()
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32)
            action_logits = self.model(observation)
            if deterministic:
                return torch.argmax(action_logits).item(), None
            else:
                # Implement stochastic action selection if needed
                pass
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
