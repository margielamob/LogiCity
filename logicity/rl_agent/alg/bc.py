import torch
import torch.nn as nn
import torch.optim as optim

class BCAgent:
    def __init__(self, policy_class, policy_kwargs, learning_rate=1e-3):
        self.policy = policy_class(**policy_kwargs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Customize based on action space

    def train(self, training_data, epochs=10, batch_size=64):
        dataset = torch.utils.data.TensorDataset(torch.tensor(training_data['observations'], dtype=torch.float32),
                                                 torch.tensor(training_data['actions'], dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for observations, actions in dataloader:
                predicted_actions = self.model(observations)
                loss = self.loss_fn(predicted_actions, actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def predict(self, observation):
        self.model.eval()
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32)
            action = self.model(observation)
        return action.numpy()
