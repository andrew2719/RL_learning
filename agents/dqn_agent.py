import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:

    def __init__(self, grid_size, action_size=4):

        self.grid_size = grid_size
        self.state_size = 2
        self.action_size = action_size

        self.gamma = 0.9
        self.epsilon = 0.2
        self.lr = 0.001

        self.network = QNetwork(self.state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.tensor(state, dtype=torch.float32)

        q_values = self.network(state)

        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.network(state)

        next_q_values = self.network(next_state)

        target = q_values.clone().detach()

        target[action] = reward + self.gamma * torch.max(next_q_values)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
