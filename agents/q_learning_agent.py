import numpy as np
import random


class QLearningAgent:

    def __init__(self, grid_size, action_size=4):

        self.grid_size = grid_size
        self.action_size = action_size

        # Q-table: (x, y, action)
        self.q_table = np.zeros((grid_size, grid_size, action_size))

        # learning parameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def select_action(self, state):

        x, y = state

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return np.argmax(self.q_table[x, y])

    def update(self, state, action, reward, next_state):

        x, y = state
        nx, ny = next_state

        best_next = np.max(self.q_table[nx, ny])

        old_value = self.q_table[x, y, action]

        new_value = old_value + self.alpha * (
            reward + self.gamma * best_next - old_value
        )

        self.q_table[x, y, action] = new_value
