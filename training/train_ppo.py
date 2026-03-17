from env.grid_world import GridWorld
from agents.ppo_agent import PPOAgent
from utils.visualize import show_grid

import numpy as np


env = GridWorld(size=5)

state_size = 2
action_size = 4

agent = PPOAgent(state_size, action_size)

episodes = 1000


for ep in range(episodes):

    state = env.reset()

    done = False

    log_probs = []
    values = []
    rewards = []

    total_reward = 0

    while not done:

        action, log_prob, value = agent.select_action(state)

        next_state, reward, done = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = next_state
        total_reward += reward

        

    agent.update(log_probs, values, rewards)

    print("Episode:", ep, "Reward:", total_reward)