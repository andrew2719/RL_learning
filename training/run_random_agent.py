from env.grid_world import GridWorld
from agents.random_agent import RandomAgent
from utils.visualize import show_grid


env = GridWorld(size=5)

agent = RandomAgent()


episodes = 10

for ep in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0

    while not done:

        action = agent.select_action()

        next_state, reward, done = env.step(action)

        state = next_state
        total_reward += reward

        show_grid(state)

    print("Episode:", ep, "Reward:", total_reward)
