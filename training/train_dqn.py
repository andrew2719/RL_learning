from env.grid_world import GridWorld
from agents.dqn_agent import DQNAgent


env = GridWorld(size=5)

agent = DQNAgent(grid_size=5)

episodes = 1000

for ep in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0

    while not done:

        action = agent.select_action(state)

        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    print("Episode:", ep, "Reward:", total_reward)
