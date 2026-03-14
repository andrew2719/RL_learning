from env.grid_world import GridWorld
from agents.q_learning_agent import QLearningAgent
from utils.visualize import show_grid


env = GridWorld(size=5)

agent = QLearningAgent(grid_size=5)


episodes = 5000

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
        # show_grid(env.render())

    print("Episode:", ep, "Reward:", total_reward)

print(agent.q_table)
