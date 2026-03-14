import random


class RandomAgent:

    def __init__(self, action_space=4):

        self.action_space = action_space

    def select_action(self):

        return random.randint(0, self.action_space - 1)
