import numpy as np
import gymnasium as gym
from bettermdptools.mdp import MDP

class MountainCarMDP:
    def __init__(self, position_bins=20, velocity_bins=20):
        self.env = gym.make('MountainCar-v0')
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        # TODO: Implement discretization and MDP conversion

    def discretize_state(self, state):
        pass

    def get_state_space(self):
        pass

    def get_action_space(self):
        pass
