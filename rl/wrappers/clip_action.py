import gym
import numpy as np
from gym.spaces import Box, Dict

class ClipAction(gym.ActionWrapper):

    def __init__(self, env, min_action, max_action):
        super().__init__(env)

        min_action = np.asarray(min_action)
        max_action = np.asarray(max_action)

        min_action = min_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        max_action = max_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        min_action = np.maximum(min_action, env.action_space.low)
        max_action = np.minimum(max_action, env.action_space.high)

        self.action_space = Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)