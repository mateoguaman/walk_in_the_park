from typing import Dict

import gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    episode_frames = []
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        episode_frames.append(env.render())
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
            episode_frames.append(env.render())

    return {
        'return': np.mean(env.return_queue),
        'length': np.mean(env.length_queue),
        'frames': episode_frames
    }
