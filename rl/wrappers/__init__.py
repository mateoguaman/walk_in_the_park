import numpy as np
import gym
from gym.wrappers.flatten_observation import FlattenObservation

from rl.wrappers.single_precision import SinglePrecision
from rl.wrappers.universal_seed import UniversalSeed
from rl.wrappers.action_rescale import RescaleActionAsymmetric
from rl.wrappers.clip_action import ClipAction


def wrap_gym(env: gym.Env, rescale_actions: bool = True, init_qpos: np.ndarray = np.asarray([0.05, 0.9, -1.8] * 4), limit_action_range: float = 1.) -> gym.Env:
    action_qpos_min = (env.action_space.low - init_qpos) * limit_action_range + init_qpos
    action_qpos_max = (env.action_space.high - init_qpos) * limit_action_range + init_qpos
    
    env = ClipAction(env, action_qpos_min, action_qpos_max)
    env = RescaleActionAsymmetric(env, -1.0, 1.0, init_qpos)
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env