import os
import sys
import gym

from dm_control import suite
from dm_control import viewer
from dm_control import _render
from absl import app, flags
from ml_collections import config_flags
from dm_control import composer

from env_utils import make_mujoco_env
from flax.training import checkpoints
from rl.agents import SACLearner
from rl.wrappers import wrap_gym
from sim.robots import A1
from sim.tasks import Run
from functools import partial
from gym.wrappers.flatten_observation import FlattenObservation





import copy
from typing import OrderedDict

import dm_env
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low,
                          high=high,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float('-inf')
            high = float('inf')
        else:
            raise ValueError()

        return spaces.Box(low=low,
                          high=high,
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


def dmc_obs2gym_obs(obs):
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dmc_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)
    
def _convert_space(obs_space):
    if isinstance(obs_space, Box):
        obs_space = Box(obs_space.low, obs_space.high, obs_space.shape)
    elif isinstance(obs_space, Dict):
        for k, v in obs_space.spaces.items():
            obs_space.spaces[k] = _convert_space(v)
        obs_space = Dict(obs_space.spaces)
    else:
        raise NotImplementedError
    return obs_space


def _convert_obs(obs):
    if isinstance(obs, np.ndarray):
        if obs.dtype == np.float64:
            return obs.astype(np.float32)
        else:
            return obs
    elif isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = _convert_obs(v)
        return obs
    


def sac_policy(agent, obs_space, action_space, time_step):
    # import pdb;pdb.set_trace()
    obs = time_step.observation
    obs_gym = dmc_obs2gym_obs(obs)
    # obs_gym_dict = dict(obs_gym)
    obs_flat = np.concatenate([val for val in obs_gym.values()])
    obs_conv = _convert_obs(obs_flat)
    # obs_flat = np.concatenate([val for val in obs_gym.values()]) #spaces.flatten(obs_space, obs_conv)
    # obs_flat = FlattenObservation(obs_conv)
    # obs_flat = spaces.flatten(obs_space, obs_gym_dict)
    action = agent.eval_actions(obs_flat)
    # action = action_space.sample()
    return action



def main(_):

    ## Define variables and flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
    # flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
    flags.DEFINE_integer('seed', 42, 'Random seed.')
    # flags.DEFINE_integer('eval_episodes', 1,
                        #  'Number of episodes used for evaluation.')
    # flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
    # flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
    # flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
    # flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
    # flags.DEFINE_integer('start_training', int(1e4),
                        #  'Number of training steps to start training.')
    # flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
    # flags.DEFINE_boolean('wandb', True, 'Log wandb.')
    # flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
    flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
    flags.DEFINE_integer('action_history', 1, 'Action history.')
    flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
    flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
    # flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
    config_flags.DEFINE_config_file(
        'config',
        'configs/droq_config.py',
        'File path to the training hyperparameter configuration.',
        lock_config=False)
    # FLAGS(sys.argv)

    # import pdb;pdb.set_trace()

    # env = suite.load(domain_name="humanoid", task_name="stand")
    # viewer.launch(env)

    ## Try without DMCGYM
    robot = A1(action_history=FLAGS.action_history)
    task = Run(robot,
                control_timestep=round(1.0 / FLAGS.control_frequency, 3),
                randomize_ground=False)
    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    # import pdb;pdb.set_trace()
    action_space = dmc_spec2gym_space(env.action_spec())
    obs_space =  dmc_spec2gym_space(env.observation_spec())
    obs_space = spaces.flatten_space(obs_space)
    obs_space = copy.deepcopy(obs_space)
    obs_space = _convert_space(obs_space)
    obs_space = spaces.flatten_space(obs_space)

    # env = make_mujoco_env(
    #             FLAGS.env_name,
    #             control_frequency=FLAGS.control_frequency,
    #             action_filter_high_cut=FLAGS.action_filter_high_cut, 
    #             action_history=FLAGS.action_history)
    # env = wrap_gym(env, rescale_actions=True)
    # env = gym.wrappers.RecordVideo(
    #         env,
    #         f'videos/eval_{FLAGS.action_filter_high_cut}',
    #         episode_trigger=lambda x: True)  ## Mateo: Also records video. Seems to do this even if it doesn't save the video
    # env.seed(FLAGS.seed + 42) 

    ## Create environment -> Creates task and dmcgym implicitly


    ## Load agent from checkpoint if it exists, else, do random actions
    kwargs = dict(FLAGS.config)  ## Mateo: I should look at what this does
    # agent = SACLearner.create(FLAGS.seed, env.observation_space,
    #                             env.action_space, **kwargs) 
    agent = SACLearner.create(FLAGS.seed, obs_space,
                                action_space, **kwargs) 

    chkpt_dir = os.path.join(os.getcwd(), 'saved/checkpoints')  ## This one should probably not be hardcoded
    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir) 

    if last_checkpoint is None:
        print("RANDOM AGENT")
    else:
        print("TRAINED AGENT")
        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

    policy = partial(sac_policy, agent, obs_space, action_space)
    viewer.launch(env, policy)

    ## Visualize agent
    # width=500
    # height=500,
    # name="a1_renderer"
    # window = viewer.gui.RenderWindow(width, height, name)
    # viewer.launch(env)



    # num_episodes = 100
    # import pdb;pdb.set_trace()
    # for ep in range(num_episodes):
    #     time_step = 
    #     observation, done = env.reset(), False
    #     step = 0
    #     while not done:
    #         print(f"Episode {ep}, step {step}")
    #         action = agent.eval_actions(observation)
    #         observation, reward, done, info, time_step = env.step(action)
    #         # env.render(mode="human")
    #         # import pdb;pdb.set_trace()
    #         step += 1



    # viewer.launch(env, agent)

if __name__ == "__main__":
    app.run(main)