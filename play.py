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

import matplotlib.pyplot as plt



import copy
from typing import OrderedDict

import dm_env
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict
np.set_printoptions(precision=3, suppress=True, linewidth=100)



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
    
def dmc_obs2flat_gym_obs(time_step):
    obs = time_step.observation
    obs_gym = dmc_obs2gym_obs(obs)
    obs_flat = np.concatenate([val for val in obs_gym.values()])
    obs_conv = _convert_obs(obs_flat)
    return obs_conv

def dmc_timestep2gym_step(time_step):
    observation = dmc_obs2flat_gym_obs(time_step)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {}
    if done and time_step.discount == 1.0:
        info['TimeLimit.truncated'] = True
    return observation, reward, done, info

def sac_policy(agent, action_transformer, obs_space, action_space, time_step):
    obs_flat = dmc_obs2flat_gym_obs(time_step)
    action = agent.eval_actions(obs_flat)
    action = action_transformer.transform(action)
    return action

class ActionTransform:
    def __init__(self, action_space, action_filter_high_cut=None, seed=42):
        '''
        Here, the ainput action will be the raw output of the neural network.
        The order of action transformations in dmcgym seems to be the following:
        1. RescaleAction.action()
        2. ClipAction.action()
        3. TimeLimit.step()
        4. FlattenObservation.step()
        5. SinglePrecision.observation()

        Now, we will see what are the inputs to the action operations (1. and 2.)
        It definitely seems like operations 1. and 2. are the only operations applied to the output of the neural network.

        Now, let's see what are the exact transformations applied in 1.
        The parameters in 1. are the following:
            self.action_space: Box(-1.0, 1.0, (12,), float32)
            low: [-0.15  0.3  -1.8  -0.15  0.3  -1.8  -0.15  0.3  -1.8  -0.15  0.3  -1.8 ]
            high: [ 0.25  1.1  -1.    0.25  1.1  -1.    0.25  1.1  -1.    0.25  1.1  -1.  ]
            self.min_action: [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
            self.max_action: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

            self.min_action comes from -1 + np.zeros()
            self.max_action comes from 1 + np.zeros()
            self.action_space comes from making a space out of self.min_action and self.max_action
            low comes from _INIT_QPOS-ACTION_OFFSET
            high comes from _INIT_QPOS+ACTION_OFFSET

        There are two operations that occur in 1.:
            a) action = low + (high - low) * (
                (action - self.min_action) / (self.max_action - self.min_action)
            )
            b) action = np.clip(action, low, high)

        Now, let's see what are the exact transformations applied in 2.
        The parameters in 2. are the following:
            self.action_space: Box([-0.15  0.3  -1.8  -0.15  0.3  -1.8  -0.15  0.3  -1.8  -0.15  0.3  -1.8 ], 
                    [ 0.25  1.1  -1.    0.25  1.1  -1.    0.25  1.1  -1.    0.25  1.1  -1.  ], (12,), float32)

            self.action_space.low comes from _INIT_QPOS-ACTION_OFFSET
            self.action_space.low comes from _INIT_QPOS+ACTION_OFFSET

        There is one operation that occurs in 2.:
            a) np.clip(action, self.action_space.low, self.action_space.high)

        NOTE: With this transformation, I am able to visualize the policy successfully! Actions that actually get 
        passed to the dmcgym environment match the ones using this transformation. However, note that the observations are not
        identical. Especially the velocities seem to differ. This could be due to different seeds, single vs double resolution
        for floats, or something else that is undiagnosed.

        Transformations that get applied to the action after it gets passed to the environment:
        1. action = self.kp * (desired_qpos - qpos) - kd * qvel
            Note that this means that the action output by the network is added as a residual on top of the current joint positions
            self.kp = 60
            self.kd = 10
            qvel is the raw joint velocity obtained from mujoco
            qpos is the raw joint positions obtained from mujoco
        2. minimum, maximum = self.ctrllimits
        action = np.clip(action, minimum, maximum)
            Note that the control limits are: 
                min: [-33.5, -33.5, -33.5, -33.5, -33.5, -33.5, -33.5, -33.5, -33.5, -33.5, -33.5, -33.5]
                max: [33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5, 33.5]

        '''
        assert isinstance(action_space, Box)
        if action_filter_high_cut is not None:
            raise NotImplementedError()
        self.action_space = action_space

        ## Make an action space with restricted joints

        self._INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4)
        self.ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)

        if self.action_space.shape[0] == 12:
            lower_bound = self._INIT_QPOS - self.ACTION_OFFSET
            upper_bound = self._INIT_QPOS + self.ACTION_OFFSET
        else:
            lower_bound = np.concatenate([self._INIT_QPOS - self.ACTION_OFFSET, [-1.0]])
            upper_bound = np.concatenate([self._INIT_QPOS + self.ACTION_OFFSET, [1.0]])

        min_action = np.asarray(lower_bound)
        max_action = np.asarray(upper_bound)

        min_action = min_action + np.zeros(self.action_space.shape,
                                           dtype=self.action_space.dtype)

        max_action = max_action + np.zeros(self.action_space.shape,
                                           dtype=self.action_space.dtype)

        min_action = np.maximum(min_action, self.action_space.low)
        max_action = np.minimum(max_action, self.action_space.high)

        self.restricted_action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=self.action_space.shape,
            dtype=self.action_space.dtype,
        )

        self.action_space.seed(seed)
        self.restricted_action_space.seed(seed)

        ## Make an action space with same dimensions as other one, but within [1, -1]
        min_norm_action = -1
        max_norm_action = 1
        self.min_norm_action = (
            np.zeros(self.restricted_action_space.shape, dtype=self.restricted_action_space.dtype) + min_norm_action
        )
        self.max_norm_action = (
            np.zeros(self.restricted_action_space.shape, dtype=self.restricted_action_space.dtype) + max_norm_action
        )
        self.norm_action_space = spaces.Box(
            low=self.min_norm_action,
            high=self.max_norm_action,
            shape=self.restricted_action_space.shape,
            dtype=self.restricted_action_space.dtype,
        )


    def transform(self, action):
        ## 1. 
        low = self.restricted_action_space.low
        high = self.restricted_action_space.high
        min_action = self.norm_action_space.low
        max_action = self.norm_action_space.high

        action = low + (high - low) * (
            (action - min_action) / (max_action - min_action)
        )
        action = np.clip(action, low, high)

        ## 2. This seems redundant
        action = np.clip(action, low, high)


        return action        



def main(_):

    ## Define variables and flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
    flags.DEFINE_integer('seed', 42, 'Random seed.')
    flags.DEFINE_integer('max_steps', int(1e3), 'Number of training steps.')
    flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
    flags.DEFINE_integer('action_history', 1, 'Action history.')
    flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
    flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
    config_flags.DEFINE_config_file(
        'config',
        'configs/droq_config.py',
        'File path to the training hyperparameter configuration.',
        lock_config=False)

    ## Try without DMCGYM
    robot = A1(action_history=FLAGS.action_history)
    task = Run(robot,
                control_timestep=round(1.0 / FLAGS.control_frequency, 3),
                randomize_ground=True)
    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    # import pdb;pdb.set_trace()
    action_space = dmc_spec2gym_space(env.action_spec())
    obs_space =  dmc_spec2gym_space(env.observation_spec())
    obs_space = spaces.flatten_space(obs_space)
    obs_space = copy.deepcopy(obs_space)
    obs_space = _convert_space(obs_space)
    obs_space = spaces.flatten_space(obs_space)
    obs_space.seed(FLAGS.seed + 42)

    ## Create eval environment to compare the observations received using this env vs the dm_control env
    eval_env = make_mujoco_env(
                FLAGS.env_name,
                control_frequency=FLAGS.control_frequency,
                action_filter_high_cut=FLAGS.action_filter_high_cut, 
                action_history=FLAGS.action_history)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    # eval_env = gym.wrappers.RecordVideo(
    #         eval_env,
    #         f'videos/eval_{FLAGS.action_filter_high_cut}',
    #         episode_trigger=lambda x: True)  ## Mateo: Also records video. Seems to do this even if it doesn't save the video
    eval_env.seed(FLAGS.seed + 42) 

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

    action_transformer = ActionTransform(action_space=action_space, seed=FLAGS.seed + 42)

    ## Uncomment this to visualize policy
    policy = partial(sac_policy, agent, action_transformer, obs_space, action_space)
    viewer.launch(env, policy)  ## Uncomment this to visualize policy

    # num_episodes = 100
    # for ep in range(num_episodes):
    #     time_step, done = env.reset(), False
    #     dmc_obs = dmc_obs2flat_gym_obs(time_step)
    #     eval_obs, eval_done = eval_env.reset(), False
    #     step = 0
    #     while not eval_done:
    #         print("==========")
    #         print(f"Episode {ep}, step {step}")
    #         print("dmcgym observation (ground truth): ")
    #         print(eval_obs)
    #         print("dm_control observation: ")
    #         print(dmc_obs)
    #         print("----------")

    #         action_eval = agent.eval_actions(eval_obs)
    #         print("action_eval (ground truth): ")
    #         print(action_eval)
    #         action_dmc = agent.eval_actions(dmc_obs)
    #         print("action_dmc")
    #         print(action_dmc)   
    #         trans_action = action_transformer.transform(action_dmc)
    #         print("transformed action: ")
    #         print(trans_action)
    #         print("----------")
    #         # print("Taking step in manual dm_control env:")
    #         time_step = env.step(trans_action)
    #         next_dmc_obs, dmc_reward, dmc_done, dmc_info = dmc_timestep2gym_step(time_step)
    #         # print("Taking step in dmcgym env (ground_truth):")
    #         # print("Action that gets passed into eval_env in play.py")
    #         next_eval_obs, eval_reward, eval_done, eval_info = eval_env.step(action_eval)

    #         print(f"dmcgym reward (ground truth): {eval_reward}")
    #         print(f"dm_control reward: {dmc_reward}")
            

    #         dmc_obs = next_dmc_obs
    #         eval_obs = next_eval_obs
    #         step += 1








    # viewer.launch(env, agent)

if __name__ == "__main__":
    app.run(main)


'''
Transformations that get applied to the action before it gets passed to the environment:
1. Set time limit to be 400 (NOT an action transformation, ignore)
2. Clips action within action space limits: np.clip(action, self.action_space.low, self.action_space.high)
3. Potentially implements an action filter (NOT USED DURING TRAINING, ignore)
4. Clips actions to be bounded within the  following:  [INIT_QPOS - ACTION_OFFSET, INIT_QPOS + ACTION_OFFSET]
    where ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4)
          INIT_QPOS = sim.robots.a1.A1._INIT_QPOS
   Note: there is a class that implements a gym env wrapper for this. Should copy this class but remove gym dependency
5. Sets the seed for the obs and action spaces: self.env.action_space.seed(seed)
6. Rescales the action to be within [-1, 1] (and sets the action space to be -1, 1):
        low = self.env.action_space.low  ## Here will be -1 with the right dimensions
        high = self.env.action_space.high  ## Here will be  with the right dimensions
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
7. Clips actions to be bounded within the following: [-1, 1]



Transformations that get applied to the observation before it gets passed to the neural network: 
1. Flatten: np.concatenate([flatten(s, x[key]) for key, s in space.spaces.items()])
2. Convert to float16 instead of float32
3. Set seed for the obs space
4. Flatten again (Feels unnecesary)

Question: Does it get applied for multiple steps?? Yes, 50, but it is the same on both (sim_freq/control_freq = 1000/20=50)
First of all, let's verify that the input action that gets to the Sim code is the same as the one that is output by the transform function
[PASSING] Check: are these ^ equal?
'''

'''
What I should see:
1. Initialize a dm_control environment using composer (this doesn't work)
2. Initialize an eval_env using dmcgym (this works)
3. Get obs_dmc and obs_eval by resetting each of the environments
4. [PASSING] Check: These should be the same
    If this fails, the initial environment set up is different
5. Take an action by querying action_eval = agent.eval_policy(obs_eval)
6. Take an action by querying action_dmc = agent.eval_policy(dmc_obs2flat_gym_obs(time_step))
7. [PASSING] Check: These actions should be the same
    If this fails, the observation transformation or the policy are different
8. Get the next observation and reward when applying this action to the environment: next_obs_eval, reward_eval, etc = eval_env.step(action_eval)
9. Get the next observation and reward when applying this action to the environment: next_obs_dmc, reward_dmc, etc = dmc_env.step(action_dmc)
10. [Failing] Check: next_obs_eval should be equal to next_obs_dmc
    If this fails, the way actions are processed is different:
    Symptom: The next observations differ. More importantly, the actual actions passed to the environment are different.
        This suggests that actions in the dmcgym environment are getting processed in a way that is not yet happening using raw dm_control
'''


