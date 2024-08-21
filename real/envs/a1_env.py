import time

import numpy as np

import gym
import gym.spaces
from absl import logging
from dm_control.utils import rewards
from real import resetters
from real.envs import env_builder
from real.robots import a1, a1_robot, robot_config
from real.utilities import pose3d
import transforms3d as tr3d
from real.envs.aprl_reward import APRLReward

def get_run_reward(x_velocity: float, move_speed: float,
                   cos_pitch_cos_roll: float, terminate_pitch_roll_deg: float):
    termination = np.cos(np.deg2rad(terminate_pitch_roll_deg))
    upright = rewards.tolerance(cos_pitch_cos_roll,
                                bounds=(termination, float('inf')),
                                sigmoid='linear',
                                margin=termination + 1,
                                value_at_margin=0)

    forward = rewards.tolerance(x_velocity,
                                bounds=(move_speed, 2 * move_speed),
                                margin=move_speed,
                                value_at_margin=0,
                                sigmoid='linear')

    return upright * forward  # [0, 1] => [0, 10]


class A1Real(gym.Env):
    def __init__(
            self,
            zero_action: np.ndarray = np.asarray([0.05, 0.9, -1.8] * 4),
            action_offset: np.ndarray = np.asarray([0.2, 0.4, 0.4] * 4),
            limit_action_range: float = 1.0,
    ):
        logging.info(
            "WARNING: this code executes low-level control on the robot.")
        input("Press enter to continue...")
        self.zero_action = zero_action
        # self.action_offset = action_offset
        self.limit_action_range = limit_action_range

        self.env = env_builder.build_imitation_env()

        self.resetter = resetters.GetupResetter(self.env,
                                                True,
                                                standing_pose=zero_action)
        self.original_kps = self.env.robot._motor_kps.copy()
        self.original_kds = self.env.robot._motor_kds.copy()

        # min_actions = zero_action - action_offset
        # max_actions = zero_action + action_offset
        # min_actions = (self.env.robot._joint_angle_lower_limits - self.zero_action) * self.limit_action_range + self.zero_action
        # max_actions = (self.env.robot._joint_angle_upper_limits - self.zero_action) * self.limit_action_range + self.zero_action
        min_actions = self.env.robot._joint_angle_lower_limits
        max_actions = self.env.robot._joint_angle_upper_limits

        self.action_space = gym.spaces.Box(min_actions, max_actions)
        self._estimated_velocity = np.zeros(3)
        self._reset_var()

        self.aprl_reward = APRLReward()

        obs = self.observation()

        self.observation_space = gym.spaces.Box(float("-inf"),
                                                float("inf"),
                                                shape=obs.shape,
                                                dtype=np.float32)
        

    def _reset_var(self):
        self.prev_action = np.zeros_like(self.action_space.low)
        self.prev_qpos = self.zero_action
        self._last_timestamp = time.time()
        self._prev_pose = None

        ## Adding observations for APRL rewards
        self.last_torques = np.zeros(12)
        self.last_forces = np.array([0.0, 0.0, 0.0, 0.0])
        self.last_contacts = np.array([False, False, False, False])

    def reset(self):
        self.env._robot.SetMotorGains(kp=self.original_kps,
                                      kd=self.original_kds)
        self.resetter()
        self.env._robot.SetMotorGains(kp=[60.0] * 12, kd=[4.0] * 12)
        self._reset_var()
        input("Press Enter to Continue")

        return self.observation()

    def _get_imu(self):
        rpy = self.env._robot.GetBaseRollPitchYaw()
        drpy = self.env._robot.GetBaseRollPitchYawRate()

        assert len(rpy) >= 3, rpy
        assert len(drpy) >= 3, drpy

        # channels = ["R", "P", "dR", "dP", "dY"]
        channels = ["R", "P", "dR", "dP"]
        observations = np.zeros(len(channels))
        for i, channel in enumerate(channels):
            if channel == "R":
                observations[i] = rpy[0]
            if channel == "Rcos":
                observations[i] = np.cos(rpy[0])
            if channel == "Rsin":
                observations[i] = np.sin(rpy[0])
            if channel == "P":
                observations[i] = rpy[1]
            if channel == "Pcos":
                observations[i] = np.cos(rpy[1])
            if channel == "Psin":
                observations[i] = np.sin(rpy[1])
            if channel == "Y":
                observations[i] = rpy[2]
            if channel == "Ycos":
                observations[i] = np.cos(rpy[2])
            if channel == "Ysin":
                observations[i] = np.sin(rpy[2])
            if channel == "dR":
                observations[i] = drpy[0]
            if channel == "dP":
                observations[i] = drpy[1]
            if channel == "dY":
                observations[i] = drpy[2]
        return observations

    def _compute_delta_time(self, current_time):
        delta_time_s = current_time - self._last_timestamp
        self._last_timestamp = current_time
        return delta_time_s

    def _update_vel(self, delta_time_s):
        self._estimated_velocity = self.env._robot.GetBaseVelocity()

    def observation(self):
        # delta_time_s = self._compute_delta_time(time.time())

        self.env._robot.ReceiveObservation()
        self.update_current_values()
        # qpos = self.env._robot.GetMotorAngles()
        # qvel = self.env._robot.GetMotorVelocities()
        # imu = self._get_imu()

        # self._update_vel(delta_time_s)
        # vel = self._estimated_velocity.copy()
        # new_vel = np.array([vel[1], vel[0], vel[2]])

        # self._foot_contacts = self.env._robot.GetFootContacts().astype(
            # np.float32)
        
        ## APRL Observations
        # torques = self.env._robot._observed_motor_torques

        # foot_forces_normalized = self.env._robot.GetFootForcesNormalized()


        # return np.concatenate(
        #     [qpos, qvel, new_vel, imu, self.prev_action,
        #      self._foot_contacts]).astype(np.float32)
        return np.concatenate(
            [self.qpos, self.qvel, self.imu, self.vel, self.torques, self.foot_forces_normalized, 
             self.prev_action]).astype(np.float32)
    
    def update_current_values(self):
        delta_time_s = self._compute_delta_time(time.time())
        self._update_vel(delta_time_s)
        self.qpos = self.env._robot.GetMotorAngles()
        self.qvel = self.env._robot.GetMotorVelocities()
        self.rpy = self.env._robot.GetBaseRollPitchYaw()
        self.drpy = self.env._robot.GetBaseRollPitchYawRate()
        self.quaternion = self.env._robot._base_orientation
        self.imu = self._get_imu()  ## Note: Changed this internally to match APRL obs. Uncomment inside to change back to WitP.
        self.vel = self._estimated_velocity.copy()
        # self.new_vel = np.array([self.vel[1], self.vel[0], self.vel[2]])  ## TODO: verify why this is necessary
        self.torques = self.env._robot._observed_motor_torques
        self.energy = (self.qvel * self.torques).sum()
        self.foot_forces = self.env._robot.GetFootForces()
        self.foot_forces_normalized = self.env._robot.GetFootForcesNormalized()
        self.contacts = self.env._robot.GetFootContacts()
        self._foot_contacts = self.env._robot.GetFootContacts().astype(
            np.float32)  ## Redundant, but kept to keep it available for the original witp reward

        ## TODO: think about making this a separate function
        self.aprl_reward.base_ang_vel = self.drpy
        self.aprl_reward.base_lin_vel = self.vel
        self.aprl_reward.roll = self.rpy[0]
        self.aprl_reward.pitch = self.rpy[1]
        self.aprl_reward.yaw = self.rpy[2]
        self.aprl_reward.up = tr3d.quaternions.quat2mat(self.quaternion)[-1,-1]
        self.aprl_reward.default_dof_pos = self.zero_action
        self.aprl_reward.last_dof_pos = self.prev_qpos
        self.aprl_reward.dof_pos = self.qpos
        self.aprl_reward.dof_vel = self.qvel
        self.aprl_reward.torques = self.torques
        self.aprl_reward.last_torques = self.last_torques
        self.aprl_reward.forces = self.foot_forces_normalized
        self.aprl_reward.last_forces = self.last_forces
        self.aprl_reward.contacts = self.contacts
        self.aprl_reward.last_contacts = self.last_contacts
        self.aprl_reward.dt = self.env._robot.time_step * self.env._robot._action_repeat

    def update_last_values(self):
        self.prev_action[:] = self.action
        self.prev_qpos[:] = self.qpos
        self.last_torques[:] = self.torques
        self.last_forces[:] = self.foot_forces_normalized
        self.last_contacts[:] = self.contacts

    def step(self, action):
        assert self.env._robot._action_repeat == 50
        self.action = action
        self.env._robot.Step(action, robot_config.MotorControlMode.POSITION)
        ## TODO: Could extract update_current_values outside of self.observation to make it more explicit
        obs = self.observation()
        # self.prev_action[:] = action

        accel_velocity = self.env._robot.GetBaseVelocity()  ## TODO: Currently seems to be the same as velocity, get rid of it?
        
        ## TODO: Uncomment to recover WitP reward
        # velocity = self._estimated_velocity.copy()
        # roll, pitch, yaw = self.env._robot.GetTrueBaseRollPitchYaw()
        # drpy = self.env._robot.GetBaseRollPitchYawRate()

        # lin_vel = self._estimated_velocity[0]
        # target_vel = .5

        term_rad_roll = term_rad_pitch = np.deg2rad(30)

        # reward = rewards.tolerance(lin_vel * np.cos(pitch),
        #                            bounds=(target_vel, 2 * target_vel),
        #                            margin=2 * target_vel,
        #                            value_at_margin=0,
        #                            sigmoid='linear')
        # reward -= 0.1 * np.abs(drpy[-1])
        # reward *= max(self._foot_contacts)
        # reward *= 10.0
        ## TODO: done uncomment

        # qvel = self.env._robot.GetMotorVelocities()
        # torque = self.env._robot._observed_motor_torques
        # energy = (qvel * torque).sum()

        reward = self.aprl_reward.get_reward()


        if abs(self.rpy[0]) > term_rad_roll or abs(
                self.rpy[1]) > term_rad_pitch or not self.env._robot._is_safe:
            done = True
        else:
            done = False

        info = {
            'velocity':
            self.vel,
            'acc_velocity':
            accel_velocity,
            'raw_acc':
            self.env._robot._velocity_estimator._raw_acc.copy(),
            'calibrated_acc':
            self.env._robot._velocity_estimator._calibrated_acc.copy(),
            'leg_vels':
            np.array(self.env._robot._velocity_estimator._observed_velocities),
            'rpy':
            np.array([self.rpy[0], self.rpy[1], self.rpy[2]]),  ## TODO: COuld just replace with rpy
            'jangles':
            self.env._robot.GetMotorAngles(),
            'energy':
            self.energy,
            'x_vel':
            self.vel[0] * np.exp(-np.abs(self.drpy[1:]).mean()),
            'dr':
            self.drpy[0],
            'dp':
            self.drpy[1],
            'dy':
            self.drpy[2]
        }

        self.update_last_values()

        return obs, reward, done, info
    
    @property
    def joint_qpos_init(self) -> np.ndarray:
        return self.zero_action
    
    @property
    def action_qpos_mins(self) -> np.ndarray:
        return (self.env.robot._joint_angle_lower_limits - self.joint_qpos_init) * self.limit_action_range + self.joint_qpos_init
    
    @property
    def action_qpos_maxs(self) -> np.ndarray:
        return (self.env.robot._joint_angle_upper_limits - self.joint_qpos_init) * self.limit_action_range + self.joint_qpos_init
