from typing import Optional, Tuple

import dm_control.utils.transformations as tr
import numpy as np
from dm_control import composer
from dm_control.locomotion import arenas
from dm_control.utils import rewards

from sim.arenas import HField
from sim.tasks.utils import _find_non_contacting_height
from sim.tasks.aprl_reward import APRLReward

import transforms3d as tr3d

DEFAULT_CONTROL_TIMESTEP = 0.03
DEFAULT_PHYSICS_TIMESTEP = 0.001


def get_run_reward(x_velocity: float, move_speed: float, cos_pitch: float,
                   dyaw: float):
    reward = rewards.tolerance(cos_pitch * x_velocity,
                               bounds=(move_speed, 2 * move_speed),
                               margin=2 * move_speed,
                               value_at_margin=0,
                               sigmoid='linear')
    reward -= 0.1 * np.abs(dyaw)

    return 10 * reward  # [0, 1] => [0, 10]


class Run(composer.Task):

    def __init__(self,
                 robot,
                 terminate_pitch_roll: Optional[float] = 30,
                 physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
                 floor_friction: Tuple[float] = (1, 0.005, 0.0001),
                 randomize_ground: bool = True,
                 add_velocity_to_observations: bool = True,
                 arena_type: str = 'floor',
                 slope: float = 0.0,
                 friction: float = 1.0):

        self.floor_friction = floor_friction
        if arena_type == 'floor':
            self._floor = arenas.Floor(size=(10, 10))
        elif arena_type == 'hfield':
            self._floor = HField(size=(10, 10), randomize=randomize_ground)
            self._floor.mjcf_model.size.nconmax = 400
            self._floor.mjcf_model.size.njmax = 2000
        elif arena_type == 'bowl':
            from sim.arenas import Bowl
            self._floor = Bowl(size=(10, 10), slope=slope)
        else:
            raise ValueError(f"Unknown arena type: {arena_type}")

        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = (friction,) + self.floor_friction[1:]

        self._robot = robot
        self._floor.add_free_entity(self._robot)

        # observables = (self._robot.observables.proprioception +
        #                self._robot.observables.kinematic_sensors +
        #                [self._robot.observables.prev_action])
        # for observable in observables:
        #     observable.enabled = True

        # if not add_velocity_to_observations:
        #     self._robot.observables.sensors_velocimeter.enabled = False

        observables = self._robot.observables.aprl_obs
        for observable in observables:
            observable.enabled = True

        if hasattr(self._floor, '_top_camera'):
            self._floor._top_camera.remove()
        self._robot.mjcf_model.worldbody.add('camera',
                                             name='side_camera',
                                             pos=[0, -1, 0.5],
                                             xyaxes=[1, 0, 0, 0, 0.342, 0.940],
                                             mode="trackcom",
                                             fovy=60.0)

        self.control_timestep = control_timestep
        self.set_timesteps(physics_timestep=physics_timestep,
                           control_timestep=control_timestep)

        self._terminate_pitch_roll = terminate_pitch_roll

        self._move_speed = 0.5

        self.aprl_reward = APRLReward()

        # Initialize storage for last values
        self.last_dof_pos = self._robot._INIT_QPOS
        self.last_torques = np.zeros(12)
        self.last_forces = np.array([0.0, 0.0, 0.0, 0.0])
        self.last_contacts = np.array([False, False, False, False])

    def update_current_values(self, physics):
        # print("Inside update_current_values")
        xmat = physics.bind(self._robot.root_body).xmat.reshape(3, 3)
        roll, pitch, yaw = tr.rmat_to_euler(xmat, 'XYZ')
        velocimeter = physics.bind(self._robot.mjcf_model.sensor.velocimeter)
        gyro = physics.bind(self._robot.mjcf_model.sensor.gyro)
        framequat = physics.bind(self._robot.mjcf_model.sensor.framequat).sensordata.copy()
        up = tr3d.quaternions.quat2mat(framequat)[-1,-1]

        self.aprl_reward.base_ang_vel = gyro.sensordata.copy()
        self.aprl_reward.base_lin_vel = velocimeter.sensordata.copy()
        self.aprl_reward.roll = roll
        self.aprl_reward.pitch = pitch
        self.aprl_reward.yaw = yaw
        self.aprl_reward.up = up
        self.aprl_reward.default_dof_pos = self._robot._INIT_QPOS
        self.aprl_reward.last_dof_pos = self.last_dof_pos
        self.aprl_reward.dof_pos = physics.bind(self._robot.joints).qpos.copy()
        self.aprl_reward.dof_vel = physics.bind(self._robot.joints).qvel.copy()
        self.aprl_reward.torques = physics.bind(self._robot.actuators).ctrl.copy()
        self.aprl_reward.last_torques = self.last_torques
        self.aprl_reward.forces = self._robot.get_foot_forces_normalized(physics)
        self.aprl_reward.last_forces = self.last_forces
        self.aprl_reward.contacts = self._robot.get_foot_contacts(physics)
        self.aprl_reward.last_contacts = self.last_contacts
        self.aprl_reward.dt = self.control_timestep

    def get_reward(self, physics):
        # print("Inside reward")
        # self.update_current_values(physics)
        # xmat = physics.bind(self._robot.root_body).xmat.reshape(3, 3)
        # roll, pitch, yaw = tr.rmat_to_euler(xmat, 'XYZ')
        # velocimeter = physics.bind(self._robot.mjcf_model.sensor.velocimeter)

        # gyro = physics.bind(self._robot.mjcf_model.sensor.gyro)
        # framequat = physics.bind(self._robot.mjcf_model.sensor.framequat).sensordata.copy()
        # up = tr3d.quaternions.quat2mat(framequat)[-1,-1]

        # import ipdb;ipdb.set_trace()
        # self.aprl_reward.base_ang_vel = gyro.sensordata.copy()
        # self.aprl_reward.base_lin_vel = velocimeter.sensordata.copy()  
        # self.aprl_reward.roll = roll  ## TODO: Make sure it's local frame
        # self.aprl_reward.pitch = pitch
        # self.aprl_reward.yaw = yaw
        # self.aprl_reward.up = up  
        # self.aprl_reward.default_dof_pos = self._robot._INIT_QPOS 
        # self.aprl_reward.last_dof_pos = self.last_dof_pos
        # self.aprl_reward.dof_pos = physics.bind(self._robot.joints).qpos.copy() 
        # self.aprl_reward.dof_vel = physics.bind(self._robot.joints).qvel.copy() 
        # self.aprl_reward.torques = physics.bind(self._robot.actuators).ctrl.copy() 
        # self.aprl_reward.last_torques = self.last_torques 
        # self.aprl_reward.forces = self._robot.get_foot_forces(physics) 
        # self.aprl_reward.last_forces = self.last_forces 
        # self.aprl_reward.contacts = self._robot.get_foot_contacts(physics) 
        # self.aprl_reward.last_contacts = self.last_contacts 
        # self.aprl_reward.dt = self.control_timestep  

        # print(f"Last torques: {self.aprl_reward.last_torques}")
        # print(f"Current torques: {self.aprl_reward.torques}")
        # print(f"Last forces: {self.aprl_reward.last_forces}")
        # print(f"Current forces: {self.aprl_reward.forces}")
        # print(f"Last contacts: {self.aprl_reward.last_contacts}")
        # print(f"Current contacts: {self.aprl_reward.contacts}")

        aprl_reward = self.aprl_reward.get_reward()

        return aprl_reward

        # return get_run_reward(x_velocity=velocimeter.sensordata[0],
        #                       move_speed=self._move_speed,
        #                       cos_pitch=np.cos(pitch),
        #                       dyaw=gyro.sensordata[-1])


    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)

        # Terrain randomization
        if hasattr(self._floor, 'regenerate'):
            self._floor.regenerate(random_state)
            self._floor.mjcf_model.visual.map.znear = 0.00025
            self._floor.mjcf_model.visual.map.zfar = 50.

        new_friction = (random_state.uniform(low=self.floor_friction[0] - 0.25,
                                             high=self.floor_friction[0] +
                                             0.25), self.floor_friction[1],
                        self.floor_friction[2])
        for geom in self._floor.mjcf_model.find_all('geom'):
            geom.friction = new_friction

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._floor.initialize_episode(physics, random_state)

        self._failure_termination = False

        _find_non_contacting_height(physics,
                                    self._robot,
                                    qpos=self._robot._INIT_QPOS)
        
        # # Initialize last values to zero or appropriate initial states
        # self.last_dof_pos = self._robot._INIT_QPOS
        # self.last_torques = np.zeros(12)
        # self.last_forces = np.array([0.0, 0.0, 0.0, 0.0])
        # self.last_contacts = np.array([False, False, False, False])

        self.update_last_values(physics)
        self.update_current_values(physics)

    def before_step(self, physics, action, random_state):
        # print("Inside before_step")
        self.update_last_values(physics)
        # pass

    def before_substep(self, physics, action, random_state):
        self._robot.apply_action(physics, action, random_state)

    def action_spec(self, physics):
        return self._robot.action_spec

    def update_last_values(self, physics):
        # print("Inside update_last_values")
        # import ipdb;ipdb.set_trace()
        self.last_joint_positions = physics.bind(self._robot.joints).qpos.copy()
        self.last_torques = physics.bind(self._robot.actuators).ctrl.copy()
        self.last_forces = self._robot.get_foot_forces_normalized(physics)
        self.last_contacts = self._robot.get_foot_contacts(physics)

    def after_step(self, physics, random_state):
        # print("Inside after_step")
        self._failure_termination = False

        if self._terminate_pitch_roll is not None:
            roll, pitch, _ = self._robot.get_roll_pitch_yaw(physics)

            if (np.abs(roll) > self._terminate_pitch_roll
                    or np.abs(pitch) > self._terminate_pitch_roll):
                self._failure_termination = True

        self.update_current_values(physics)

    def should_terminate_episode(self, physics):
        return self._failure_termination

    def get_discount(self, physics):
        if self._failure_termination:
            return 0.0
        else:
            return 1.0

    @property
    def root_entity(self):
        return self._floor
