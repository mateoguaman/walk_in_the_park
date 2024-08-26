import numpy as np

class APRLReward:
    def __init__(self):
        self.target_vel = 1.0
        self.commands = np.array([self.target_vel, 0.0, 0.0, 0.0])
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.base_lin_vel = np.array([0.0, 0.0, 0.0])
        self.base_ang_vel = np.array([0.0, 0.0, 0.0])
        self.up = 1.0
        self.default_dof_pos = np.zeros(12)
        self.dof_pos = np.zeros(12)
        self.last_dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.torques = np.zeros(12)
        self.last_torques = np.zeros(12)
        self.contacts = np.array([True, True, True, True])
        self.last_contacts = self.contacts.copy()
        self.forces = np.array([1.0, 1.0, 1.0, 1.0])
        self.last_forces = self.forces.copy()
        self.dt = 1./20.

    def near_quadratic_bound(self, value, target, left_margin, right_margin, out_of_margin_activation="linear", power=2.0, value_at_margin=0.0):
        delta = value - target
        fract = np.where(delta > 0, delta / right_margin, delta / left_margin)

        if out_of_margin_activation == "near_quadratic":
            rew = 1 - (1 - value_at_margin) * (np.abs(fract) ** power)
        else:
            clipped_fract = np.clip(fract, -1.0, 1.0)
            rew = 1 - (1 - value_at_margin) * (np.abs(clipped_fract) ** power)
            oodfract = fract - clipped_fract
            if out_of_margin_activation == "linear":
                rew -= (1 - value_at_margin) * np.abs(oodfract)
            elif out_of_margin_activation == "quadratic":
                rew -= (1 - value_at_margin) * (oodfract ** 2)
            elif out_of_margin_activation == "gaussian":
                rew += value_at_margin * np.exp(-oodfract**2/0.25)

        return rew

    def sigmoids(self, x, value_at_1, sigmoid):
        if sigmoid == "gaussian":
            scale = np.sqrt(-2 * np.log(value_at_1))
            return np.exp(-0.5 * (x*scale)**2)
        elif sigmoid == "linear":
            scale = 1 - value_at_1
            scaled_x = x * scale
            return np.where(np.abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    def tolerance(self, x, bounds, margin, sigmoid="gaussian", value_at_margin=0.1):
        lower, upper = bounds
        in_bounds = np.logical_and(lower <= x, x <= upper)
        zero_margin = np.where(in_bounds, 1.0, 0.0)
        d = np.where(x < lower, (lower - x) / margin, (x - upper) / margin)
        nonzero_margin = np.where(in_bounds, 1.0, self.sigmoids(d, value_at_margin, sigmoid))
        value = np.where(margin == 0.0, zero_margin, nonzero_margin)
        return value

    def _reward_aprl_level_velocity(self):
        target_velocity_x = self.commands[0]
        velocity_norm = np.linalg.norm(self.base_lin_vel)
        projected_x_velocity = np.cos(self.pitch) * self.base_lin_vel[0]
        reward_v = self.near_quadratic_bound(
            projected_x_velocity,
            target_velocity_x,
            target_velocity_x,
            target_velocity_x,
            "linear",
            1.6,
            0.0
        )

        target_delta_yaw = self.commands[2]
        reward_v *= (1 + np.cos(target_delta_yaw)) / 2
        reward_v *= max(target_velocity_x, 1./5.)
        reward_v *= 20.

        reward_v = reward_v if target_velocity_x >= 0.05 else 3 * (1 - velocity_norm * 5)

        up = self.up
        upright_coefficient = (0.5 * up + 0.5)**2
        reward_v_constrained = reward_v * upright_coefficient
        return reward_v_constrained

    def _reward_aprl_qpos(self):
        offset = np.array([0.2, 0.4, 0.4] * 4)
        bounds = (self.default_dof_pos - offset, self.default_dof_pos + offset)
        qpos = self.dof_pos
        qpos_penalty_norm = self.tolerance(qpos, bounds, offset, sigmoid="gaussian", value_at_margin=0.6)
        qpos_penalty_norm = np.prod(qpos_penalty_norm)
        qpos_penalty = (0.6 - qpos_penalty_norm) * 2.0
        return qpos_penalty

    def _reward_aprl_pitch_rate(self):
        return 0.4 * (np.abs(self.base_ang_vel[1])**1.4) - 0.5

    def _reward_aprl_roll_rate(self):
        return 0.2 * (np.abs(self.base_ang_vel[0])**1.4) - 0.5

    def _reward_aprl_energy(self):
        return np.sum(np.abs(self.dof_vel * self.torques))

    def _reward_aprl_smooth_torque(self):
        diff_torque = self.torques - self.last_torques
        diff_torque_norm = np.linalg.norm(diff_torque) ** 1.5
        smooth_torque_penalty = 0.005 * (diff_torque_norm - 150)
        return smooth_torque_penalty

    def _reward_aprl_joint_diagonal_difference(self):
        FR_qvel = self.dof_vel[0:3]
        FL_qvel = self.dof_vel[3:6]
        FL_qvel[0] *= -1
        RR_qvel = self.dof_vel[6:9]
        RL_qvel = self.dof_vel[9:12]
        RL_qvel[0] *= -1

        diagonal_difference = np.linalg.norm(FR_qvel[1:] - RL_qvel[1:]) + np.linalg.norm(FL_qvel[1:] - RR_qvel[1:])

        diagonal_difference_penalty = 0.04 * diagonal_difference
        velocity_norm = np.linalg.norm(self.base_lin_vel)
        diagonal_difference_penalty *= (0.5 + velocity_norm)
        return diagonal_difference_penalty

    def _reward_aprl_joint_shoulder_difference(self):
        FR_qvel = self.dof_vel[0:3]
        FL_qvel = self.dof_vel[3:6]
        FL_qvel[0] *= -1
        RR_qvel = self.dof_vel[6:9]
        RL_qvel = self.dof_vel[9:12]
        RL_qvel[0] *= -1

        shoulder_difference = np.linalg.norm(FR_qvel[1:] - RL_qvel[1:]) * (np.sum(self.contacts[0:2]) == 1.0) + np.linalg.norm(FL_qvel[1:] - RR_qvel[1:]) * (np.sum(self.contacts[2:]) == 1.0) * 2.0
        velocity_norm = np.linalg.norm(self.base_lin_vel)

        shoulder_difference_penalty = 0.02 * (-1 * shoulder_difference)
        shoulder_difference_penalty *= (0.5 + velocity_norm)
        return shoulder_difference_penalty

    def _reward_aprl_contact(self):
        diff_joint_qpos = self.dof_pos - self.last_dof_pos
        prev_contacts = self.last_contacts
        prev_forces = self.last_forces
        contacts = self.contacts
        forces = self.forces

        CONTACT_DELTA_QPOS_THRESHOLD = -0.2
        CONTACT_DELTA_FORCE_THRESHOLD = 0.4
        qpos_threshold = CONTACT_DELTA_QPOS_THRESHOLD * self.dt
        force_threshold = CONTACT_DELTA_FORCE_THRESHOLD * self.dt

        delta_force = forces - prev_forces
        diff_joint_qpos_reshaped = diff_joint_qpos.reshape(4, 3)
        delta_qpos = diff_joint_qpos_reshaped[:, 1:3].sum(axis=1)

        condition_contact = np.logical_and(
            prev_contacts,
            np.logical_or(
                delta_qpos >= qpos_threshold,
                delta_force < -force_threshold
            )
        )

        condition_no_contact = np.logical_and(
            np.logical_not(prev_contacts),
            np.logical_or(
                delta_qpos <= -qpos_threshold,
                delta_force > force_threshold
            )
        )

        rew_norms = np.logical_or(condition_contact, condition_no_contact)
        rew_norm = rew_norms.mean()
        rew_norm *= 2.
        return rew_norm

    def _reward_aprl_yaw_rate(self):
        rew_aprl_yaw = self.near_quadratic_bound(np.abs(self.base_ang_vel[2]),
                                                 self.commands[2],
                                                 np.pi/4 * self.dt,
                                                 np.pi/4 * self.dt,
                                                 "gaussian",
                                                 1.6,
                                                 0.1)
        rew_aprl_yaw *= 4.0 * (0.1 * np.linalg.norm(self.base_lin_vel))
        return rew_aprl_yaw
    
    def get_reward(self):
        return self._reward_aprl_level_velocity() + self._reward_aprl_contact() - self._reward_aprl_qpos() - self._reward_aprl_pitch_rate() - self._reward_aprl_roll_rate() - self._reward_aprl_smooth_torque() - self._reward_aprl_joint_diagonal_difference() - self._reward_aprl_joint_shoulder_difference()