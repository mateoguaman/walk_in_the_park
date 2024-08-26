import numpy as np
from typing import Optional, Union

class RescaleActionAsymmetricStandalone:
    """
    A standalone class that rescales actions asymmetrically.
    """

    def __init__(
        self, 
        original_space,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        center_action: Optional[np.ndarray] = None
    ):
        """
        Initialize the RescaleActionAsymmetricStandalone class.

        Args:
            original_space: The original action space.
            low (Union[float, np.ndarray]): The lower bound(s) of the new action space.
            high (Union[float, np.ndarray]): The upper bound(s) of the new action space.
            center_action (Optional[np.ndarray]): The center of the action space. If None, it will be calculated.
        """
        self.original_space = original_space
        self._center_action = center_action
        self.low = low
        self.high = high
    
    @property
    def center_action(self) -> np.ndarray:
        """
        Get the center of the action space.

        Returns:
            np.ndarray: The center of the action space.
        """
        return self._center_action if self._center_action is not None else (
            (self.original_space.high + self.original_space.low) / 2.0
        )
    
    @center_action.setter
    def center_action(self, value: Optional[np.ndarray]):
        """
        Set the center of the action space.

        Args:
            value (Optional[np.ndarray]): The new center of the action space.
        """
        self._center_action = value
    
    @property
    def action_space(self):
        """
        Get the new action space.

        Returns:
            The new action space with updated bounds.
        """
        return type(self.original_space)(
            low=self.low,
            high=self.high,
            shape=self.original_space.shape,
            dtype=self.original_space.dtype
        )

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """
        Transform an action from the new space to the original space.

        Args:
            action (np.ndarray): The action in the new space.

        Returns:
            np.ndarray: The transformed action in the original space.
        """
        new_center = (self.high + self.low) / 2.0
        
        new_delta_action = action - new_center
        below_center_idx = new_delta_action < 0
        other_idx = np.logical_not(below_center_idx)

        new_delta_high = self.high - new_center
        if not isinstance(new_delta_high, float):
            new_delta_high = new_delta_high[other_idx]
        new_delta_low = new_center - self.low
        if not isinstance(new_delta_low, float):
            new_delta_low = new_delta_low[below_center_idx]

        center_action = self.center_action
        delta_center = new_delta_action.copy()
        delta_center[below_center_idx] *= (center_action[below_center_idx] - self.original_space.low[below_center_idx]) / new_delta_low
        delta_center[other_idx] *= (self.original_space.high[other_idx] - center_action[other_idx]) / new_delta_high
        ret = (center_action + delta_center).astype(self.original_space.dtype)
        return ret

    def inverse_transform_action(self, action: np.ndarray) -> np.ndarray:
        """
        Transform an action from the original space to the new space.

        Args:
            action (np.ndarray): The action in the original space.

        Returns:
            np.ndarray: The transformed action in the new space.
        """
        delta_center = action - self.center_action
        below_center_idx = delta_center < 0
        other_idx = np.logical_not(below_center_idx)

        new_center = (self.high + self.low) / 2.0
        new_delta_high = self.high - new_center
        if not isinstance(new_delta_high, float):
            new_delta_high = new_delta_high[other_idx]
        new_delta_low = new_center - self.low
        if not isinstance(new_delta_low, float):
            new_delta_low = new_delta_low[below_center_idx]

        center_action = self.center_action

        new_delta_center = delta_center.copy()
        new_delta_center[below_center_idx] *= new_delta_low / (center_action[below_center_idx] - self.original_space.low[below_center_idx])
        new_delta_center[other_idx] *= new_delta_high / (self.original_space.high[other_idx] - center_action[other_idx])
        return new_center + new_delta_center


import gym
import gym.spaces

class RescaleActionAsymmetric(gym.ActionWrapper):
    """
    A gym ActionWrapper that uses RescaleActionAsymmetricStandalone to rescale actions.
    """

    def __init__(
        self, 
        env: gym.Env, 
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        center_action: Optional[np.ndarray] = None
    ):
        """
        Initialize the RescaleActionAsymmetric wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            low (Union[float, np.ndarray]): The lower bound(s) of the new action space.
            high (Union[float, np.ndarray]): The upper bound(s) of the new action space.
            center_action (Optional[np.ndarray]): The center of the action space. If None, it will be calculated.
        """
        super().__init__(env)
        self.rescaler = RescaleActionAsymmetricStandalone(
            env.action_space, low, high, center_action
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        """
        Get the new action space.

        Returns:
            gym.spaces.Box: The new action space with updated bounds.
        """
        return self.rescaler.action_space

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Transform an action from the new space to the original space.

        Args:
            action (np.ndarray): The action in the new space.

        Returns:
            np.ndarray: The transformed action in the original space.
        """
        return self.rescaler.transform_action(action)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Transform an action from the original space to the new space.

        Args:
            action (np.ndarray): The action in the original space.

        Returns:
            np.ndarray: The transformed action in the new space.
        """
        return self.rescaler.inverse_transform_action(action)