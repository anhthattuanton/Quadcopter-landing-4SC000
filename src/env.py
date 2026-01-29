"""
Planar Quadcopter Gymnasium Environment.

This module defines a custom Gymnasium environment for training a reinforcement
learning agent to land a 2D quadcopter on a moving platform.

The environment follows the Gymnasium API and can be used with standard RL
libraries like Stable-Baselines3.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.dynamics_2d import PlanarQuadcopterDynamics
from src.platform_logic import MovingPlatform
from src.reward_func import calculate_reward
from src.simulation_data import (
    a_pad_max,
    arm_length,
    dt,
    vx_max,
    vx_pad_max,
    vy_max,
    x_init_max,
    x_init_min,
    y_init_max,
    y_init_min,
)


class PlanarQuadcopterEnv(gym.Env):
    """
    Gymnasium environment for 2D quadcopter landing on a moving platform.
    
    The agent controls a planar quadcopter with two motors and must land
    safely on a horizontally moving platform. The observation includes
    the quadcopter state and platform state.
    
    Observation Space (9 dimensions):
        [0] x: Horizontal position (m)
        [1] y: Vertical position (m)
        [2] theta: Orientation angle (rad)
        [3] x_dot: Horizontal velocity (m/s)
        [4] y_dot: Vertical velocity (m/s)
        [5] theta_dot: Angular velocity (rad/s)
        [6] platform_x: Platform horizontal position (m)
        [7] platform_v: Platform horizontal velocity (m/s)
        [8] platform_a: Platform horizontal acceleration (m/s²)
    
    Action Space (2 dimensions):
        [0] Left motor thrust: Normalized to [-1, 1]
        [1] Right motor thrust: Normalized to [-1, 1]
    
    Rewards:
        See reward_func.py for detailed reward structure.
    
    Termination:
        - Quadcopter touches ground (y <= 0)
        - Quadcopter goes out of bounds
    
    Attributes:
        dynamics (PlanarQuadcopterDynamics): Physics simulation handler.
        platform (MovingPlatform): Moving landing target.
        arm_length (float): Quadcopter arm length for visualization.
        state (np.ndarray): Current environment state.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        """Initialize the environment with action/observation spaces and components."""
        super().__init__()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        low = np.array(
            [x_init_min, 0, -np.pi, -vx_max, -vy_max, -2 * np.pi, 
             -10, -vx_pad_max, -a_pad_max],
            dtype=np.float32,
        )
        high = np.array(
            [x_init_max, y_init_max, np.pi, vx_max, vy_max, 2 * np.pi,
             10, vx_pad_max, a_pad_max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.dynamics = PlanarQuadcopterDynamics()
        self.arm_length = arm_length
        self.state = None

        self.platform = MovingPlatform(
            x_min=-10, x_max=10, v_max=vx_pad_max, a_max=a_pad_max, dt=dt
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a random initial state.
        
        The quadcopter starts at a random position within bounds, with small
        random velocities. The platform is also reset to a random state.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional reset options (unused).
        
        Returns:
            tuple: (observation, info)
                observation (np.ndarray): Initial state array of shape (9,).
                info (dict): Empty dictionary (Gymnasium API requirement).
        """
        super().reset(seed=seed)

        self.state = np.zeros(9, dtype=np.float32)

        for n in range(len(self.state)):
            if n == 1:
                self.state[n] = self.np_random.uniform(y_init_min, y_init_max - 5.0)
            elif n == 6:
                self.platform.reset()
                self.state[6] = self.platform.x
                self.state[7] = self.platform.v
                self.state[8] = self.platform.a
            elif n != 7 and n != 8:
                self.state[n] = self.np_random.uniform(
                    low=self.observation_space.low[n] / 4,
                    high=self.observation_space.high[n] / 4,
                )

        return self.state, {}

    def step(self, action):
        """
        Execute one simulation step with the given action.
        
        Updates the quadcopter state using physics simulation, advances the
        platform, and computes the reward.
        
        Args:
            action (np.ndarray): Motor commands of shape (2,), normalized to [-1, 1].
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                observation (np.ndarray): New state array of shape (9,).
                reward (float): Reward for this transition.
                terminated (bool): True if episode ended (landed or out of bounds).
                truncated (bool): Always False (no time limit truncation).
                info (dict): Diagnostic information from reward function.
        """
        x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new = (
            self.dynamics.step(self.state, action)
        )

        self.platform.step()

        self.state = np.array(
            [x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new,
             self.platform.x, self.platform.v, self.platform.a],
            dtype=np.float32,
        )

        reward, terminated, info = calculate_reward(self.state)
        truncated = False

        return self.state, reward, terminated, truncated, info
