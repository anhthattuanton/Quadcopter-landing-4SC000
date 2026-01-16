import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation_data import (a_pad_max, arm_length, dt, m, max_thrust, vx_max,
                             vx_pad_max, vy_max, x_init_max, x_init_min,
                             y_init_max, y_init_min)


class PlanarQuadcopterEnv(gym.Env):
    """
    A 2D Planar Quadcopter environment suitable for RL.
    Goal: Land gently on a moving platform.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Define action and observation space
        # Actions: [Main Thrust, Differential Thrust] or [Motor1, Motor2]
        # Using continuous values between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: [x, y, theta, x_dot, y_dot, theta_dot,
        # platform_x, platform_vel, platform_accel]
        low = np.array(
            [
                x_init_min,
                y_init_min,
                -np.pi,
                -vx_max,
                -vy_max,
                -10,
                -vx_pad_max,
                -a_pad_max,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [x_init_max, y_init_max, np.pi, vx_max, vy_max, 10, vx_pad_max, a_pad_max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Simulation constants 
        self.dt = dt
        self.gravity = 9.81
        self.mass = m
        self.arm_length = arm_length
        self.max_thrust = max_thrust
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize quadcopter at random position at top
        # Initialize platform at random position at bottom
        self.state = np.zeros(9, dtype=np.float32)
        self.state[1] = 10.0  # Initial Height

        return self.state, {}

    def step(self, action):
        """
        Apply physics equations here.
        TODO: Ask Copilot to implement 2D quadcopter dynamics.
        """
        # Placeholder logic
        truncated = False
        terminated = False
        reward = 0.0

        # Update physics (x, y, theta based on action)
        # Update platform position (moving sine wave or linear)

        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass
