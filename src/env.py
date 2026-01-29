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
    A 2D Planar Quadcopter environment suitable for RL.
    Goal: Land gently on a moving platform.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: [x, y, theta, x_dot, y_dot, theta_dot,
        # platform_x, platform_vel, platform_accel]
        low = np.array(
            [
                x_init_min,
                0,
                -np.pi,
                -vx_max,
                -vy_max,
                -2 * np.pi,
                -10,
                -vx_pad_max,
                -a_pad_max,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                x_init_max,
                y_init_max,
                np.pi,
                vx_max,
                vy_max,
                2 * np.pi,
                10,
                vx_pad_max,
                a_pad_max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize dynamics and platform
        self.dynamics = PlanarQuadcopterDynamics()
        self.arm_length = arm_length  # Keep for visualization
        self.state = None

        self.platform = MovingPlatform(
            x_min=-10, x_max=10, v_max=vx_pad_max, a_max=a_pad_max, dt=dt
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a blank state
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
        Executes one time step within the environment.

        Returns:
            observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        # 1. Compute drone dynamics
        x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new = (
            self.dynamics.step(self.state, action)
        )

        # 2. Platform Logic
        self.platform.step()

        # 3. Update state
        self.state = np.array(
            [
                x_new,
                y_new,
                theta_new,
                x_dot_new,
                y_dot_new,
                theta_dot_new,
                self.platform.x,
                self.platform.v,
                self.platform.a,
            ],
            dtype=np.float32,
        )

        # 4. Calculate Reward
        reward, terminated, info = calculate_reward(
            self.state
        )

        truncated = False

        return self.state, reward, terminated, truncated, info
