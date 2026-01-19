import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.platform_logic import MovingPlatform
from src.simulation_data import (
    J,
    a_pad_max,
    arm_length,
    dt,
    m,
    max_thrust,
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

        # Simulation constants
        self.dt = dt
        self.gravity = 9.81
        self.mass = m
        self.arm_length = arm_length
        self.max_thrust = max_thrust
        self.J = J
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
                self.state[n] = self.np_random.uniform(y_init_min, y_init_max)
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
            observation (np.array), reward (dict), terminated (bool), truncated (bool), info (dict)
        """
        # 1. Action Handling
        action = np.clip(action, -1.0, 1.0)
        F_left = (action[0] + 1.0) * (self.max_thrust / 2.0)
        F_right = (action[1] + 1.0) * (self.max_thrust / 2.0)

        # Extract current state
        x, y, theta, x_dot, y_dot, theta_dot = self.state[0:6]

        # 2. Calculate Dynamics
        F_total = F_left + F_right
        moment = (F_right - F_left) * self.arm_length

        # Accelerations
        accel_x = -(F_total / self.mass) * np.sin(theta)
        accel_y = (F_total / self.mass) * np.cos(theta) - self.gravity
        accel_theta = moment / self.J

        # 3. Integration (Semi-Implicit Euler)
        x_dot_new = x_dot + accel_x * self.dt
        y_dot_new = y_dot + accel_y * self.dt
        theta_dot_new = theta_dot + accel_theta * self.dt

        x_new = x + x_dot_new * self.dt
        y_new = y + y_dot_new * self.dt
        theta_new = theta + theta_dot_new * self.dt

        # Normalize theta to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        # 4. Platform Logic
        self.platform.step()

        # Update state
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

        # 5. Termination
        distance_to_platform = np.abs(x_new - self.platform.x)
        velocity_magnitude = np.sqrt(x_dot_new**2 + y_dot_new**2)

        terminated = y_new <= 0.0
        truncated = False

        info = {
            "distance_to_platform": distance_to_platform,
            "velocity": velocity_magnitude,
            "platform_x": self.platform.x,
        }

        return self.state, {}, terminated, truncated, info
