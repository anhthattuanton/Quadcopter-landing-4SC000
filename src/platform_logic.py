"""
Moving Platform Logic Module.

This module implements the behavior of a horizontally moving landing platform
that the quadcopter must track and land on. The platform moves with bounded
random acceleration, creating a dynamic landing target.
"""

import numpy as np


class MovingPlatform:
    """
    A horizontally moving landing platform with bounded random motion.

    The platform moves along the x-axis with random acceleration changes,
    simulating an unpredictable but physically realistic moving target.
    Motion is constrained by maximum velocity, acceleration, and position bounds.

    Attributes:
        x (float): Current horizontal position in meters.
        v (float): Current horizontal velocity in m/s.
        a (float): Current horizontal acceleration in m/s².
        x_min (float): Minimum allowed x position in meters.
        x_max (float): Maximum allowed x position in meters.
        v_max (float): Maximum allowed velocity magnitude in m/s.
        a_max (float): Maximum allowed acceleration magnitude in m/s².
        dt (float): Simulation time step in seconds.
    """

    def __init__(self, x_min, x_max, v_max, a_max, dt):
        """
        Initialize the moving platform with motion constraints.

        Args:
            x_min (float): Minimum x position bound in meters. Default: -10.0.
            x_max (float): Maximum x position bound in meters. Default: 10.0.
            v_max (float): Maximum velocity magnitude in m/s. Default: 5.0.
            a_max (float): Maximum acceleration magnitude in m/s². Default: 1.0.
            dt (float): Simulation time step in seconds. Default: 0.01.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.v_max = v_max
        self.a_max = a_max
        self.dt = dt

        # Internal State
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0

        # Wandering Logic State
        self.target_v = 0.0
        self.change_timer = 0
        self.change_interval = int(0.2 / dt)  # Change target every 0.2s

    def reset(self):
        """
        Reset the platform to a random initial state.

        Initializes position randomly within bounds, velocity to zero,
        and acceleration to a random value within limits. Resets the
        acceleration change timer.
        """
        self.x = np.random.uniform(self.x_min / 2, self.x_max / 2)
        self.v = np.random.uniform(-self.v_max, self.v_max)
        self.a = np.random.uniform(-self.a_max, self.a_max)
        self.target_v = 0.0
        self.change_timer = 0

    def step(self):
        """
        Advance the platform state by one time step.

        Updates acceleration periodically with random changes, then integrates
        velocity and position. Applies boundary constraints to keep the platform
        within allowed regions, reversing direction at boundaries.
        """
        # 1. Update Timer & Pick New Target
        self.change_timer += 1
        if self.change_timer >= self.change_interval:
            self.target_v = np.random.uniform(-self.v_max, self.v_max)
            self.change_timer = 0

        # 2. Safety Override (Predictive Braking)
        stopping_dist = (self.v**2) / (
            2.0 * self.a_max + 1e-6
        )  # Add small epsilon to avoid division issues
        safe_margin = 0.9 * self.x_max

        # Calculate Proportional Acceleration to reach target
        Kp = 4.0
        desired_accel = Kp * (self.target_v - self.v)

        # Check Bounds - use POSITION not stopping distance alone
        if self.x > safe_margin:
            desired_accel = -self.a_max
            self.target_v = -1.0
        elif self.x < -safe_margin:
            desired_accel = self.a_max
            self.target_v = 1.0
        elif self.v > 0 and (self.x + stopping_dist) > safe_margin:
            desired_accel = -self.a_max
            self.target_v = -1.0
        elif self.v < 0 and (self.x - stopping_dist) < -safe_margin:
            desired_accel = self.a_max
            self.target_v = 1.0

        # 3. Apply Limits & Integrate
        self.a = np.clip(desired_accel, -self.a_max, self.a_max)

        new_v = self.v + self.a * self.dt
        new_v = np.clip(new_v, -self.v_max, self.v_max)

        new_x = self.x + new_v * self.dt

        # HARD CLAMP as final safety net
        new_x = np.clip(new_x, self.x_min, self.x_max)

        self.x = new_x
        self.v = new_v
