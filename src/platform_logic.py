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

    def __init__(self, x_min=-10.0, x_max=10.0, v_max=5.0, a_max=1.0, dt=0.01):
        """
        Initialize the moving platform.
        
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

        self.x = 0.0
        self.v = 0.0
        self.a = 0.0

        self._accel_change_timer = 0
        self._accel_change_interval = int(0.5 / dt)

    def reset(self):
        """
        Reset the platform to a random initial state.
        
        Initializes position randomly within bounds, velocity to zero,
        and acceleration to a random value within limits.
        """
        self.x = np.random.uniform(self.x_min, self.x_max)
        self.v = 0.0
        self.a = np.random.uniform(-self.a_max, self.a_max)
        self._accel_change_timer = 0

    def step(self):
        """
        Advance the platform state by one time step.
        
        Updates acceleration periodically with random changes, then integrates
        velocity and position. Applies boundary constraints to keep the platform
        within allowed regions.
        """
        self._accel_change_timer += 1
        if self._accel_change_timer >= self._accel_change_interval:
            self.a = np.random.uniform(-self.a_max, self.a_max)
            self._accel_change_timer = 0

        self.v += self.a * self.dt
        self.v = np.clip(self.v, -self.v_max, self.v_max)

        self.x += self.v * self.dt

        if self.x <= self.x_min:
            self.x = self.x_min
            self.v = abs(self.v)
            self.a = abs(self.a)
        elif self.x >= self.x_max:
            self.x = self.x_max
            self.v = -abs(self.v)
            self.a = -abs(self.a)
