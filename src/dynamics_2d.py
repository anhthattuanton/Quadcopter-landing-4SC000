"""
2D Planar Quadcopter Dynamics Module.

This module implements the physics simulation for a planar (2D) quadcopter,
including force computation, Newton-Euler dynamics, and numerical integration.

The quadcopter is modeled with:
- 3 degrees of freedom: x (horizontal), y (vertical), theta (rotation)
- 2 motors producing vertical thrust forces
- Rigid body dynamics with moment of inertia
"""

import numpy as np

from src.simulation_data import J, arm_length, dt, m, max_thrust


class PlanarQuadcopterDynamics:
    """
    Physics simulation for a 2D planar quadcopter.
    
    This class handles the complete dynamics simulation including:
    - Converting normalized control inputs to motor forces
    - Computing accelerations using Newton-Euler equations
    - Integrating state using semi-implicit Euler method
    
    Attributes:
        dt (float): Simulation time step in seconds.
        gravity (float): Gravitational acceleration in m/s².
        mass (float): Quadcopter mass in kg.
        arm_length (float): Distance from center to motor in meters.
        max_thrust (float): Maximum thrust per motor in Newtons.
        J (float): Moment of inertia in kg·m².
    """

    def __init__(self):
        """Initialize dynamics with parameters from simulation_data."""
        self.dt = dt
        self.gravity = 9.81
        self.mass = m
        self.arm_length = arm_length
        self.max_thrust = max_thrust
        self.J = J

    def compute_forces(self, action):
        """
        Convert normalized action inputs to actual motor thrust forces.
        
        The action space is [-1, 1] which maps to [0, max_thrust] for each motor.
        
        Args:
            action (np.ndarray): Array of shape (2,) with normalized motor commands.
                action[0]: Left motor command in range [-1, 1].
                action[1]: Right motor command in range [-1, 1].
        
        Returns:
            tuple: (F_left, F_right) motor forces in Newtons.
        """
        action = np.clip(action, -1.0, 1.0)
        F_left = (action[0] + 1.0) * (self.max_thrust / 2.0)
        F_right = (action[1] + 1.0) * (self.max_thrust / 2.0)
        return F_left, F_right

    def compute_accelerations(self, theta, F_left, F_right):
        """
        Compute linear and angular accelerations using Newton-Euler equations.
        
        The equations of motion for a planar quadcopter are:
        - F_total = F_left + F_right (total thrust)
        - M = (F_right - F_left) * L (moment about center)
        - a_x = -(F_total / m) * sin(theta)
        - a_y = (F_total / m) * cos(theta) - g
        - alpha = M / J
        
        Args:
            theta (float): Current orientation angle in radians.
            F_left (float): Left motor thrust force in Newtons.
            F_right (float): Right motor thrust force in Newtons.
        
        Returns:
            tuple: (accel_x, accel_y, accel_theta) accelerations.
                accel_x (float): Horizontal acceleration in m/s².
                accel_y (float): Vertical acceleration in m/s².
                accel_theta (float): Angular acceleration in rad/s².
        """
        F_total = F_left + F_right
        moment = (F_right - F_left) * self.arm_length

        accel_x = -(F_total / self.mass) * np.sin(theta)
        accel_y = (F_total / self.mass) * np.cos(theta) - self.gravity
        accel_theta = moment / self.J

        return accel_x, accel_y, accel_theta

    def integrate(self, x, y, theta, x_dot, y_dot, theta_dot, 
                  accel_x, accel_y, accel_theta):
        """
        Integrate state forward using semi-implicit Euler method.
        
        Semi-implicit Euler updates velocities first, then uses the new
        velocities to update positions. This provides better energy
        conservation than explicit Euler.
        
        Args:
            x (float): Current horizontal position in meters.
            y (float): Current vertical position in meters.
            theta (float): Current orientation angle in radians.
            x_dot (float): Current horizontal velocity in m/s.
            y_dot (float): Current vertical velocity in m/s.
            theta_dot (float): Current angular velocity in rad/s.
            accel_x (float): Horizontal acceleration in m/s².
            accel_y (float): Vertical acceleration in m/s².
            accel_theta (float): Angular acceleration in rad/s².
        
        Returns:
            tuple: Updated state (x, y, theta, x_dot, y_dot, theta_dot).
                All positions in meters, velocities in m/s or rad/s.
                theta is normalized to [-π, π].
        """
        x_dot_new = x_dot + accel_x * self.dt
        y_dot_new = y_dot + accel_y * self.dt
        theta_dot_new = theta_dot + accel_theta * self.dt

        x_new = x + x_dot_new * self.dt
        y_new = y + y_dot_new * self.dt
        theta_new = theta + theta_dot_new * self.dt

        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        return x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new

    def step(self, state, action):
        """
        Perform one complete simulation step.
        
        Combines force computation, acceleration calculation, and integration
        into a single method call.
        
        Args:
            state (np.ndarray): Current state array of shape (9,).
                state[0]: x position (m)
                state[1]: y position (m)
                state[2]: theta orientation (rad)
                state[3]: x velocity (m/s)
                state[4]: y velocity (m/s)
                state[5]: theta angular velocity (rad/s)
                state[6:9]: Platform state (unused in dynamics)
            action (np.ndarray): Normalized motor commands of shape (2,).
        
        Returns:
            tuple: Updated drone state (x, y, theta, x_dot, y_dot, theta_dot).
        """
        x, y, theta, x_dot, y_dot, theta_dot = state[0:6]

        F_left, F_right = self.compute_forces(action)
        accel_x, accel_y, accel_theta = self.compute_accelerations(theta, F_left, F_right)
        
        return self.integrate(
            x, y, theta, x_dot, y_dot, theta_dot, 
            accel_x, accel_y, accel_theta
        )
