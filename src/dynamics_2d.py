import numpy as np

from src.simulation_data import J, arm_length, dt, m, max_thrust


class PlanarQuadcopterDynamics:
    """
    2D Planar Quadcopter Dynamics.
    Handles physics simulation for a quadcopter in 2D (x, y, theta).
    """

    def __init__(self):
        self.dt = dt
        self.gravity = 9.81
        self.mass = m
        self.arm_length = arm_length
        self.max_thrust = max_thrust
        self.J = J

    def compute_forces(self, action):
        """
        Convert normalized action [-1, 1] to actual motor forces.

        Args:
            action: numpy array [left_action, right_action] in range [-1, 1]

        Returns:
            F_left, F_right: motor forces in Newtons
        """
        action = np.clip(action, -1.0, 1.0)
        F_left = (action[0] + 1.0) * (self.max_thrust / 2.0)
        F_right = (action[1] + 1.0) * (self.max_thrust / 2.0)
        return F_left, F_right

    def compute_accelerations(self, theta, F_left, F_right):
        """
        Compute accelerations using Newton-Euler equations.

        Args:
            theta: current orientation angle (radians)
            F_left: left motor force (N)
            F_right: right motor force (N)

        Returns:
            accel_x, accel_y, accel_theta: accelerations
        """
        F_total = F_left + F_right
        moment = (F_right - F_left) * self.arm_length

        accel_x = -(F_total / self.mass) * np.sin(theta)
        accel_y = (F_total / self.mass) * np.cos(theta) - self.gravity
        accel_theta = moment / self.J

        return accel_x, accel_y, accel_theta

    def integrate(
        self, x, y, theta, x_dot, y_dot, theta_dot, accel_x, accel_y, accel_theta
    ):
        """
        Semi-Implicit Euler integration.

        Args:
            x, y, theta: current positions
            x_dot, y_dot, theta_dot: current velocities
            accel_x, accel_y, accel_theta: current accelerations

        Returns:
            x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new: updated state
        """
        # Update velocities first
        x_dot_new = x_dot + accel_x * self.dt
        y_dot_new = y_dot + accel_y * self.dt
        theta_dot_new = theta_dot + accel_theta * self.dt

        # Update positions
        x_new = x + x_dot_new * self.dt
        y_new = y + y_dot_new * self.dt
        theta_new = theta + theta_dot_new * self.dt

        # Normalize theta to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        return x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new

    def step(self, state, action):
        """
        Perform one simulation step.

        Args:
            state: numpy array [x, y, theta, x_dot, y_dot, theta_dot, ...]
            action: numpy array [left_action, right_action] in range [-1, 1]

        Returns:
            x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new: updated drone state
        """
        # Extract current drone state
        x, y, theta, x_dot, y_dot, theta_dot = state[0:6]

        # Compute forces
        F_left, F_right = self.compute_forces(action)

        # Compute accelerations
        accel_x, accel_y, accel_theta = self.compute_accelerations(
            theta, F_left, F_right
        )

        # Integrate
        x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new = self.integrate(
            x, y, theta, x_dot, y_dot, theta_dot, accel_x, accel_y, accel_theta
        )

        return x_new, y_new, theta_new, x_dot_new, y_dot_new, theta_dot_new
