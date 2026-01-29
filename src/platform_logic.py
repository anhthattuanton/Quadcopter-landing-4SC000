import numpy as np


class MovingPlatform:
    def __init__(self, x_min, x_max, v_max, a_max, dt):
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
        self.x = np.random.uniform(self.x_min / 2, self.x_max / 2)
        self.v = np.random.uniform(-self.v_max, self.v_max)
        self.a = np.random.uniform(-self.a_max, self.a_max)
        self.target_v = 0.0
        self.change_timer = 0

    def step(self):
        """Calculates the next position safely."""
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
