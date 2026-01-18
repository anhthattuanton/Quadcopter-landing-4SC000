import gymnasium as gym
import matplotlib.pyplot as plt
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
        # Actions: [Main Thrust, Differential Thrust] or [Motor1, Motor2]
        # Using continuous values between -1 and 1
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

        # GUI control flags
        self.paused = False
        self.should_reset = False
        self.should_quit = False

    def _on_key_press(self, event):
        """Handle keyboard events for GUI controls."""
        if event.key == " ":
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RUNNING"
            print(f"Simulation {status}")
        elif event.key == "r":
            self.should_reset = True
            self.paused = False
        elif event.key == "q":
            self.should_quit = True

    def reset(self, seed=None, options=None):
        # 1. Setup the Random Number Generator (This is all super() does!)
        super().reset(seed=seed)

        # 2. Create a blank state
        self.state = np.zeros(9, dtype=np.float32)

        for n in range(len(self.state)):
            if n == 1:
                self.state[n] = self.np_random.uniform(y_init_min, y_init_max)
            elif n == 6:
                self.platform.reset()
                # Sync state
                self.state[6] = self.platform.x
                self.state[7] = self.platform.v
                self.state[8] = self.platform.a
            elif n != 7 and n != 8:
                self.state[n] = self.np_random.uniform(
                    low=self.observation_space.low[n] / 4,
                    high=self.observation_space.high[n] / 4,
                )

        # Reset internal logic variables
        self.platform_accel_counter = 0

        # Reset trajectory history
        self.trajectory_x = []
        self.trajectory_y = []

        # Reset control flags (but keep should_quit)
        self.paused = False
        self.should_reset = False

        return self.state, {}

    def step(self, action):
        """
        Executes one time step within the environment.

        CONTEXT:
        - This is a Planar Quadcopter (2D) defined by x, y, theta.
        - State Vector (8 elements): [x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_vx]
        - Action Space: Continuous [-1, 1] representing [Left_Motor_Thrust, Right_Motor_Thrust].
        - Simulation dt: 0.02 seconds.
        - Physics Constants: m=1.0kg, g=9.81, L=0.25m (arm length), I=0.05 (inertia).

        TASKS TO IMPLEMENT:
        1. Action Handling:
           - Clamp actions to [-1, 1].
           - Scale actions: Real Force = (action + 1) * (max_thrust / 2). Assumes max_thrust = 10N.

        2. Calculate Dynamics (Newton-Euler Equations):
           - Total Force F = F_left + F_right
           - Moment M = (F_right - F_left) * L
           - Accel X = -(F / m) * sin(theta)
           - Accel Y = (F / m) * cos(theta) - g
           - Angular Accel = M / I

        3. Integration (Semi-Implicit Euler):
           - Update velocities first: v_new = v_old + a * dt
           - Update positions second: p_new = p_old + v_new * dt
           - Update theta ensuring it stays within reasonable bounds (-pi to pi ideally, but not strictly necessary for local control).

        4. Platform Logic:
           - Update platform_x based on a sine wave function: center + amplitude * sin(frequency * time).
           - Update platform_vx (derivative of the sine wave).

        5. Termination & Rewards:
           - Calculate distance_to_platform.
           - CRASH: If y <= 0.0:
             - If distance_to_platform < 0.2 and velocity < 1.0: REWARD = +100 (Landed), Terminated=True
             - Else: REWARD = -100 (Crashed), Terminated=True
           - BOUNDS: If abs(x) > 20 or y > 20: REWARD = -50, Terminated=True
           - STEP REWARD: -0.01 * distance_to_platform (Dense reward to guide agent).

        Returns:
            observation (np.array), reward (float), terminated (bool), truncated (bool), info (dict)
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

        # 4. Platform Logic (Random Walk with Bounds)
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

        # 5. Termination & Rewards
        distance_to_platform = np.abs(x_new - self.platform.x)
        velocity_magnitude = np.sqrt(x_dot_new**2 + y_dot_new**2)

        terminated = False
        truncated = False

        # Check if quadcopter hit the ground
        if y_new <= 0.0:
            terminated = True

        info = {
            "distance_to_platform": distance_to_platform,
            "velocity": velocity_magnitude,
            "platform_x": self.platform.x,
        }

        return self.state, {}, terminated, truncated, info

    def render(self):
        # Initialize figure and axis only once
        if not hasattr(self, "fig") or self.fig is None:
            # Create figure with extra space on the right for HUD
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            self.fig.subplots_adjust(right=0.65)  # Make room for HUD on the right
            plt.ion()  # Enable interactive mode
            self.hud_text = None  # Initialize HUD text object

            # Connect keyboard event handler
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

            # Connect window close event handler
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        # Check if figure still exists
        if not plt.fignum_exists(self.fig.number):
            self.should_quit = True
            return

        # Initialize trajectory lists if they don't exist
        if not hasattr(self, "trajectory_x"):
            self.trajectory_x = []
            self.trajectory_y = []

        # Clear the axis every frame
        self.ax.cla()

        # Set axis limits
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(0, 35)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # Update title to show pause status
        status_str = " [PAUSED]" if self.paused else ""
        self.ax.set_title(f"Planar Quadcopter Landing{status_str}")
        self.ax.grid(True, alpha=0.3)

        # Extract state
        x, y, theta = self.state[0], self.state[1], self.state[2]
        x_dot, y_dot, theta_dot = self.state[3], self.state[4], self.state[5]
        platform_x = self.state[6]
        platform_v = self.state[7]
        platform_a = self.state[8]

        # Store trajectory (only if not paused)
        if not self.paused and (
            len(self.trajectory_x) == 0
            or (self.trajectory_x[-1] != x or self.trajectory_y[-1] != y)
        ):
            self.trajectory_x.append(x)
            self.trajectory_y.append(y)

        # Draw the trajectory (dashed line)
        if len(self.trajectory_x) > 1:
            self.ax.plot(
                self.trajectory_x,
                self.trajectory_y,
                "g--",
                linewidth=1.5,
                alpha=0.7,
                label="Trajectory",
            )

        # Draw the Platform (Red rectangle at y=0) - enlarged
        platform_width = 4.0
        platform_height = 0.6
        platform_left = platform_x - platform_width / 2
        platform_rect = plt.Rectangle(
            (platform_left, 0),
            platform_width,
            platform_height,
            color="red",
            linewidth=2,
        )
        self.ax.add_patch(platform_rect)

        # Draw the Quadcopter (scaled up for visibility)
        visual_scale = 10.0  # Scale factor for visualization
        visual_arm_length = self.arm_length * visual_scale

        # Calculate arm endpoints using trigonometry
        # Left arm endpoint
        left_x = x - visual_arm_length * np.cos(theta)
        left_y = y - visual_arm_length * np.sin(theta)
        # Right arm endpoint
        right_x = x + visual_arm_length * np.cos(theta)
        right_y = y + visual_arm_length * np.sin(theta)

        # Draw arms as blue line
        self.ax.plot([left_x, right_x], [left_y, right_y], "b-", linewidth=4)

        # Draw center dot
        self.ax.plot(x, y, "bo", markersize=12)

        # Draw motor positions (small circles at arm ends)
        self.ax.plot(left_x, left_y, "ko", markersize=10)
        self.ax.plot(right_x, right_y, "ko", markersize=10)

        # Draw orientation arrow (perpendicular to arms, pointing "up" relative to drone)
        arrow_length = visual_arm_length * 1.2
        arrow_dx = -arrow_length * np.sin(theta)
        arrow_dy = arrow_length * np.cos(theta)
        self.ax.arrow(
            x,
            y,
            arrow_dx,
            arrow_dy,
            head_width=1.2,
            head_length=0.8,
            fc="orange",
            ec="darkorange",
            linewidth=2,
        )

        # Add HUD (Heads-Up Display) outside the plot area
        theta_deg = np.degrees(theta)

        # Determine landing status
        if y <= 0.0:
            status = "LANDED/CRASHED"
        elif self.paused:
            status = "PAUSED"
        else:
            status = "FLYING"

        hud_string = (
            f"STATUS: {status}\n"
            f"\n"
            f"QUADCOPTER STATE:\n"
            f"──────────────────────\n"
            f"  Position:\n"
            f"    x  = {x:8.2f} m\n"
            f"    y  = {y:8.2f} m\n"
            f"    θ  = {theta_deg:8.2f}°\n"
            f"\n"
            f"  Velocity:\n"
            f"    vx = {x_dot:8.2f} m/s\n"
            f"    vy = {y_dot:8.2f} m/s\n"
            f"    ω  = {theta_dot:8.2f} rad/s\n"
            f"\n"
            f"PLATFORM STATE:\n"
            f"──────────────────────\n"
            f"  x = {platform_x:8.2f} m\n"
            f"  v = {platform_v:8.2f} m/s\n"
            f"  a = {platform_a:8.2f} m/s²\n"
            f"\n"
            f"CONTROLS:\n"
            f"──────────────────────\n"
            f"  SPACE - Pause/Resume\n"
            f"  R     - Reset\n"
            f"  Q     - Quit"
        )

        # Remove old HUD text if it exists
        if self.hud_text is not None:
            self.hud_text.remove()

        # Add text outside the plot area using figure coordinates
        self.hud_text = self.fig.text(
            0.68,
            0.95,
            hud_string,
            transform=self.fig.transFigure,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.9),
        )

        # Force canvas update and process events
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def _on_close(self, event):
        """Handle window close event."""
        self.should_quit = True

    def close(self):
        """Close the environment and cleanup."""
        if hasattr(self, "fig") and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        plt.ioff()  # Disable interactive mode
