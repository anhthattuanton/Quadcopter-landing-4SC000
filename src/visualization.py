import matplotlib.pyplot as plt
import numpy as np

from src.simulation_data import dt, FRAME_SKIP


class QuadcopterVisualizer:
    """
    Visualization class for the Planar Quadcopter environment.
    Handles all rendering, HUD display, and GUI controls.
    """

    def __init__(self, env):
        """
        Initialize the visualizer with a reference to the environment.

        Args:
            env: PlanarQuadcopterEnv instance
        """
        self.env = env
        self.fig = None
        self.ax = None
        self.hud_text = None
        self.trajectory_x = []
        self.trajectory_y = []

        # GUI control flags
        self.paused = False
        self.should_reset = False
        self.should_quit = False

        # Visual settings
        self.visual_scale = 10.0
        self.platform_width = 4.0
        self.platform_height = 0.6
        self.x_limits = (-50, 50)
        self.y_limits = (0, 35)

        # Time tracking
        self.sim_time = 0.0
        self.dt = dt

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

    def _on_close(self, event):
        """Handle window close event."""
        self.should_quit = True

    def reset(self):
        """Reset visualization state (trajectory, flags, time)."""
        self.trajectory_x = []
        self.trajectory_y = []
        self.paused = False
        self.should_reset = False
        self.sim_time = 0.0

    def render(self, state):
        """
        Render the current state of the environment.

        Args:
            state: numpy array with environment state
                   [x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, platform_a]
        """
        # Initialize figure and axis only once
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            self.fig.subplots_adjust(right=0.65)
            plt.ion()
            self.hud_text = None

            # Connect event handlers
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        # Check if figure still exists
        if not plt.fignum_exists(self.fig.number):
            self.should_quit = True
            return

        # Clear the axis every frame
        self.ax.cla()

        # Set axis limits and labels
        self.ax.set_xlim(*self.x_limits)
        self.ax.set_ylim(*self.y_limits)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # Update title to show pause status
        status_str = " [PAUSED]" if self.paused else ""
        self.ax.set_title(f"Planar Quadcopter Landing{status_str}")
        self.ax.grid(True, alpha=0.3)

        # Extract state
        x, y, theta = state[0], state[1], state[2]
        x_dot, y_dot, theta_dot = state[3], state[4], state[5]
        platform_x = state[6]
        platform_v = state[7]
        platform_a = state[8]

        # Store trajectory (only if not paused)
        if not self.paused and (
            len(self.trajectory_x) == 0
            or (self.trajectory_x[-1] != x or self.trajectory_y[-1] != y)
        ):
            self.trajectory_x.append(x)
            self.trajectory_y.append(y)

        # Update simulation time (only if not paused)
        if not self.paused:
            self.sim_time += self.dt * FRAME_SKIP

        # Draw trajectory
        self._draw_trajectory()

        # Draw platform
        self._draw_platform(platform_x)

        # Draw quadcopter
        self._draw_quadcopter(x, y, theta)

        # Draw HUD
        self._draw_hud(
            x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, platform_a
        )

        # Force canvas update and process events
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def _draw_trajectory(self):
        """Draw the drone's trajectory path."""
        if len(self.trajectory_x) > 1:
            self.ax.plot(
                self.trajectory_x,
                self.trajectory_y,
                "g--",
                linewidth=1.5,
                alpha=0.7,
                label="Trajectory",
            )

    def _draw_platform(self, platform_x):
        """Draw the landing platform."""
        platform_left = platform_x - self.platform_width / 2
        platform_rect = plt.Rectangle(
            (platform_left, 0),
            self.platform_width,
            self.platform_height,
            color="red",
            linewidth=2,
        )
        self.ax.add_patch(platform_rect)

    def _draw_quadcopter(self, x, y, theta):
        """Draw the quadcopter with arms, motors, and orientation arrow."""
        visual_arm_length = self.env.arm_length * self.visual_scale

        # Calculate arm endpoints
        left_x = x - visual_arm_length * np.cos(theta)
        left_y = y - visual_arm_length * np.sin(theta)
        right_x = x + visual_arm_length * np.cos(theta)
        right_y = y + visual_arm_length * np.sin(theta)

        # Draw arms
        self.ax.plot([left_x, right_x], [left_y, right_y], "b-", linewidth=4)

        # Draw center dot
        self.ax.plot(x, y, "bo", markersize=12)

        # Draw motor positions
        self.ax.plot(left_x, left_y, "ko", markersize=10)
        self.ax.plot(right_x, right_y, "ko", markersize=10)

        # Draw orientation arrow
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

    def _draw_hud(
        self, x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, platform_a
    ):
        """Draw the Heads-Up Display with state information."""
        theta_deg = np.degrees(theta)

        # Determine status
        if y <= 0.0:
            status = "LANDED/CRASHED"
        elif self.paused:
            status = "PAUSED"
        else:
            status = "FLYING"

        hud_string = (
            f"STATUS: {status}\n"
            f"TIME: {self.sim_time:8.2f} s\n"
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

        # Add text outside the plot area
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

    def close(self):
        """Close the visualization and cleanup."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        plt.ioff()
