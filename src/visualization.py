"""
Quadcopter Visualization Module.

This module provides real-time visualization of the quadcopter simulation
using matplotlib. It handles rendering of the quadcopter, platform, trajectory,
and heads-up display (HUD) with state information.

Features:
    - Real-time animation with trajectory tracking
    - Interactive controls (pause, reset, quit)
    - HUD showing quadcopter and platform states
    - Scalable quadcopter visualization
"""

import matplotlib.pyplot as plt
import numpy as np

from src.simulation_data import dt


class QuadcopterVisualizer:
    """
    Visualization handler for the planar quadcopter environment.
    
    Manages matplotlib figure, renders environment state, handles user input,
    and displays diagnostic information via HUD.
    
    Attributes:
        env: Reference to PlanarQuadcopterEnv for accessing arm_length.
        fig: Matplotlib figure object.
        ax: Matplotlib axes object.
        hud_text: Text object for HUD display.
        trajectory_x (list): History of x positions for trajectory plotting.
        trajectory_y (list): History of y positions for trajectory plotting.
        paused (bool): True if animation is paused.
        should_reset (bool): True if user requested environment reset.
        should_quit (bool): True if user requested to quit.
        sim_time (float): Accumulated simulation time in seconds.
    """

    def __init__(self, env):
        """
        Initialize the visualizer with environment reference.
        
        Args:
            env: PlanarQuadcopterEnv instance for accessing physical parameters.
        """
        self.env = env
        self.fig = None
        self.ax = None
        self.hud_text = None
        self.trajectory_x = []
        self.trajectory_y = []

        self.paused = False
        self.should_reset = False
        self.should_quit = False

        self.visual_scale = 10.0
        self.platform_width = 4.0
        self.platform_height = 0.6
        self.x_limits = (-50, 50)
        self.y_limits = (0, 35)

        self.sim_time = 0.0
        self.dt = dt

    def _on_key_press(self, event):
        """
        Handle keyboard events for interactive control.
        
        Args:
            event: Matplotlib key press event.
                Space: Toggle pause/resume
                'r': Request environment reset
                'q': Request quit
        """
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
        """Handle window close event by setting quit flag."""
        self.should_quit = True

    def reset(self):
        """Reset visualization state including trajectory and simulation time."""
        self.trajectory_x = []
        self.trajectory_y = []
        self.paused = False
        self.should_reset = False
        self.sim_time = 0.0

    def render(self, state):
        """
        Render the current environment state.
        
        Creates figure if needed, updates all visual elements, and processes
        user input events.
        
        Args:
            state (np.ndarray): Environment state array of shape (9,).
                See PlanarQuadcopterEnv for state format.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            self.fig.subplots_adjust(right=0.65)
            plt.ion()
            self.hud_text = None
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        if not plt.fignum_exists(self.fig.number):
            self.should_quit = True
            return

        if not self.paused:
            self.sim_time += self.dt

        self.ax.cla()

        self.ax.set_xlim(*self.x_limits)
        self.ax.set_ylim(*self.y_limits)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        status_str = " [PAUSED]" if self.paused else ""
        self.ax.set_title(f"Planar Quadcopter Landing{status_str}")
        self.ax.grid(True, alpha=0.3)

        x, y, theta = state[0], state[1], state[2]
        x_dot, y_dot, theta_dot = state[3], state[4], state[5]
        platform_x, platform_v, platform_a = state[6], state[7], state[8]

        if not self.paused and (
            len(self.trajectory_x) == 0
            or (self.trajectory_x[-1] != x or self.trajectory_y[-1] != y)
        ):
            self.trajectory_x.append(x)
            self.trajectory_y.append(y)

        self._draw_trajectory()
        self._draw_platform(platform_x)
        self._draw_quadcopter(x, y, theta)
        self._draw_hud(x, y, theta, x_dot, y_dot, theta_dot,
                       platform_x, platform_v, platform_a)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def _draw_trajectory(self):
        """Draw the drone's flight path as a dashed green line."""
        if len(self.trajectory_x) > 1:
            self.ax.plot(
                self.trajectory_x, self.trajectory_y,
                "g--", linewidth=1.5, alpha=0.7, label="Trajectory"
            )

    def _draw_platform(self, platform_x):
        """
        Draw the landing platform as a red rectangle.
        
        Args:
            platform_x (float): Platform center x position in meters.
        """
        platform_left = platform_x - self.platform_width / 2
        platform_rect = plt.Rectangle(
            (platform_left, 0), self.platform_width, self.platform_height,
            color="red", linewidth=2
        )
        self.ax.add_patch(platform_rect)

    def _draw_quadcopter(self, x, y, theta):
        """
        Draw the quadcopter with arms, motors, and orientation arrow.
        
        Args:
            x (float): Quadcopter x position in meters.
            y (float): Quadcopter y position in meters.
            theta (float): Quadcopter orientation in radians.
        """
        visual_arm_length = self.env.arm_length * self.visual_scale

        left_x = x - visual_arm_length * np.cos(theta)
        left_y = y - visual_arm_length * np.sin(theta)
        right_x = x + visual_arm_length * np.cos(theta)
        right_y = y + visual_arm_length * np.sin(theta)

        self.ax.plot([left_x, right_x], [left_y, right_y], "b-", linewidth=4)
        self.ax.plot(x, y, "bo", markersize=12)
        self.ax.plot(left_x, left_y, "ko", markersize=10)
        self.ax.plot(right_x, right_y, "ko", markersize=10)

        arrow_length = visual_arm_length * 1.2
        arrow_dx = -arrow_length * np.sin(theta)
        arrow_dy = arrow_length * np.cos(theta)
        self.ax.arrow(
            x, y, arrow_dx, arrow_dy,
            head_width=1.2, head_length=0.8,
            fc="orange", ec="darkorange", linewidth=2
        )

    def _draw_hud(self, x, y, theta, x_dot, y_dot, theta_dot,
                  platform_x, platform_v, platform_a):
        """
        Draw the heads-up display with state information.
        
        Args:
            x, y, theta: Quadcopter position and orientation.
            x_dot, y_dot, theta_dot: Quadcopter velocities.
            platform_x, platform_v, platform_a: Platform state.
        """
        theta_deg = np.degrees(theta)

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

        if self.hud_text is not None:
            self.hud_text.remove()

        self.hud_text = self.fig.text(
            0.68, 0.95, hud_string,
            transform=self.fig.transFigure,
            fontsize=10, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.9)
        )

    def close(self):
        """Close the visualization window and cleanup resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        plt.ioff()
