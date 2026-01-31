"""
Simulation Parameters and Constants for the 2D Planar Quadcopter Environment.

This module contains all configurable parameters for:
- Quadcopter physical properties (mass, inertia, dimensions)
- Motion constraints (velocity limits, thrust limits)
- Landing pad properties (velocity, acceleration limits)
- Simulation settings (time step, initial position bounds)
- Visualization settings (frame skip for playback)

All values use SI units (meters, seconds, kilograms, Newtons).
"""

# === QUADCOPTER PHYSICAL PROPERTIES ===

m = 0.724
"""float: Mass of the quadcopter in kilograms (kg)."""

J = 0.011
"""float: Moment of inertia of the quadcopter about the z-axis in kg·m²."""

arm_length = 0.15
"""float: Length of each quadcopter arm from center to motor in meters (m)."""

max_thrust = 7.0
"""float: Maximum thrust force per motor in Newtons (N)."""

# === QUADCOPTER MOTION CONSTRAINTS ===

vx_max = 21.0
"""float: Maximum horizontal velocity of the quadcopter in m/s."""

vy_max = 5.0
"""float: Maximum vertical velocity of the quadcopter in m/s."""

# === LANDING PAD PROPERTIES ===

vx_pad_max = 5.0
"""float: Maximum horizontal velocity of the landing pad in m/s."""

a_pad_max = 2.0
"""float: Maximum horizontal acceleration of the landing pad in m/s²."""

# === SIMULATION SETTINGS ===

dt = 0.01
"""float: Simulation time step in seconds (s). Smaller values increase accuracy but slow simulation."""

# === INITIAL POSITION BOUNDS ===

y_init_min = 15.0
"""float: Minimum initial height of the quadcopter in meters (m)."""

y_init_max = 30.0
"""float: Maximum initial height of the quadcopter in meters (m)."""

x_init_max = 50.0
"""float: Maximum initial horizontal position in meters (m). Defines right boundary."""

x_init_min = -50.0
"""float: Minimum initial horizontal position in meters (m). Defines left boundary."""

# === VISUALIZATION SETTINGS ===

FRAME_SKIP = 10
"""int: Number of simulation frames to skip between rendered frames for faster playback.
Higher values speed up playback but reduce smoothness."""
