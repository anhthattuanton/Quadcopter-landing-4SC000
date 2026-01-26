# Quadcopter configuration parameters
m = 0.724
"""Mass of the quadcopter (kg)
"""

vy_max = 5.0
"""
Maximum vertical velocity of the quadcopter (m/s)
"""

vx_max = 21.0
"""
Maximum horizontal velocity of the quadcopter (m/s)
"""

arm_length = 0.15
"""Length of the quadcopter arms (meters)
"""

max_thrust = 7.0
"""Maximum thrust per rotor (Newtons)

"""

J = 0.011
"""Moment of inertia of the quadcopter (kg*m^2)
"""

# Landing pad configuration parameters
vx_pad_max = 5.0
"""Maximum horizontal velocity of the landing pad (m/s)
"""

a_pad_max = 2.0
"""Maximum horizontal acceleration of the landing pad (m/s^2)
"""

dt = 0.01
"""Time step for the simulation (seconds)
"""

# Initial position bounds
y_init_min = 15.0
"""Minimum initial height of the quadcopter (meters)
"""

y_init_max = 30.0
"""Maximum initial height of the quadcopter (meters)
"""

x_init_max = 50.0
"""Maximum initial horizontal position (meters)
"""

x_init_min = -50.0
"""Minimum initial horizontal position (meters)
"""

FRAME_SKIP = 10
"""Number of frames to skip during visualization rendering"""
