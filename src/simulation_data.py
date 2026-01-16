"""
Drone specifications for quadcopter landing simulation. (planar model)
"""

# Mass (kg)
m = 0.724

# Maximum vertical velocity (m/s)
vy_max = 10

# Maximum horizontal velocity (m/s)
vx_max = 21

# Arm length (meters)
arm_length = 0.15

# Maximum thrust (Newtons)
max_thrust = 15.0


"""
Landing pad specifications
"""
# Maximum horizontal velocity of the landing pad (m/s)
vx_pad_max = 5
# Maximum horizontal acceleration of the landing pad (m/s^2)
a_pad_max = 1

"""
Simulation specifications for RL.
"""
# Time step (seconds)
dt = 0.01

# Minimum initial height of the quadcopter (meters)
y_init_min = 20

# Maximum initial height of the quadcopter (meters)
y_init_max = 30

# Maximum initial horizontal position (meters)
x_init_max = 50

# Minimum initial horizontal position (meters)
x_init_min = -50
