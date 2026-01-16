m = 0.724
"""Mass of the quadcopter (kg)
"""

vy_max = int(10)
"""
Maximum vertical velocity of the quadcopter (m/s)
"""

vx_max = int(21)
"""
Maximum horizontal velocity of the quadcopter (m/s)
"""

arm_length = float(0.15)
"""Length of the quadcopter arms (meters)
"""

max_thrust = float(15.0)
"""Maximum thrust per rotor (Newtons)

"""

vx_pad_max = int(5) 
"""Maximum horizontal velocity of the landing pad (m/s)
"""

a_pad_max = int(1)
"""Maximum horizontal acceleration of the landing pad (m/s^2)
"""

dt = float(0.01)
"""Time step for the simulation (seconds)
"""


y_init_min = int(20)   
"""Minimum initial height of the quadcopter (meters)
""" 

y_init_max = int(30)
"""Maximum initial height of the quadcopter (meters)
"""

x_init_max = int(50)
"""Maximum initial horizontal position (meters)
"""

x_init_min = int(-50)
"""Minimum initial horizontal position (meters)
"""