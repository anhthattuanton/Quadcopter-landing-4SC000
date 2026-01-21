import numpy as np

from src.simulation_data import x_init_max, y_init_max


def calculate_reward(state, action, platform_x):
    """
    Calculate the reward for the current state and action.

    Args:
        state: numpy array [x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, platform_a]
        action: numpy array [left_thrust, right_thrust] normalized to [-1, 1]
        platform_x: current x position of the platform

    Returns:
        reward (float): the calculated reward
        terminated (bool): whether the episode has ended
        info (dict): additional information
    """
    # Unpack state
    x, y, theta, x_dot, y_dot, theta_dot, _, platform_vx, _ = state

    # 1. Distance components
    dist_x = np.abs(x - platform_x)
    dist_y = y
    dist_total = np.sqrt(dist_x**2 + dist_y**2)

    # 2. Velocity components
    vel_total = np.sqrt(x_dot**2 + y_dot**2)

    # 3. Relative velocity (how well drone matches platform motion)
    relative_vx = np.abs(x_dot - platform_vx)

    # --- SUCCESS FLAGS ---
    is_on_pad = dist_x < 0.5
    is_upright = np.abs(theta) < 0.2
    is_slow = vel_total < 1.0
    is_landed = dist_y <= 0.1  # Small tolerance

    reward = 0.0
    terminated = False

    # --- A. TERMINAL REWARDS ---
    if is_landed:
        terminated = True
        if is_on_pad:
            if is_upright and is_slow:
                reward = 500.0  # 🏆 PERFECT LANDING
            elif is_upright:
                reward = 200.0  # Good landing
            elif is_slow:
                reward = 150.0  # Hard landing
            else:
                reward = 50.0  # Crash landing on pad
        else:
            reward = -200.0  # Crashed into ground

    elif np.abs(x) > x_init_max or y > y_init_max:
        terminated = True
        reward = -200.0  # Flew away

    # --- B. SHAPING REWARDS ---
    else:
        # 1. Horizontal tracking reward (CRITICAL for moving platform)
        # Use softer decay so drone gets gradient from far away
        r_horizontal = np.exp(-0.1 * dist_x)  # Slower decay

        # 2. Vertical progress reward (encourage descent)
        # Reward for being lower AND for descending
        r_altitude = np.exp(-0.05 * dist_y)  # Reward being close to ground
        r_descend = (
            -0.1 * y_dot if y_dot > 0 else 0.05
        )  # Punish going up, reward going down

        # 3. Velocity matching reward (match platform velocity)
        r_vel_match = np.exp(-0.5 * relative_vx)

        # 4. Stability reward
        r_stab = np.exp(-3.0 * np.abs(theta))
        r_ang_vel = -0.1 * np.abs(theta_dot)  # Penalize spinning

        # 5. Action cost
        r_act = -0.005 * np.sum(action**2)

        # 6. Time penalty (encourage faster landing)
        r_time = -0.01

        # Combine with weights
        reward = (
            3.0 * r_horizontal  # Track platform X
            + 2.0 * r_altitude  # Get close to ground
            + 1.0 * r_descend  # Move downward
            + 2.0 * r_vel_match  # Match platform speed
            + 1.0 * r_stab  # Stay upright
            + r_ang_vel
            + r_act
            + r_time
        )

    info = {
        "dist_total": dist_total,
        "dist_x": dist_x,
        "is_on_pad": is_on_pad,
        "vel_total": vel_total,
        "relative_vx": relative_vx,
    }

    return reward, terminated, info
