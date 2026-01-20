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
    x, y, theta, x_dot, y_dot, theta_dot = state[0:6]

    # 1. Distance components
    dist_x = np.abs(x - platform_x)
    dist_y = y
    dist_total = np.sqrt(dist_x**2 + dist_y**2)

    # 2. Velocity components
    # We want velocity to be 0 ONLY when we are landing.
    # While flying, we only care about stability (theta_dot).
    vel_total = np.sqrt(x_dot**2 + y_dot**2)

    # --- SUCCESS FLAGS ---
    is_on_pad = dist_x < 0.5
    is_upright = np.abs(theta) < 0.2
    is_slow = vel_total < 1.0  # Relaxed from 0.1 (too strict)
    is_landed = dist_y <= 0.0

    reward = 0.0
    terminated = False
    info = {}

    # --- A. TERMINAL REWARDS (The Big Events) ---
    if is_landed:
        terminated = True
        if is_on_pad:
            if is_upright and is_slow:
                reward = 100.0  # 🏆 PERFECT LANDING
            elif is_upright:
                reward = 40.0  # Good landing
            elif is_slow:
                reward = 50.0  # Hard landing
            else:
                reward = 20.0  # Crash landing
        else:
            reward = -100.0  # Crashed into ground (The Floor is Lava)

    elif np.abs(x) > x_init_max or y > y_init_max:
        terminated = True
        reward = -100.0  # Flew away (Cowardice)

    # --- B. SHAPING REWARDS (The Breadcrumbs) ---
    else:
        # 1. Position Reward (Exponential)
        # 1.0 when on target, 0.0 when far away.
        # Much better than linear subtraction.
        r_pos = np.exp(-1.0 * dist_total)

        # 2. Stability Reward (Keep it upright)
        # 1.0 when flat, 0.0 when 90 degrees.
        r_stab = np.exp(-5.0 * np.abs(theta))

        # 3. Action cost (Efficiency)
        r_act = -0.01 * np.sum(action**2)

        # 4. Survival/Time Reward
        # We give a tiny POSITIVE reward for staying alive and stable
        # This prevents it from diving into the ground to end the episode.
        r_survive = 0.05

        # Combine
        # Weighted sum: Position is king.
        reward = (5.0 * r_pos) + (3.0 * r_stab) + r_act + r_survive

    # Info dict for debugging
    info = {"dist_total": dist_total, "is_on_pad": is_on_pad, "vel_total": vel_total}

    return reward, terminated, info
