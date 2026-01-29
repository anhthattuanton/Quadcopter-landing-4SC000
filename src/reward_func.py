"""
Reward Function Module for Quadcopter Landing.

This module implements a phased reward strategy that separates approach
behavior from landing behavior, solving the common RL problem of conflicting
objectives (e.g., "go fast" vs "stay stable").

Phases:
    1. Travel Phase (dist > 4m): Aggressive flight towards target.
       Rewards velocity towards platform, allows banking for speed.
    
    2. Landing Phase (dist <= 4m): Precision landing.
       Rewards position accuracy, enforces stability, encourages braking.
"""

import numpy as np

from src.simulation_data import x_init_max, y_init_max


def calculate_reward(state):
    """
    Calculate reward using a phased approach strategy.
    
    The reward function uses different objectives based on distance to target:
    - Far away: Reward approach velocity, allow aggressive maneuvering
    - Close: Reward position accuracy, enforce stability and slow descent
    
    Args:
        state (np.ndarray): Environment state array of shape (9,).
            state[0]: x - Horizontal position (m)
            state[1]: y - Vertical position (m)
            state[2]: theta - Orientation angle (rad)
            state[3]: x_dot - Horizontal velocity (m/s)
            state[4]: y_dot - Vertical velocity (m/s)
            state[5]: theta_dot - Angular velocity (rad/s)
            state[6]: platform_x - Platform horizontal position (m)
            state[7]: platform_v - Platform horizontal velocity (m/s)
            state[8]: platform_a - Platform horizontal acceleration (m/s²)
    
    Returns:
        tuple: (reward, terminated, info)
            reward (float): Scalar reward value.
            terminated (bool): True if episode has ended.
            info (dict): Dictionary containing diagnostic metrics:
                - dist_x: Signed horizontal distance to platform (m)
                - dist_y: Vertical distance to ground (m)
                - dist_total: Euclidean distance to platform (m)
                - vx_rel: Horizontal velocity relative to platform (m/s)
                - vy_rel: Vertical velocity (m/s)
                - vel_total: Total relative velocity magnitude (m/s)
                - is_on_pad: Boolean, True if horizontally aligned with pad
    """
    x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, _ = state

    dist_x = x - platform_x
    dist_y = y
    dist_total = np.sqrt(dist_x**2 + dist_y**2)

    vx_rel = x_dot - platform_v
    vy_rel = y_dot
    vel_rel_total = np.sqrt(vx_rel**2 + vy_rel**2)

    PHASE_THRESHOLD = 4.0
    is_travel_phase = dist_total > PHASE_THRESHOLD

    is_on_pad = np.abs(dist_x) < 0.5
    is_upright = np.abs(theta) < 0.2
    is_soft = vel_rel_total < 1.0
    is_landed = y <= 0.0

    reward = 0.0
    terminated = False

    if is_landed:
        terminated = True
        if is_on_pad and is_upright and is_soft:
            reward = 100.0
        elif is_on_pad and is_upright:
            reward = 50.0
        elif is_on_pad:
            reward = 10.0
        else:
            reward = -50.0
        return reward, terminated, _build_info(
            dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad
        )

    if np.abs(x) > x_init_max or y > y_init_max:
        terminated = True
        reward = -50.0
        return reward, terminated, _build_info(
            dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad
        )

    reward += 0.1

    if is_travel_phase:
        if dist_total > 0.01:
            dir_to_target_x = -dist_x / dist_total
            dir_to_target_y = -dist_y / dist_total
        else:
            dir_to_target_x = 0.0
            dir_to_target_y = -1.0

        approach_velocity = (dir_to_target_x * vx_rel) + (dir_to_target_y * vy_rel)
        reward += 0.5 * approach_velocity

        if np.abs(theta) > 0.8:
            reward += -2.0 * (np.abs(theta) - 0.8)

        reward += -0.1 * np.abs(theta_dot)

    else:
        reward += 2.0 * np.exp(-dist_total)

        if np.abs(theta) > 0.1:
            reward += -3.0 * np.abs(theta)

        reward += -0.5 * vel_rel_total
        reward += -0.5 * np.abs(theta_dot)

        if vy_rel < -2.0:
            reward += -1.0 * (np.abs(vy_rel) - 2.0)

    info = _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad)
    return reward, terminated, info


def _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad):
    """
    Build the info dictionary with diagnostic metrics.
    
    Args:
        dist_x (float): Signed horizontal distance to platform (m).
        dist_y (float): Vertical distance to ground (m).
        dist_total (float): Euclidean distance to landing target (m).
        vx_rel (float): Horizontal velocity relative to platform (m/s).
        vy_rel (float): Vertical velocity (m/s).
        vel_rel_total (float): Total relative velocity magnitude (m/s).
        is_on_pad (bool): True if drone is horizontally within pad bounds.
    
    Returns:
        dict: Information dictionary for logging and debugging.
    """
    return {
        "dist_x": dist_x,
        "dist_y": dist_y,
        "dist_total": dist_total,
        "vx_rel": vx_rel,
        "vy_rel": vy_rel,
        "vel_total": vel_rel_total,
        "is_on_pad": is_on_pad,
    }
