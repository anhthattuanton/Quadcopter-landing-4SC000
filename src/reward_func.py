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
    - Far away (Travel Phase): Reward approach velocity, allow aggressive maneuvering
    - Close (Landing Phase): Reward position accuracy, enforce stability and slow descent

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

    Terminal Rewards:
        - Perfect landing (on pad, upright, soft): +100
        - Good landing (on pad, upright, hard): +50
        - Crash landing (on pad, tilted): +10
        - Missed pad: -50
        - Out of bounds: -50

    Shaping Rewards (Travel Phase):
        - Approach velocity reward: +0.5 * dot(velocity, direction_to_target)
        - Extreme tilt penalty: -2.0 * (|theta| - 0.8) if |theta| > 0.8
        - Spin penalty: -0.1 * |theta_dot|

    Shaping Rewards (Landing Phase):
        - Position reward: +2.0 * exp(-dist_total)
        - Tilt penalty: -3.0 * |theta| if |theta| > 0.1
        - Braking penalty: -0.5 * vel_rel_total
        - Spin penalty: -0.5 * |theta_dot|
        - Fast descent penalty: -1.0 * (|vy_rel| - 2.0) if vy_rel < -2.0

    Global:
        - Survival bonus: +0.1 per step
    """
    x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, _ = state

    dist_x = x - platform_x
    dist_y = y
    dist_total = np.sqrt(dist_x**2 + dist_y**2)

    v_total = np.sqrt(x_dot**2 + y_dot**2)
    vx_rel = x_dot - platform_v
    vy_rel = y_dot
    vel_rel_total = np.sqrt(vx_rel**2 + vy_rel**2)

    PHASE_THRESHOLD = 8.0
    is_travel_phase = dist_total > PHASE_THRESHOLD
    is_stable = (np.abs(theta) < 0.2) and (v_total < 1.0) and (np.abs(theta_dot) < 0.2)

    is_on_pad = np.abs(dist_x) < 0.5
    is_upright = np.abs(theta) < 0.2
    is_soft = vel_rel_total < 0.1
    is_landed = y <= 0.0

    reward = 0.0
    terminated = False

    if is_landed:
        terminated = True
        if is_on_pad and is_upright and is_soft:
            # Perfect landing: on pad, upright, and soft
            reward = 700.0
        elif is_on_pad and (is_upright or is_soft):
            # Good aim and stable, but hard landing or the other way round
            reward = 30.0
        elif is_on_pad:
            reward = 10.0
        else:
            reward = -50.0
        return (
            reward,
            terminated,
            _build_info(
                dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad
            ),
        )

    if np.abs(x) > x_init_max or y > y_init_max:
        terminated = True
        reward = -90.0
        return (
            reward,
            terminated,
            _build_info(
                dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad
            ),
        )

    # === B. SHAPING REWARDS (In-Flight) ===

    # B0. Global survival bonus (small positive for staying alive)-----------------------
    reward += -0.28

    if is_travel_phase:
        # === PHASE 1: TRAVEL REWARDS (Aggressive Flight) ===
        # Goal: Get to the platform quickly. Allow banking/tilting for speed.

        # B1. Approach Reward: Reward velocity pointing towards target
        # Direction to target (normalized)
        if not is_stable:
            r_upright = np.exp(-2 * (np.abs(theta) - 0.2)) - 1.0
            r_stable_w = np.exp(-(np.abs(theta_dot) - 0.2)) - 1.0
            r_stable_v = np.exp(-(v_total - 1.0)) - 1.0
            reward += 0.1 * (r_upright + r_stable_w + r_stable_v)
        else:
            r_dist_to_x = np.exp(-0.25 * (np.abs(dist_x) - 10.0)) - 1.0
            r_dist_to_y = np.exp(-0.25 * (np.abs(dist_y) - 10.0)) - 1.0

            # B2. Relaxed Stability: Only penalize extreme tilt (> ~45 degrees)
            # This allows the drone to bank for horizontal movement
            if np.abs(theta) > 0.8:
                r_tilt = -2.0 * (np.abs(theta) - 0.8)  # Penalize excess tilt
                reward += r_tilt

            # B3. Mild penalty for spinning (always bad)
            r_spin = np.exp(-(np.abs(theta_dot) - 1.0)) - 1.0
            reward += r_spin + r_dist_to_x + r_dist_to_y

    else:
        # === PHASE 2: LANDING REWARDS (Precision) ===
        # Goal: Slow down, stabilize, and land precisely on the pad.

        # B1. Position Reward: Exponential pull towards exact center
        # Stronger reward as drone gets closer
        r_position = 2.0 * np.exp(-dist_total)
        reward += r_position

        # B2. Strict Stability: Heavily penalize any tilt
        if np.abs(theta) > 0.1:
            r_tilt = -1.2 * np.abs(
                theta
            )  # Penalize tilt ---------------------------------------
            reward += r_tilt

        # B3. Braking Penalty: Force the drone to slow down for soft landing
        r_braking = -0.4 * np.abs(vx_rel)  # vel_rel_total
        reward += r_braking

        # B4. Penalize spinning (stricter in landing phase)
        r_spin = -0.5 * np.abs(theta_dot)
        reward += r_spin

        # B5. Vertical velocity control: Penalize descending too fast
        if vy_rel < -2.0:  # Falling faster than 2 m/s
            r_descent = -1.0 * (np.abs(vy_rel) - 2.0)
            reward += r_descent

        # B6. Reward gentle descent -----------------------------------
        if vy_rel < 0:  # If moving down
            reward += 1.2 * np.abs(vy_rel)

    info = _build_info(
        dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad
    )

    return reward, terminated, info


def _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad):
    """
    Build the info dictionary with diagnostic metrics.

    Args:
        dist_x (float): Signed horizontal distance to platform (m).
            Positive means drone is to the right of platform.
        dist_y (float): Vertical distance to ground (m).
        dist_total (float): Euclidean distance to landing target (m).
        vx_rel (float): Horizontal velocity relative to platform (m/s).
        vy_rel (float): Vertical velocity (m/s).
        vel_rel_total (float): Total relative velocity magnitude (m/s).
        is_on_pad (bool): True if drone is horizontally within pad bounds (|dist_x| < 0.5m).

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
