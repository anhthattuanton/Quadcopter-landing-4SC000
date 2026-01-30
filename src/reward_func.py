import numpy as np

from src.simulation_data import x_init_max, y_init_max


def calculate_reward(state):
    """
    Calculate the reward using a Phased Reward Strategy.
    
    Phase 1 (Travel): Aggressive flight to approach the platform.
                      Rewards velocity towards target, relaxed tilt penalty.
    
    Phase 2 (Landing): Precision landing on the platform.
                       Rewards position accuracy, strict stability, braking.

    Args:
        state: numpy array [x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, platform_a]
        action: numpy array [left_thrust, right_thrust] normalized to [-1, 1]
        platform_x: current x position of the platform

    Returns:
        reward (float): the calculated reward
        terminated (bool): whether the episode has ended
        info (dict): additional information
    """
    # === UNPACK STATE ===
    x, y, theta, x_dot, y_dot, theta_dot, platform_x, platform_v, _ = state

    # === CALCULATE RELATIVE METRICS ===
    # Signed distance to platform (positive = drone is to the right of platform)
    dist_x = x - platform_x
    dist_y = y
    dist_total = np.sqrt(dist_x**2 + dist_y**2)

    # Relative velocity (drone velocity minus platform velocity)
    v_total = np.sqrt(x_dot**2 + y_dot**2)
    vx_rel = x_dot - platform_v
    vy_rel = y_dot
    vel_rel_total = np.sqrt(vx_rel**2 + vy_rel**2)

    # === DEFINE PHASES ===
    # Travel Phase: Far from target, need to approach aggressively
    # Landing Phase: Close to target, need precision and stability
    PHASE_THRESHOLD = 8.0  # meters
    is_travel_phase = dist_total > PHASE_THRESHOLD
    is_stable = (np.abs(theta) < 0.2) and (v_total < 1.0) and (np.abs(theta_dot) < 0.2)
    # === SUCCESS FLAGS ===
    is_on_pad = np.abs(dist_x) < 0.5
    is_upright = np.abs(theta) < 0.2
    is_soft = vel_rel_total < 0.1
    is_landed = y <= 0.0

    reward = 0.0
    terminated = False

    # === A. TERMINAL CONDITIONS ===
    
    # A1. Ground Contact (Landed)
    if is_landed:
        terminated = True
        if is_on_pad and is_upright and is_soft:
            # Perfect landing: on pad, upright, and soft
            reward = 700.0
        elif is_on_pad and (is_upright or is_soft):
            # Good aim and stable, but hard landing or the other way round
            reward = 30.0
        elif is_on_pad:
            # Hit the pad but crashed/tilted
            reward = 10.0
        else:
            # Missed the pad entirely
            reward = -50.0
        
        return reward, terminated, _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad)

    # A2. Out of Bounds
    if np.abs(x) > x_init_max or y > y_init_max:
        terminated = True
        reward = -90.0
        return reward, terminated, _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad)

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
            r_tilt = -1.2 * np.abs(theta)  # Penalize tilt ---------------------------------------
            reward += r_tilt

        # B3. Braking Penalty: Force the drone to slow down for soft landing
        r_braking = -0.4 * np.abs(vx_rel) #vel_rel_total
        reward += r_braking

        # B4. Penalize spinning (stricter in landing phase)
        r_spin = -0.5 * np.abs(theta_dot)
        reward += r_spin

        # B5. Vertical velocity control: Penalize descending too fast
        if vy_rel < -2.0:  # Falling faster than 2 m/s
            r_descent = -1.0 * (np.abs(vy_rel) - 2.0)
            reward += r_descent

        # B6. Reward gentle descent -----------------------------------
        if vy_rel < 0: # If moving down
            reward += 1.2 * np.abs(vy_rel)

    info = _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad)

    return reward, terminated, info


def _build_info(dist_x, dist_y, dist_total, vx_rel, vy_rel, vel_rel_total, is_on_pad):
    """Build the info dictionary."""
    return {
        "dist_x": dist_x,
        "dist_y": dist_y,
        "dist_total": dist_total,
        "vx_rel": vx_rel,
        "vy_rel": vy_rel,
        "vel_total": vel_rel_total,
        "is_on_pad": is_on_pad,
    }
