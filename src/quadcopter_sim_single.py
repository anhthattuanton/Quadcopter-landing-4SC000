"""
Episode Recording Module for Quadcopter Simulation.

This module provides functionality to record complete episodes without
rendering, enabling faster-than-realtime simulation and smooth playback.
The recorded states can then be played back at any speed.
"""

import numpy as np


def record_episode(env, model, max_steps=10000):
    """
    Record a complete episode without rendering.

    Runs the simulation to completion (termination or max steps) and stores
    all states for later playback. This is much faster than rendering in
    real-time.

    Args:
        env: PlanarQuadcopterEnv instance.
        model: Trained policy model with predict() method (e.g., PPO from SB3).
            If None, random actions are sampled from the action space.
        max_steps (int): Maximum number of steps to record. Default: 10000.

    Returns:
        tuple: (states, infos, final_step)
            states (list): List of np.ndarray states, one per timestep.
            infos (list): List of info dicts from each step.
            final_step (int): The step number when episode ended.
    """
    states = []
    infos = []

    state, info = env.reset()
    states.append(state.copy())
    infos.append(info)

    final_step = 0

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(state, deterministic=True)
        else:
            action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)

        states.append(state.copy())
        infos.append(info)

        final_step = step + 1

        if terminated or truncated:
            break

    return states, infos, final_step


def record_episode_random(env, max_steps=10000):
    """
    Record an episode using random actions.

    Convenience function for testing visualization without a trained model.

    Args:
        env: PlanarQuadcopterEnv instance.
        max_steps (int): Maximum number of steps to record. Default: 10000.

    Returns:
        tuple: (states, infos, final_step) - Same format as record_episode().
    """
    return record_episode(env, model=None, max_steps=max_steps)
