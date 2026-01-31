"""
Episode Recording Module for Quadcopter Simulation.

This module provides functionality to record complete episodes without
rendering, enabling faster-than-realtime simulation and smooth playback.
The recorded states can then be played back at any speed.

Usage:
    Record an episode with a trained model, then play it back using
    the visualizer. This separates simulation speed from rendering speed.
"""



def record_episode(env, model, max_steps=10000):
    """
    Record a complete episode without rendering.

    Runs the simulation to completion (termination or max steps) and stores
    all states for later playback. This is much faster than rendering in
    real-time since no graphics are generated during recording.

    Args:
        env (PlanarQuadcopterEnv): Gymnasium environment instance.
        model: Trained policy model with predict() method (e.g., PPO from SB3).
            If None, random actions are sampled from the action space.
        max_steps (int): Maximum number of steps to record before forcing
            episode end. Default: 10000.

    Returns:
        tuple: (states, infos, final_step)
            states (list[np.ndarray]): List of state arrays, one per timestep.
                Each state has shape (9,).
            infos (list[dict]): List of info dictionaries from each step.
                Contains diagnostic metrics from reward function.
            final_step (int): The step number when episode ended (1-indexed).

    Example:
        >>> env = PlanarQuadcopterEnv()
        >>> model = PPO.load("models/PPO/100000")
        >>> states, infos, steps = record_episode(env, model)
        >>> print(f"Recorded {len(states)} frames over {steps} steps")
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
    Equivalent to calling record_episode(env, model=None, max_steps).

    Args:
        env (PlanarQuadcopterEnv): Gymnasium environment instance.
        max_steps (int): Maximum number of steps to record. Default: 10000.

    Returns:
        tuple: (states, infos, final_step) - Same format as record_episode().

    Example:
        >>> env = PlanarQuadcopterEnv()
        >>> states, infos, steps = record_episode_random(env)
        >>> print(f"Random agent recorded {steps} steps")
    """
    return record_episode(env, model=None, max_steps=max_steps)
