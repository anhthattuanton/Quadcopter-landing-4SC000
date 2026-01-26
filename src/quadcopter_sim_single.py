def record_episode(env, model, max_steps=10000):
    """
    Run the simulation to completion without rendering and record all states.

    Args:
        env: PlanarQuadcopterEnv instance
        model: Trained PPO model (or None for random actions)
        max_steps: Maximum number of steps to record

    Returns:
        states: List of recorded states
        infos: List of info dicts for each step
        final_step: The step number when episode ended
    """
    states = []
    infos = []

    # Reset environment
    state, info = env.reset()
    states.append(state.copy())
    infos.append(info)

    final_step = 0

    for step in range(max_steps):
        # Get action from model or random
        if model is not None:
            action, _ = model.predict(state, deterministic=True)
        else:
            action = env.action_space.sample()

        # Execute action
        state, reward, terminated, truncated, info = env.step(action)

        # Record state
        states.append(state.copy())
        infos.append(info)

        final_step = step + 1

        # Check termination
        if terminated or truncated:
            break

    return states, infos, final_step


def record_episode_random(env, max_steps=10000):
    """
    Run the simulation with random actions and record all states.

    Args:
        env: PlanarQuadcopterEnv instance
        max_steps: Maximum number of steps to record

    Returns:
        states: List of recorded states
        infos: List of info dicts for each step
        final_step: The step number when episode ended
    """
    return record_episode(env, model=None, max_steps=max_steps)
