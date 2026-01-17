import numpy as np

from src.env import PlanarQuadcopterEnv


def test_environment_initialization():
    env = PlanarQuadcopterEnv()
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (8,)


def test_reset_returns_valid_state():
    env = PlanarQuadcopterEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (8,)
    # Check if initial height (index 1) is around 10.0 as set in reset
    assert np.isclose(obs[1], 10.0)


def test_step_function_shape():
    env = PlanarQuadcopterEnv()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
