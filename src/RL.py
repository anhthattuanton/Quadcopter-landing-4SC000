"""
Reinforcement Learning Training Script for Quadcopter Landing.

This script trains a PPO agent to land a 2D quadcopter on a moving platform
using Stable-Baselines3. Models are saved periodically and training progress
is logged to TensorBoard.

Usage:
    python -m src.RL

Output:
    - Trained models saved to models/PPO/
    - TensorBoard logs saved to ppo_quadcopter_tensorboard/
"""

import os

from stable_baselines3 import PPO

from src.env import PlanarQuadcopterEnv


def train():
    """
    Train a PPO agent on the quadcopter landing task.

    Creates the environment, initializes the PPO model with MLP policy,
    and runs training in a loop with periodic model saving.

    Training Configuration:
        - Policy: MlpPolicy (Multi-Layer Perceptron)
        - Algorithm: PPO (Proximal Policy Optimization)
        - Save interval: Every 10,000 timesteps
        - Total training: ~1,000,000 timesteps (100 iterations)
    """
    env = PlanarQuadcopterEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_quadcopter_tensorboard/",
    )

    models_dir = "models/PPO"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    TIMESTEPS = 10000
    for i in range(1, 100):
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO",
        )
        model.save(f"{models_dir}/{TIMESTEPS * i}")

    print("Training Complete!")


if __name__ == "__main__":
    train()
