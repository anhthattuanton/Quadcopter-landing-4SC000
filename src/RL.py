from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable
import os
import shutil

from src.env import PlanarQuadcopterEnv

"""
TODO: fix reward function, then train PPO agent to land quadcopter on moving platform.
"""


def constant_then_decay_schedule(
    initial_value: float, decay_start: float = 0.5
) -> Callable[[float], float]:
    """
    Constant learning rate until 'decay_start', then linear decay to 0.

    :param initial_value: The max learning rate (e.g., 3e-4)
    :param decay_start: When to start decaying (0.0 to 1.0).
                        0.5 means start decaying when 50% of time is left.
                        0.8 means start decaying when 80% of time is left (early decay).
    """

    def func(progress_remaining: float) -> float:
        # Phase 1: Constant (High)
        # If we have lots of time left (more than the start point), keep it max
        if progress_remaining > decay_start:
            return initial_value

        # Phase 2: Decay (Linear)
        # We need to map the range [decay_start -> 0] to [initial_value -> 0]
        else:
            return initial_value * (progress_remaining / decay_start)

    return func


def train():
    # 1. Setup Directories
    models_dir = "models/PPO"
    log_dir = "logs"

    # 1. Clean out OLD Models (Handle both files and folders)
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)  # Deletes the folder and everything inside
    os.makedirs(models_dir)  # Re-create empty folder

    # 2. Clean out OLD Logs
    if os.path.exists(log_dir):
        # Check if TensorBoard is holding files open!
        try:
            shutil.rmtree(log_dir)
        except PermissionError:
            print(
                "⚠️ COULD NOT DELETE LOGS: TensorBoard might be running. Please close it."
            )
    os.makedirs(log_dir)

    # 2. Vectorize the Environment (Run 4 simulations at once!)
    # This speeds up training massively.
    env = make_vec_env(
        PlanarQuadcopterEnv, n_envs=8, seed=42, vec_env_cls=SubprocVecEnv
    )
    # Learning Rate Schedule
    lr_schedule = constant_then_decay_schedule(initial_value=3e-4, decay_start=0.5)

    # 3. Define the PPO Model
    # Define the Network Architecture (The "Brain" Size)
    # pi = Policy (Actor), vf = Value Function (Critic)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        # 1. Learning Rate (Your Schedule)
        learning_rate=lr_schedule,
        # 2. Network Size (CRITICAL UPGRADE)
        policy_kwargs=policy_kwargs,
        # 3. Batch Size (Higher is usually better for gradients)
        batch_size=2048,
        # 4. n_steps (Steps per CPU before updating)
        # 8 CPUs * 2048 steps = 16,384 steps per update. Good size.
        n_steps=2048,
        # 5. Gamma (Patience)
        # 0.995 allows it to "see" further into the future for that landing bonus
        gamma=0.995,
        # 6. GAE Lambda (Bias vs Variance)
        # 0.95 is standard, 0.98 is smoother for continuous control
        gae_lambda=0.98,
        # 7. Entropy Coefficient (Exploration)
        # 0.01 forces it to keep trying new things slightly
        ent_coef=0.01,
        # 8. Gradient Clipping (Safety)
        # Prevents the neural network from "exploding" if it sees bad data
        max_grad_norm=0.5,
    )

    # 4. The Training Loop
    TOTAL_TIMESTEPS = 5_000_000  # Goal: 5 Million steps
    STEPS_PER_LOOP = 100_000

    # Calculate how many times to loop (5,000,000 / 100,000 = 50 loops)
    num_loops = int(TOTAL_TIMESTEPS / STEPS_PER_LOOP)

    print(f"Training for {TOTAL_TIMESTEPS} steps in {num_loops} loops...")

    for i in range(1, num_loops + 1):
        # Train
        model.learn(
            total_timesteps=STEPS_PER_LOOP, reset_num_timesteps=False, tb_log_name="PPO"
        )

        # Save (Filename example: "models/PPO/100000")
        current_steps = i * STEPS_PER_LOOP
        model.save(f"{models_dir}/{current_steps}")
        print(f"Saved model at {current_steps} steps")

    print("Training Complete!")
    env.close()


if __name__ == "__main__":
    train()
