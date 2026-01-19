from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable
import os

from src.env import PlanarQuadcopterEnv


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

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Vectorize the Environment (Run 4 simulations at once!)
    # This speeds up training massively.
    env = make_vec_env(
        PlanarQuadcopterEnv, n_envs=8, seed=42, vec_env_cls=SubprocVecEnv
    )
    # Learning Rate Schedule
    lr_schedule = constant_then_decay_schedule(initial_value=3e-4, decay_start=0.5)

    # 3. Define the PPO Model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        # Optional: Tweaked hyperparameters for flying tasks
        learning_rate=lr_schedule,
        ent_coef=0.01,  # Encourage exploration slightly
        batch_size=2048,
    )

    # 4. The Training Loop
    TOTAL_TIMESTEPS = 2_000_000  # Goal: 2 Million steps
    STEPS_PER_LOOP = 100_000

    # Calculate how many times to loop (2,000,000 / 100,000 = 20 loops)
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
