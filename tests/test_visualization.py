# import sys
# sys.path.insert(0, 'c:\\4SC000\\Quadcopter-landing-4SC000')

from src.env import PlanarQuadcopterEnv
import matplotlib.pyplot as plt


def main():
    # Initialize the environment
    env = PlanarQuadcopterEnv()

    # Reset the environment
    state, info = env.reset()

    print("=" * 50)
    print("QUADCOPTER SIMULATION")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Pause/Resume simulation")
    print("  R     - Reset simulation")
    print("  Q     - Quit simulation")
    print("  (Close window to exit)")
    print("=" * 50)

    running = True
    step = 0
    max_steps = 10000

    while running and step < max_steps:
        # Check if figure was closed
        if (
            not plt.fignum_exists(env.fig.number)
            if hasattr(env, "fig") and env.fig is not None
            else False
        ):
            print("\nWindow closed by user.")
            break

        # Check if user wants to quit
        if env.should_quit:
            print("\nQuit requested by user.")
            break

        # Handle reset request
        if env.should_reset:
            state, info = env.reset()
            env.should_reset = False
            step = 0
            print("\nSimulation reset!")
            continue

        # Check if paused
        if env.paused:
            env.render()
            continue

        # Sample a random action
        action = env.action_space.sample()

        # Execute the action
        state, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        env.render()

        step += 1

        # Check for termination
        if terminated:
            print(f"\nSimulation ended at step {step}")
            print(f"  Final position: x={state[0]:.2f}, y={state[1]:.2f}")
            print(f"  Distance to platform: {info['distance_to_platform']:.2f} m")
            print(f"  Final velocity: {info['velocity']:.2f} m/s")
            print("\nPress R to reset, Q to quit, or close the window.")

            # Wait for user input after termination
            while True:
                # Check if figure was closed
                if (
                    not plt.fignum_exists(env.fig.number)
                    if hasattr(env, "fig") and env.fig is not None
                    else True
                ):
                    running = False
                    break

                if env.should_quit:
                    running = False
                    break

                if env.should_reset:
                    state, info = env.reset()
                    env.should_reset = False
                    step = 0
                    print("\nSimulation reset!")
                    break

                # Keep rendering to process events
                env.render()

    # Close the environment
    env.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
