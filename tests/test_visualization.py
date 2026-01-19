import matplotlib.pyplot as plt

from src.env import PlanarQuadcopterEnv
from src.visualization import QuadcopterVisualizer


def main():
    # Initialize the environment and visualizer
    env = PlanarQuadcopterEnv()
    vis = QuadcopterVisualizer(env)

    # Reset the environment
    state, info = env.reset()
    vis.reset()

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
        if vis.fig is not None and not plt.fignum_exists(vis.fig.number):
            print("\nWindow closed by user.")
            break

        # Check if user wants to quit
        if vis.should_quit:
            print("\nQuit requested by user.")
            break

        # Handle reset request
        if vis.should_reset:
            state, info = env.reset()
            vis.reset()
            step = 0
            print("\nSimulation reset!")
            continue

        # Check if paused
        if vis.paused:
            vis.render(state)
            continue

        # Sample a random action
        action = env.action_space.sample()

        # Execute the action
        state, reward, terminated, truncated, info = env.step(action)

        # Render the environment
        vis.render(state)

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
                if vis.fig is None or not plt.fignum_exists(vis.fig.number):
                    running = False
                    break

                if vis.should_quit:
                    running = False
                    break

                if vis.should_reset:
                    state, info = env.reset()
                    vis.reset()
                    step = 0
                    print("\nSimulation reset!")
                    break

                vis.render(state)

    # Close everything
    vis.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
