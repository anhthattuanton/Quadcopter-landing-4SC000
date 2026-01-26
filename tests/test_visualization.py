import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.simulation_data import FRAME_SKIP
from src.env import PlanarQuadcopterEnv
from src.visualization import QuadcopterVisualizer
from src.quadcopter_sim_single import record_episode


def main():
    # Initialize the environment and visualizer
    env = PlanarQuadcopterEnv()
    vis = QuadcopterVisualizer(env)

    # Load the trained PPO model
    model = PPO.load("models/PPO/10000000")  # Adjust to your latest model file

    print("=" * 50)
    print("QUADCOPTER SIMULATION (PPO Agent)")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Pause/Resume simulation")
    print("  R     - Reset simulation (new recording)")
    print("  Q     - Quit simulation")
    print("  (Close window to exit)")
    print("=" * 50)

    # Record the episode first
    print("\nRecording episode...")
    states, infos, final_step = record_episode(env, model)
    print(f"Recording complete! {len(states)} frames recorded.")

    # Playback variables
    frame_idx = 0
    running = True

    while running:
        # Check if figure was closed
        if vis.fig is not None and not plt.fignum_exists(vis.fig.number):
            print("\nWindow closed by user.")
            break

        # Check if user wants to quit
        if vis.should_quit:
            print("\nQuit requested by user.")
            break

        # Handle reset request - generate new recording
        if vis.should_reset:
            print("\nRecording new episode...")
            vis.reset()
            states, infos, final_step = record_episode(env, model)
            print(f"Recording complete! {len(states)} frames recorded.")
            frame_idx = 0
            continue

        # Check if paused
        if vis.paused:
            # Still render current frame while paused
            vis.render(states[frame_idx])
            continue

        # Get current state for playback
        state = states[frame_idx]
        info = infos[frame_idx]

        # Render the frame
        vis.render(state)

        # Advance to next frame
        frame_idx += FRAME_SKIP

        # Check if we've reached the end of recording
        if frame_idx >= len(states):
            print(f"\nPlayback ended at frame {frame_idx}")
            print(f"  Final position: x={state[0]:.2f}, y={state[1]:.2f}")
            print(f"  Distance to platform: {info['dist_total']:.2f} m")
            print(f"  Final velocity: {info['vel_total']:.2f} m/s")
            print(f"  On pad: {info['is_on_pad']}")
            print("\nPress R for new episode, Q to quit, or close the window.")

            # Wait at final frame
            while True:
                if vis.fig is None or not plt.fignum_exists(vis.fig.number):
                    running = False
                    break

                if vis.should_quit:
                    running = False
                    break

                if vis.should_reset:
                    print("\nRecording new episode...")
                    vis.reset()
                    states, infos, final_step = record_episode(env, model)
                    print(f"Recording complete! {len(states)} frames recorded.")
                    frame_idx = 0
                    break

                # Keep rendering final frame
                vis.render(state)

    # Close everything
    vis.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
