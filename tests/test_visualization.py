# import sys
# sys.path.insert(0, 'c:\\4SC000\\Quadcopter-landing-4SC000')

from src.env import PlanarQuadcopterEnv


def main():
    # Initialize the environment
    env = PlanarQuadcopterEnv()

    # Reset the environment
    state, info = env.reset()

    # Run for 1000 timesteps
    for step in range(1000):
        # Sample a random action
        action = env.action_space.sample()

        # Execute the action
        state, info = env.step(action)

        # Render the environment
        env.render()

        # Note: terminated/truncated handling commented out since step()
        # currently returns only (state, info)
        # Uncomment below when reward system is re-enabled:
        # state, reward, terminated, truncated, info = env.step(action)
        # if terminated:
        #     print('Crashed!')
        #     state, info = env.reset()

    # Close the environment
    env.close()
    print("Simulation complete!")


if __name__ == "__main__":
    main()
