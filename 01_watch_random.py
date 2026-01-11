import gymnasium as gym
import gymnasium_robotics
import time


def main() -> None:
    # Register robotics environments so Gym knows about them
    gym.register_envs(gymnasium_robotics)

    # Create a robot arm environment
    env = gym.make("FetchPickAndPlace-v4", render_mode="human")

    # Reset the environment to start a new episode
    obs, info = env.reset(seed=0)

    # Run the simulation for 500 steps
    for _ in range(500):
        # Pick a RANDOM action
        action = env.action_space.sample()

        # Apply the action to the robot
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.05)

        # If episode ends, reset the environment
        if terminated or truncated:
            obs, info = env.reset()

    # Close the simulation window cleanly
    env.close()


if __name__ == "__main__":
    main()
