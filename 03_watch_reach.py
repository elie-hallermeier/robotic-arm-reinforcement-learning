import glob
import os
import time

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def run_one_episode(model, env, sleep_s: float = 0.05) -> None:
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        time.sleep(sleep_s)
        done = done[0]


def main() -> None:
    gym.register_envs(gymnasium_robotics)

    def make_env():
        return gym.make(
            "FetchReachDense-v4",
            render_mode="human",
        )

    base_env = DummyVecEnv([make_env])

    # Find checkpoints
    policy_paths = sorted(
        glob.glob("checkpoints/sac_reach_*_steps.zip"),
        key=lambda p: int(os.path.basename(p).split("_")[2]),
    )

    replays = []
    for policy_path in policy_paths:
        step = os.path.basename(policy_path).split("_")[2]
        vecnorm_path = f"checkpoints/sac_reach_vecnormalize_{step}_steps.pkl"

        if not os.path.exists(vecnorm_path):
            raise RuntimeError(f"Missing VecNormalize for {step} steps: {vecnorm_path}")

        replays.append((step, policy_path, vecnorm_path))

    if not replays:
        # Fresh clone, nothing trained yet: replay the trained policy that ships with the repo
        print("No checkpoints in checkpoints/, replaying the shipped final policy")
        replays = [("500000 (final)", "sac_fetch_reach_final.zip", "vecnormalize_final.pkl")]

    print(f"Found {len(replays)} policies to replay")

    for step, policy_path, vecnorm_path in replays:
        print(f"\nReplaying {step} steps")

        vec_env = VecNormalize.load(vecnorm_path, base_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = SAC.load(policy_path, env=vec_env)

        # One slow episode per checkpoint
        run_one_episode(model, vec_env, sleep_s=0.01)

        # Small pause so you feel the transition
        time.sleep(0.0)

    base_env.close()


if __name__ == "__main__":
    main()
