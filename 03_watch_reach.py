import time
import glob
import os

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ------------------------------------------------------------
# Run one episode slowly
# ------------------------------------------------------------
def run_one_episode(model, env, sleep_s=0.05):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # VecEnv returns 4 values, NOT 5
        obs, reward, done, info = env.step(action)

        time.sleep(sleep_s)

        # done is an array because this is a VecEnv
        done = done[0]



# ------------------------------------------------------------
# Main replay logic
# ------------------------------------------------------------
def main():
    gym.register_envs(gymnasium_robotics)

    # Base env WITH rendering
    def make_env():
        return gym.make(
            "FetchReachDense-v4",
            render_mode="human",
        )

    base_env = DummyVecEnv([make_env])

    # Find policy checkpoints
    policy_checkpoints = sorted(
        glob.glob("checkpoints/sac_reach_*_steps.zip"),
        key=lambda x: int(os.path.basename(x).split("_")[2]),
    )

    if not policy_checkpoints:
        raise RuntimeError("No policy checkpoints found")

    print(f"Found {len(policy_checkpoints)} checkpoints")

    for policy_path in policy_checkpoints:
        step = os.path.basename(policy_path).split("_")[2]
        print(f"\nReplaying checkpoint at {step} steps")

        # Load matching VecNormalize
        vecnorm_path = f"checkpoints/sac_reach_vecnormalize_{step}_steps.pkl"

        if not os.path.exists(vecnorm_path):
            raise RuntimeError(f"Missing VecNormalize file: {vecnorm_path}")

        vec_env = VecNormalize.load(vecnorm_path, base_env)

        # IMPORTANT: evaluation mode
        vec_env.training = False
        vec_env.norm_reward = False

        # Load model with this env
        model = SAC.load(policy_path, env=vec_env)

        run_one_episode(
            model=model,
            env=vec_env,
            sleep_s=0.05,
        )


    base_env.close()


if __name__ == "__main__":
    main()
