import os
from typing import List

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


# ------------------------------------------------------------
# Custom callback: save at specific timesteps
# ------------------------------------------------------------
class SaveOnStepsCallback(BaseCallback):
    def __init__(self, save_steps: List[int], save_dir: str = "checkpoints", verbose: int = 1):
        super().__init__(verbose)
        self.save_steps = sorted(save_steps)
        self.save_dir = save_dir
        self._idx = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self._idx >= len(self.save_steps):
            return True

        current_steps = self.num_timesteps
        target_steps = self.save_steps[self._idx]

        if current_steps >= target_steps:
            step_str = str(target_steps)

            model_path = os.path.join(self.save_dir, f"sac_reach_{step_str}_steps")
            vecnorm_path = os.path.join(
                self.save_dir, f"sac_reach_vecnormalize_{step_str}_steps.pkl"
            )

            # Save model
            self.model.save(model_path)

            # Save VecNormalize
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                env.save(vecnorm_path)
            else:
                raise RuntimeError("VecNormalize env not found")

            if self.verbose:
                print(f"[Checkpoint] Saved at {step_str} steps")

            self._idx += 1

        return True


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def main() -> None:
    gym.register_envs(gymnasium_robotics)

    # Parallel envs (good balance for learning)
    env = make_vec_env(
        "FetchReachDense-v4",
        n_envs=16,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    # Normalize observations
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # 🔥 Dense early checkpoints, sparse later
    save_steps = [
        # chaotic motor babbling
        1_000,
        2_000,
        3_000,
        5_000,
        8_000,
        10_000,
        15_000,
        20_000,

        # early learning
        30_000,
        50_000,
        75_000,
        100_000,

        # mid learning
        150_000,
        200_000,
        300_000,
        350_000,

        # late learning
        400_000,
        450_000,
        500_000,
        550_000,
        600_000,
    ]

    checkpoint_callback = SaveOnStepsCallback(
        save_steps=save_steps,
        save_dir="checkpoints",
        verbose=1,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log="tb_logs",
        device="cpu",

        # 🔥 PURE RANDOM ACTIONS AT START
        learning_starts=30_000,

        # Exploration + stability
        ent_coef="auto_0.3",
        train_freq=4,
        gradient_steps=1,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.98,
    )

    model.learn(
        total_timesteps=600_000,
        callback=checkpoint_callback,
    )

    model.save("sac_fetch_reach_final")
    env.save("vecnormalize_final.pkl")
    env.close()


if __name__ == "__main__":
    main()
