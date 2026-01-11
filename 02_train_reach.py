import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def main() -> None:
    # ------------------------------------------------------------
    # Register robotics environments
    # ------------------------------------------------------------
    gym.register_envs(gymnasium_robotics)

    # ------------------------------------------------------------
    # Create vectorized environment (FAST, NO RENDER)
    # ------------------------------------------------------------
    env = make_vec_env(
        "FetchReachDense-v4",
        n_envs=16,                       # sweet spot for M2 Pro
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    # ------------------------------------------------------------
    # Normalize observations (VERY important for robotics)
    # ------------------------------------------------------------
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,              # keep reward scale intact
        clip_obs=10.0,
    )

    # ------------------------------------------------------------
    # Save checkpoints every N steps
    # ------------------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,               # every 20k steps
        save_path="checkpoints",
        name_prefix="sac_reach",
        save_replay_buffer=False,
        save_vecnormalize=True,          # CRITICAL
    )

    # ------------------------------------------------------------
    # Create SAC model
    # ------------------------------------------------------------
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log="tb_logs",
        device="cpu",                    # CPU is best on Apple Silicon
        train_freq=4,
        gradient_steps=1,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.98,
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    model.learn(
        total_timesteps=1_000_000,       # Reach benefits from longer training
        callback=checkpoint_callback,
    )

    # ------------------------------------------------------------
    # Save final model + normalizer
    # ------------------------------------------------------------
    model.save("sac_fetch_reach_final")
    env.save("vecnormalize_final.pkl")

    env.close()


if __name__ == "__main__":
    main()
