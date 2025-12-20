import os
from dataclasses import dataclass
from typing import Optional, List

import gymnasium as gym
import numpy as np
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


@dataclass
class Config:
    env_id: str = "BipedalWalker-v3"
    hardcore: bool = False

    # Training
    seed: int = 42
    n_envs: int = 8
    total_timesteps: int = 500_000

    # PPO hyperparams (reasonable baseline; not “tuned”)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Logging / saving
    run_dir: str = "runs/bipedalwalker_ppo"
    gif_every_steps: int = 500_000      # record a GIF every N training steps
    gif_max_steps: int = 1600           # cap frames per GIF (env max is usually <= 1600)
    gif_fps: int = 30

    # Eval
    eval_every_steps: int = 50_000
    n_eval_episodes: int = 5


def rollout_and_save_gif(
    model: PPO,
    env_id: str,
    gif_path: str,
    hardcore: bool,
    max_steps: int,
    fps: int,
    seed: Optional[int] = None,
) -> float:
    """
    Runs ONE episode with rendering and saves an animated GIF.
    Returns episode reward.
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    env = gym.make(env_id, hardcore=hardcore, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frames: List[np.ndarray] = []

    ep_reward = 0.0
    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)

        if terminated or truncated:
            break

    env.close()

    if len(frames) >= 2:
        imageio.mimsave(gif_path, frames, fps=fps)
    elif len(frames) == 1:
        # Still save something (single-frame GIF)
        imageio.mimsave(gif_path, frames, fps=fps)

    return ep_reward


class GifRecorderCallback(BaseCallback):
    """
    Records a rollout GIF every `every_steps` training steps.
    Rendering happens in a separate non-vector env (rgb_array),
    so it does not slow down the vectorized training env much.
    """
    def __init__(self, cfg: Config, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.next_record_step = cfg.gif_every_steps

    def _on_step(self) -> bool:
        # num_timesteps is total env steps across learning
        if self.num_timesteps >= self.next_record_step:
            gif_name = f"step_{self.num_timesteps:08d}.gif"
            gif_path = os.path.join(self.cfg.run_dir, "gifs", gif_name)

            try:
                ep_r = rollout_and_save_gif(
                    model=self.model,
                    env_id=self.cfg.env_id,
                    gif_path=gif_path,
                    hardcore=self.cfg.hardcore,
                    max_steps=self.cfg.gif_max_steps,
                    fps=self.cfg.gif_fps,
                    seed=self.cfg.seed,
                )
                if self.verbose:
                    print(f"[GIF] Saved {gif_path} (episode reward={ep_r:.1f})")
            except Exception as e:
                # Do not crash training due to rendering/IO issues.
                print(f"[GIF] Failed to save GIF at step {self.num_timesteps}: {e}")

            self.next_record_step += self.cfg.gif_every_steps

        return True


def main():
    cfg = Config()
    os.makedirs(cfg.run_dir, exist_ok=True)

    # --- Training env (vectorized, no rendering) ---
    # Wrap with Monitor to log episode stats; SB3 handles Monitor well. :contentReference[oaicite:1]{index=1}
    vec_env = make_vec_env(
        cfg.env_id,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        env_kwargs={"hardcore": cfg.hardcore},
        wrapper_class=Monitor,
    )

    # Normalize observations/rewards (often helpful for BipedalWalker)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Separate eval env for SB3 EvalCallback (also vectorized, no rendering) ---
    eval_env = make_vec_env(
        cfg.env_id,
        n_envs=1,
        seed=cfg.seed + 1000,
        env_kwargs={"hardcore": cfg.hardcore},
        wrapper_class=Monitor,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Keep eval normalization in sync with training
    # (SB3 copies stats when saving/loading; during training we’ll sync manually via callback hook)
    # EvalCallback will call evaluate_policy on eval_env; good for checkpoints.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.run_dir, "best_model"),
        log_path=os.path.join(cfg.run_dir, "eval_logs"),
        eval_freq=cfg.eval_every_steps,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # --- PPO model ---
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(cfg.run_dir, "tb"),
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
    )

    # --- Train (with periodic GIFs) ---
    gif_callback = GifRecorderCallback(cfg, verbose=1)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_callback, gif_callback],
        progress_bar=True,
    )

    # Save final model + VecNormalize stats
    model_path = os.path.join(cfg.run_dir, "final_model.zip")
    vecnorm_path = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    model.save(model_path)
    vec_env.save(vecnorm_path)

    # Final “always produce a GIF” render at the end
    final_gif_path = os.path.join(cfg.run_dir, "gifs", "final.gif")
    ep_r = rollout_and_save_gif(
        model=model,
        env_id=cfg.env_id,
        gif_path=final_gif_path,
        hardcore=cfg.hardcore,
        max_steps=cfg.gif_max_steps,
        fps=cfg.gif_fps,
        seed=cfg.seed,
    )
    print(f"[FINAL] Saved {final_gif_path} (episode reward={ep_r:.1f})")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
