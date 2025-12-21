import os
from dataclasses import dataclass
from typing import Optional, List, Callable

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

    # Reward shaping (tunable)
    r_forward_coef: float = 1.0
    r_alive_coef: float = 1.5
    r_height_coef: float = 1.5
    r_effort_coef: float = 0.01
    r_height_tol: float = 0.5
    r_vx_clip_low: float = -1.0
    r_vx_clip_high: float = 3.0

    # PPO hyperparams
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
    gif_every_steps: int = 50_000      # record a GIF every N training steps
    gif_max_steps: int = 1600          # cap frames per GIF
    gif_fps: int = 30

    # Eval
    eval_every_steps: int = 50_000
    n_eval_episodes: int = 5


class WalkerRewardShaping(gym.Wrapper):
    """
    Reward = forward velocity + alive bonus + walking-height bonus - effort penalty

    forward velocity: obs[2]
    alive bonus: +alive_coef each step until terminated/truncated
    walking height: encourage hull y-position near target (set at reset)
    effort penalty: -effort_coef * sum(action^2)
    """
    def __init__(
        self,
        env: gym.Env,
        forward_coef: float = 1.0,
        alive_coef: float = 1.0,
        height_coef: float = 1.0,
        effort_coef: float = 0.02,
        height_tol: float = 0.5,
        vx_clip: tuple[float, float] = (-1.0, 3.0),
    ):
        super().__init__(env)
        self.forward_coef = float(forward_coef)
        self.alive_coef = float(alive_coef)
        self.height_coef = float(height_coef)
        self.effort_coef = float(effort_coef)
        self.height_tol = float(height_tol)
        self.vx_clip = (float(vx_clip[0]), float(vx_clip[1]))

        self.target_height: Optional[float] = None

    def set_coefs(
        self,
        forward_coef: Optional[float] = None,
        alive_coef: Optional[float] = None,
        height_coef: Optional[float] = None,
        effort_coef: Optional[float] = None,
    ):
        if forward_coef is not None:
            self.forward_coef = float(forward_coef)
        if alive_coef is not None:
            self.alive_coef = float(alive_coef)
        if height_coef is not None:
            self.height_coef = float(height_coef)
        if effort_coef is not None:
            self.effort_coef = float(effort_coef)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self.target_height = float(self.env.unwrapped.hull.position[1])
        except Exception:
            self.target_height = None
        return obs, info

    def step(self, action):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)

        # forward velocity (x) is obs[2] for BipedalWalker
        vx = float(obs[2])
        vx_c = float(np.clip(vx, self.vx_clip[0], self.vx_clip[1]))
        r_forward = self.forward_coef * vx_c

        # alive bonus (per timestep while episode continues)
        r_alive = 0.0 if (terminated or truncated) else self.alive_coef

        # height bonus (prefer true hull y; fallback to hull angle proxy)
        r_height = 0.0
        if self.target_height is not None:
            try:
                hull_y = float(self.env.unwrapped.hull.position[1])
                z = (hull_y - self.target_height) / self.height_tol
                # peak at target height; negative if far away (clipped)
                r_height = self.height_coef * float(np.clip(1.0 - z * z, -1.0, 1.0))
            except Exception:
                hull_angle = float(obs[0])
                r_height = self.height_coef * float(np.exp(-abs(hull_angle)))
        else:
            hull_angle = float(obs[0])
            r_height = self.height_coef * float(np.exp(-abs(hull_angle)))

        # effort penalty
        r_effort = -self.effort_coef * float(np.sum(np.square(action)))

        shaped_reward = float(r_forward + r_alive + r_height + r_effort)

        # log components
        info = dict(info)
        info["r_forward"] = float(r_forward)
        info["r_alive"] = float(r_alive)
        info["r_height"] = float(r_height)
        info["r_effort"] = float(r_effort)
        info["r_total"] = float(shaped_reward)

        return obs, shaped_reward, terminated, truncated, info


def make_env_fn(cfg: Config) -> Callable[[], gym.Env]:
    """
    Returns a thunk (callable with no args) that constructs a *fresh* env.
    This is the correct pattern for SB3's make_vec_env.
    """
    def _init():
        env = gym.make(cfg.env_id, hardcore=cfg.hardcore)
        env = WalkerRewardShaping(
            env,
            forward_coef=cfg.r_forward_coef,
            alive_coef=cfg.r_alive_coef,
            height_coef=cfg.r_height_coef,
            effort_coef=cfg.r_effort_coef,
            height_tol=cfg.r_height_tol,
            vx_clip=(cfg.r_vx_clip_low, cfg.r_vx_clip_high),
        )
        env = Monitor(env)  # monitor should wrap the final reward you train on
        return env

    return _init


def rollout_and_save_gif(
    model: PPO,
    cfg: Config,
    gif_path: str,
    max_steps: int,
    fps: int,
    seed: Optional[int] = None,
) -> float:
    """
    Runs ONE episode with rendering and saves an animated GIF.
    Returns episode reward (under the same shaped reward as training).
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    env = gym.make(cfg.env_id, hardcore=cfg.hardcore, render_mode="rgb_array")
    env = WalkerRewardShaping(
        env,
        forward_coef=cfg.r_forward_coef,
        alive_coef=cfg.r_alive_coef,
        height_coef=cfg.r_height_coef,
        effort_coef=cfg.r_effort_coef,
        height_tol=cfg.r_height_tol,
        vx_clip=(cfg.r_vx_clip_low, cfg.r_vx_clip_high),
    )

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

    if len(frames) >= 1:
        imageio.mimsave(gif_path, frames, fps=fps)

    return ep_reward


class GifRecorderCallback(BaseCallback):
    """
    Records a rollout GIF every `gif_every_steps` training steps.
    Rendering happens in a separate non-vector env (rgb_array),
    so it does not slow down the vectorized training env much.
    """
    def __init__(self, cfg: Config, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.next_record_step = cfg.gif_every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_record_step:
            gif_name = f"step_{self.num_timesteps:08d}.gif"
            gif_path = os.path.join(self.cfg.run_dir, "gifs", gif_name)

            try:
                ep_r = rollout_and_save_gif(
                    model=self.model,
                    cfg=self.cfg,
                    gif_path=gif_path,
                    max_steps=self.cfg.gif_max_steps,
                    fps=self.cfg.gif_fps,
                    seed=self.cfg.seed,
                )
                if self.verbose:
                    print(f"[GIF] Saved {gif_path} (episode reward={ep_r:.1f})")
            except Exception as e:
                print(f"[GIF] Failed to save GIF at step {self.num_timesteps}: {e}")

            self.next_record_step += self.cfg.gif_every_steps

        return True


def main():
    cfg = Config()
    os.makedirs(cfg.run_dir, exist_ok=True)

    # --- Training env (vectorized, no rendering) ---
    vec_env = make_vec_env(
        make_env_fn(cfg),
        n_envs=cfg.n_envs,
        seed=cfg.seed,
    )

    # Normalize observations/rewards (often helpful for BipedalWalker)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Separate eval env ---
    eval_env = make_vec_env(
        make_env_fn(cfg),
        n_envs=1,
        seed=cfg.seed + 1000,
    )
    # For eval: normalize obs, but keep rewards interpretable
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.run_dir, "best_model"),
        log_path=os.path.join(cfg.run_dir, "eval_logs"),
        eval_freq=cfg.eval_every_steps,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

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

    # Final GIF
    final_gif_path = os.path.join(cfg.run_dir, "gifs", "final.gif")
    ep_r = rollout_and_save_gif(
        model=model,
        cfg=cfg,
        gif_path=final_gif_path,
        max_steps=cfg.gif_max_steps,
        fps=cfg.gif_fps,
        seed=cfg.seed,
    )
    print(f"[FINAL] Saved {final_gif_path} (episode reward={ep_r:.1f})")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
