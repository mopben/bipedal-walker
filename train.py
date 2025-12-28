import os
import copy
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import gymnasium as gym
import imageio
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization


@dataclass(frozen=True)
class Config:
    env_id: str = "BipedalWalker-v3"
    hardcore: bool = False

    seed: int = 42
    n_envs: int = 32
    total_timesteps: int = 1_000_000

    # ---- Reward shaping schedule (training wheels) ----
    # The environment’s base reward is already shaped; these terms are meant ONLY to reduce early collapse.
    # Professional default: anneal to zero so the final policy is optimized for the true task reward.
    shaping_enabled: bool = True
    alive_coef_start: float = 0.01     # per-step bonus while not done (small!)
    alive_coef_end: float = 0.0
    alive_anneal_steps: int = 50_000  # linearly anneal over first N env timesteps

    # Keep these OFF by default (base env already incentivizes forward progress and penalizes torque)
    forward_coef: float = 0.0          # optional: coef * clip(vx, low, high)
    effort_coef: float = 0.0           # optional: -coef * sum(action^2)
    vx_clip_low: float = -1.0
    vx_clip_high: float = 3.0

    # ---- PPO ----
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.18
    ent_coef: float = 0.0             # “lock in” gait; reduce drift
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ---- Artifacts ----
    run_dir: str = "runs/bipedalwalker_ppo_expert_baseline"
    gif_every_steps: int = 200_000
    gif_max_steps: int = 1600
    gif_fps: int = 30

    # ---- Eval (base reward only) ----
    eval_every_steps: int = 200_000
    n_eval_episodes: int = 5


# ----------------------------- Env / Reward Shaping -----------------------------

class WalkerRewardShaping(gym.Wrapper):
    """
    Additive reward shaping (training wheels). IMPORTANT: base reward is preserved.

    shaped_reward = base_reward
                  + forward_coef * clip(vx, low, high)
                  + alive_coef (only if not done)
                  - effort_coef * sum(action^2)
    """
    def __init__(
        self,
        env: gym.Env,
        forward_coef: float,
        alive_coef: float,
        effort_coef: float,
        vx_clip: Tuple[float, float],
    ):
        super().__init__(env)
        self.forward_coef = float(forward_coef)
        self.alive_coef = float(alive_coef)
        self.effort_coef = float(effort_coef)
        self.vx_clip = (float(vx_clip[0]), float(vx_clip[1]))

    # Called via VecEnv.env_method("set_shaping", ...) even through Monitor wrapper
    def set_shaping(
        self,
        forward_coef: Optional[float] = None,
        alive_coef: Optional[float] = None,
        effort_coef: Optional[float] = None,
    ) -> None:
        if forward_coef is not None:
            self.forward_coef = float(forward_coef)
        if alive_coef is not None:
            self.alive_coef = float(alive_coef)
        if effort_coef is not None:
            self.effort_coef = float(effort_coef)

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)

        vx = float(obs[2])  # BipedalWalker forward velocity
        r_forward = self.forward_coef * float(np.clip(vx, *self.vx_clip))
        r_alive = 0.0 if (terminated or truncated) else self.alive_coef
        r_effort = -self.effort_coef * float(np.sum(np.square(action)))

        r = float(base_r + r_forward + r_alive + r_effort)

        info = dict(info)
        info.update(r_forward=r_forward, r_alive=r_alive, r_effort=r_effort, r_total=r, r_base=float(base_r))
        return obs, r, terminated, truncated, info


def make_base_env(cfg: Config, render_mode: Optional[str] = None) -> gym.Env:
    """Environment with NO shaping (base reward only)."""
    env = gym.make(cfg.env_id, hardcore=cfg.hardcore, render_mode=render_mode)
    return Monitor(env)


def make_train_env(cfg: Config, render_mode: Optional[str] = None) -> gym.Env:
    """Training environment (optionally with shaping)."""
    env = gym.make(cfg.env_id, hardcore=cfg.hardcore, render_mode=render_mode)
    if cfg.shaping_enabled:
        env = WalkerRewardShaping(
            env,
            forward_coef=cfg.forward_coef,
            alive_coef=cfg.alive_coef_start,
            effort_coef=cfg.effort_coef,
            vx_clip=(cfg.vx_clip_low, cfg.vx_clip_high),
        )
    return Monitor(env)


def make_train_env_thunk(cfg: Config) -> Callable[[], gym.Env]:
    return lambda: make_train_env(cfg, render_mode=None)


def make_eval_env_thunk(cfg: Config) -> Callable[[], gym.Env]:
    # Eval MUST be on base reward only (no shaping wrapper)
    return lambda: make_base_env(cfg, render_mode=None)


# ----------------------------- GIF Rollout (correct normalization, base reward env) -----------------------------

def _build_normalized_render_venv(cfg: Config, src_norm: VecNormalize) -> VecNormalize:
    """
    Create a 1-env render VecEnv with VecNormalize stats copied from the training VecNormalize.
    Rendering uses the base environment (no shaping) so rewards align with evaluation.
    """
    venv = DummyVecEnv([lambda: make_base_env(cfg, render_mode="rgb_array")])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=src_norm.clip_obs)

    # Copy running stats from training env
    venv.obs_rms = copy.deepcopy(src_norm.obs_rms)
    venv.training = False
    venv.norm_reward = False
    return venv


def rollout_and_save_gif(
    model: PPO,
    cfg: Config,
    src_norm: VecNormalize,
    gif_path: str,
    max_steps: int,
    fps: int,
    seed: Optional[int] = None,
) -> float:
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    venv = _build_normalized_render_venv(cfg, src_norm)

    # SB3 VecEnvs generally do NOT accept reset(seed=...) on VecNormalize wrappers
    if seed is not None:
        venv.seed(seed)
    obs = venv.reset()

    frames = []
    ep_r = 0.0
    for _ in range(max_steps):
        frames.append(venv.envs[0].render())

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        ep_r += float(rewards[0])

        if bool(dones[0]):
            break

    imageio.mimsave(gif_path, frames, fps=fps)
    venv.close()
    return ep_r


# ----------------------------- Callbacks -----------------------------

class ShapingScheduleCallback(BaseCallback):
    """
    Linearly anneal alive shaping from alive_coef_start -> alive_coef_end over alive_anneal_steps.
    Applies to the *training* envs only. Eval/GIF envs remain base reward only.

    This avoids the "camping for alive reward" failure mode while still improving early stability.
    """
    def __init__(self, cfg: Config):
        super().__init__(verbose=0)
        self.cfg = cfg

    def _alive_coef_at(self, t: int) -> float:
        if self.cfg.alive_anneal_steps <= 0:
            return float(self.cfg.alive_coef_end)
        frac = min(max(t / float(self.cfg.alive_anneal_steps), 0.0), 1.0)
        return float((1.0 - frac) * self.cfg.alive_coef_start + frac * self.cfg.alive_coef_end)

    def _on_step(self) -> bool:
        if not self.cfg.shaping_enabled:
            return True

        # self.num_timesteps is in "env steps" already (SB3 tracks global env timesteps)
        alive = self._alive_coef_at(self.num_timesteps)

        # training_env is VecNormalize -> underlying VecEnv supports env_method
        # The outer Monitor wrapper forwards attributes to inner WalkerRewardShaping via gym.Wrapper __getattr__.
        try:
            self.training_env.env_method(
                "set_shaping",
                forward_coef=self.cfg.forward_coef,
                alive_coef=alive,
                effort_coef=self.cfg.effort_coef,
            )
        except Exception:
            # If shaping is disabled or wrapper not present, ignore
            pass
        return True


class GifRecorderCallback(BaseCallback):
    def __init__(self, cfg: Config):
        super().__init__(verbose=0)
        self.cfg = cfg
        self.next_step = cfg.gif_every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_step:
            return True

        env = self.model.get_env()
        if not isinstance(env, VecNormalize):
            raise TypeError("Training env must be VecNormalize to record normalized GIFs correctly.")

        path = os.path.join(self.cfg.run_dir, "gifs", f"step_{self.num_timesteps:08d}.gif")
        try:
            ep_r = rollout_and_save_gif(
                model=self.model,
                cfg=self.cfg,
                src_norm=env,
                gif_path=path,
                max_steps=self.cfg.gif_max_steps,
                fps=self.cfg.gif_fps,
                seed=self.cfg.seed,
            )
            print(f"[GIF] {path} (base_ep_reward={ep_r:.1f})")
        except Exception as e:
            print(f"[GIF] Failed at step {self.num_timesteps}: {e}")

        self.next_step += self.cfg.gif_every_steps
        return True


class SyncedEvalCallback(EvalCallback):
    """
    EvalCallback that keeps eval VecNormalize statistics synced with the training VecNormalize.
    Eval runs on base env reward only (no shaping wrapper).
    """
    def _on_step(self) -> bool:
        try:
            sync_envs_normalization(self.training_env, self.eval_env)
        except Exception:
            pass
        return super()._on_step()


# ----------------------------- Main -----------------------------

def main():
    cfg = Config()
    os.makedirs(cfg.run_dir, exist_ok=True)

    # Train env: reward shaping optionally enabled, but reward normalization OFF.
    train_env = make_vec_env(make_train_env_thunk(cfg), n_envs=cfg.n_envs, seed=cfg.seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Eval env: base reward only; VecNormalize synced from train env.
    eval_env = make_vec_env(make_eval_env_thunk(cfg), n_envs=1, seed=cfg.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    model = PPO(
        "MlpPolicy",
        train_env,
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
        sde_sample_freq=4
    )

    callbacks = CallbackList([
        ShapingScheduleCallback(cfg),
        SyncedEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(cfg.run_dir, "best_model"),
            log_path=os.path.join(cfg.run_dir, "eval_logs"),
            # Convert env-step frequency to callback-step frequency for vectorized envs.
            # Each callback step advances num_timesteps by n_envs.
            eval_freq=max(1, cfg.eval_every_steps // cfg.n_envs),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        ),
        GifRecorderCallback(cfg),
    ])

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks, progress_bar=True)

    # Save final artifacts (model + normalization)
    model_path = os.path.join(cfg.run_dir, "final_model.zip")
    vecnorm_path = os.path.join(cfg.run_dir, "vecnormalize.pkl")
    model.save(model_path)
    train_env.save(vecnorm_path)

    # Render final model on BASE reward env
    rollout_and_save_gif(
        model=model,
        cfg=cfg,
        src_norm=train_env,
        gif_path=os.path.join(cfg.run_dir, "gifs", "final.gif"),
        max_steps=cfg.gif_max_steps,
        fps=cfg.gif_fps,
        seed=cfg.seed,
    )

    # Render best eval model (if it exists)
    best_path = os.path.join(cfg.run_dir, "best_model", "best_model.zip")
    if os.path.exists(best_path):
        best_model = PPO.load(best_path, env=train_env)
        rollout_and_save_gif(
            model=best_model,
            cfg=cfg,
            src_norm=train_env,
            gif_path=os.path.join(cfg.run_dir, "gifs", "best.gif"),
            max_steps=cfg.gif_max_steps,
            fps=cfg.gif_fps,
            seed=cfg.seed,
        )
        print(f"[DONE] Saved best.gif from {best_path}")

    print(f"[DONE] Saved final_model.zip and vecnormalize.pkl in {cfg.run_dir}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
