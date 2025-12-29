import os
import copy
import argparse
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import gymnasium as gym
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization


# ----------------------------- Config -----------------------------

@dataclass(frozen=True)
class Config:
    env_id: str = "BipedalWalker-v3"
    hardcore: bool = False

    seed: int = 42
    n_envs: int = 32
    total_timesteps: int = 2_000_000

    # ---- Reward shaping schedule (training wheels) ----
    shaping_enabled: bool = True
    alive_coef_start: float = 0.01
    alive_coef_end: float = 0.0
    alive_anneal_steps: int = 250_000

    forward_coef: float = 0.0
    effort_coef: float = 0.0
    vx_clip_low: float = -1.0
    vx_clip_high: float = 3.0

    # ---- PPO ----
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.18
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ---- Artifacts ----
    run_dir: str = "runs/ablations/baseline/seed_42"
    gif_every_steps: int = 400_000
    gif_max_steps: int = 1600
    gif_fps: int = 30

    # ---- Eval (base reward only) ----
    eval_every_steps: int = 100_000
    n_eval_episodes: int = 10


# ----------------------------- Reward Shaping Wrapper -----------------------------

class WalkerRewardShaping(gym.Wrapper):
    """Additive shaping (training wheels). Base reward is preserved."""

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

        vx = float(obs[2])  # forward velocity
        r_forward = self.forward_coef * float(np.clip(vx, *self.vx_clip))
        r_alive = 0.0 if (terminated or truncated) else self.alive_coef
        r_effort = -self.effort_coef * float(np.sum(np.square(action)))

        r = float(base_r + r_forward + r_alive + r_effort)

        info = dict(info)
        info.update(
            r_forward=float(r_forward),
            r_alive=float(r_alive),
            r_effort=float(r_effort),
            r_total=float(r),
            r_base=float(base_r),
        )
        return obs, r, terminated, truncated, info


# ----------------------------- RND (training-only) -----------------------------

class RunningMeanStd:
    """Online mean/std for scalar normalization."""
    def __init__(self, eps: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: float) -> None:
        x = float(x)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2

    @property
    def std(self) -> float:
        denom = max(self.count - 1.0, 1.0)
        return float(np.sqrt(self.var / denom))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDWrapper(gym.Wrapper):
    """
    Random Network Distillation - training only wrapper.

    Intrinsic reward = normalized predictor MSE vs fixed random target network.
    We expose set_rnd_scale() so a global schedule (via callback) can anneal RND.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        hidden: int = 64,
        out_dim: int = 32,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        super().__init__(env)
        self.device = device

        self.target = MLP(obs_dim, out_dim, hidden).to(device)
        self.predictor = MLP(obs_dim, out_dim, hidden).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = optim.Adam(self.predictor.parameters(), lr=lr)
        self.rms = RunningMeanStd()

        # Current scale (set by callback). Default 0.0 so it is safe if unset.
        self._rnd_scale: float = 0.0

    def set_rnd_scale(self, scale: float) -> None:
        self._rnd_scale = float(scale)

    def step(self, action):
        obs, r_ext, terminated, truncated, info = self.env.step(action)

        # No intrinsic reward if scale is zero (or extremely small)
        if self._rnd_scale <= 0.0:
            info = dict(info)
            info.setdefault("r_int", 0.0)
            info.setdefault("r_ext", float(r_ext))
            return obs, float(r_ext), terminated, truncated, info

        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)

        with torch.no_grad():
            tgt = self.target(x)
        pred = self.predictor(x)

        mse = torch.mean((pred - tgt) ** 2)
        mse_scalar = float(mse.detach().cpu().item())

        # Update predictor
        self.opt.zero_grad(set_to_none=True)
        mse.backward()
        self.opt.step()

        # Normalize prediction error
        self.rms.update(mse_scalar)
        norm_err = (mse_scalar - self.rms.mean) / (self.rms.std + 1e-8)
        r_int = self._rnd_scale * float(norm_err)

        r_total = float(r_ext + r_int)
        info = dict(info)
        info.update(r_int=float(r_int), r_ext=float(r_ext), rnd_err=float(mse_scalar))
        return obs, r_total, terminated, truncated, info


# ----------------------------- Env builders -----------------------------

def make_base_env(cfg: Config, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(cfg.env_id, hardcore=cfg.hardcore, render_mode=render_mode)
    # Monitor should be near the base; do not put it as the outermost wrapper
    env = Monitor(env)
    return env


def make_train_env(cfg: Config, args: argparse.Namespace, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(cfg.env_id, hardcore=cfg.hardcore, render_mode=render_mode)

    # Training wheels shaping
    if cfg.shaping_enabled:
        env = WalkerRewardShaping(
            env,
            forward_coef=cfg.forward_coef,
            alive_coef=cfg.alive_coef_start,
            effort_coef=cfg.effort_coef,
            vx_clip=(cfg.vx_clip_low, cfg.vx_clip_high),
        )

    # Optional RND (training only)
    if args.use_rnd:
        obs_dim = int(np.prod(env.observation_space.shape))
        env = RNDWrapper(
            env,
            obs_dim=obs_dim,
            hidden=args.rnd_hidden,
            out_dim=args.rnd_out_dim,
            lr=args.rnd_lr,
            device="cpu",
        )
    env = Monitor(env)

    return env


def make_train_env_thunk(cfg: Config, args: argparse.Namespace) -> Callable[[], gym.Env]:
    return lambda: make_train_env(cfg, args, render_mode=None)


def make_eval_env_thunk(cfg: Config) -> Callable[[], gym.Env]:
    # Eval must be BASE reward only (no shaping, no RND)
    return lambda: make_base_env(cfg, render_mode=None)


# ----------------------------- GIF rendering (base reward, synced norm) -----------------------------

def _build_normalized_render_venv(cfg: Config, src_norm: VecNormalize) -> VecNormalize:
    """1-env render VecEnv with VecNormalize stats copied from training."""
    venv = DummyVecEnv([lambda: make_base_env(cfg, render_mode="rgb_array")])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=src_norm.clip_obs)
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
    deterministic: bool = True,
) -> float:
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    venv = _build_normalized_render_venv(cfg, src_norm)
    if seed is not None:
        venv.seed(seed)
    obs = venv.reset()

    frames = []
    ep_r = 0.0
    for _ in range(max_steps):
        frames.append(venv.envs[0].render())

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = venv.step(action)
        ep_r += float(rewards[0])

        if bool(dones[0]):
            break

    imageio.mimsave(gif_path, frames, fps=fps)
    venv.close()
    return ep_r


# ----------------------------- Callbacks -----------------------------

class ShapingScheduleCallback(BaseCallback):
    """Linearly anneal alive shaping from start to end over alive_anneal_steps."""

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

        alive = self._alive_coef_at(self.num_timesteps)
        # If shaping wrapper exists anywhere in the chain, __getattr__ will forward
        try:
            self.training_env.env_method(
                "set_shaping",
                forward_coef=self.cfg.forward_coef,
                alive_coef=alive,
                effort_coef=self.cfg.effort_coef,
            )
        except Exception:
            pass
        return True


class RNDScheduleCallback(BaseCallback):
    """Anneal RND scale from start to end over rnd_anneal_steps. No-op if RND not enabled."""

    def __init__(self, rnd_scale_start: float, rnd_scale_end: float, rnd_anneal_steps: int):
        super().__init__(verbose=0)
        self.start = float(rnd_scale_start)
        self.end = float(rnd_scale_end)
        self.steps = int(rnd_anneal_steps)

    def _scale_at(self, t: int) -> float:
        if self.steps <= 0:
            return self.end
        frac = min(max(t / float(self.steps), 0.0), 1.0)
        return float((1.0 - frac) * self.start + frac * self.end)

    def _on_step(self) -> bool:
        scale = self._scale_at(self.num_timesteps)
        try:
            self.training_env.env_method("set_rnd_scale", scale)
        except Exception:
            # If RND isn't present, this is fine.
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
                deterministic=True,
            )
            print(f"[GIF] {path} (base_ep_reward={ep_r:.1f})")
        except Exception as e:
            print(f"[GIF] Failed at step {self.num_timesteps}: {e}")

        self.next_step += self.cfg.gif_every_steps
        return True


class SyncedEvalCallback(EvalCallback):
    """EvalCallback that keeps eval VecNormalize statistics synced with training."""

    def _on_step(self) -> bool:
        try:
            sync_envs_normalization(self.training_env, self.eval_env)
        except Exception:
            pass
        return super()._on_step()


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--exp_name", type=str, default="baseline")

    # PPO improvements / ablations
    p.add_argument("--use_sde", action="store_true")
    p.add_argument("--sde_sample_freq", type=int, default=4)
    p.add_argument("--sde_init_sigma", type=float, default=0.3)
    p.add_argument("--target_kl", type=float, default=0.0)  # 0 disables

    # RND toggle + schedule + net params
    p.add_argument("--use_rnd", action="store_true")
    p.add_argument("--rnd_scale_start", type=float, default=0.05)
    p.add_argument("--rnd_scale_end", type=float, default=0.0)
    p.add_argument("--rnd_anneal_steps", type=int, default=250_000)
    p.add_argument("--rnd_lr", type=float, default=1e-4)
    p.add_argument("--rnd_hidden", type=int, default=64)
    p.add_argument("--rnd_out_dim", type=int, default=32)

    # Optional knobs for quicker iteration
    p.add_argument("--n_envs", type=int, default=32)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument("--eval_every_steps", type=int, default=100_000)
    p.add_argument("--gif_every_steps", type=int, default=400_000)

    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> Config:
    run_dir = os.path.join("runs", "ablations", args.exp_name, f"seed_{args.seed}")
    return Config(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_eval_episodes=args.n_eval_episodes,
        eval_every_steps=args.eval_every_steps,
        gif_every_steps=args.gif_every_steps,
        run_dir=run_dir,
    )


# ----------------------------- Main -----------------------------

def main() -> None:

    args = parse_args()
    cfg = build_cfg(args)
    print(f"[CONFIG] gif_every_steps = {cfg.gif_every_steps}")

    os.makedirs(cfg.run_dir, exist_ok=True)

    # Train env: shaping optional; RND optional; reward normalization OFF.
    train_env = make_vec_env(make_train_env_thunk(cfg, args), n_envs=cfg.n_envs, seed=cfg.seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Eval env: base reward only; VecNormalize synced from train env.
    eval_env = make_vec_env(make_eval_env_thunk(cfg), n_envs=1, seed=cfg.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    # PPO kwargs
    ppo_kwargs: Dict[str, Any] = dict(
        policy="MlpPolicy",
        env=train_env,
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

    # Improvement: gSDE
    if args.use_sde:
        ppo_kwargs.update(
            use_sde=True,
            sde_sample_freq=args.sde_sample_freq,
            policy_kwargs={"log_std_init": float(np.log(args.sde_init_sigma))},
        )

    # Improvement: KL early stopping
    if args.target_kl and args.target_kl > 0:
        ppo_kwargs.update(target_kl=float(args.target_kl))

    model = PPO(**ppo_kwargs)

    callbacks = [
        ShapingScheduleCallback(cfg),
        SyncedEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(cfg.run_dir, "best_model"),
            log_path=os.path.join(cfg.run_dir, "eval_logs"),
            # EvalCallback eval_freq is in *callback calls*, so divide by n_envs
            eval_freq=max(1, cfg.eval_every_steps // cfg.n_envs),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        ),
        GifRecorderCallback(cfg),
    ]

    # RND schedule callback only if RND is enabled
    if args.use_rnd:
        callbacks.insert(
            1,
            RNDScheduleCallback(
                rnd_scale_start=args.rnd_scale_start,
                rnd_scale_end=args.rnd_scale_end,
                rnd_anneal_steps=args.rnd_anneal_steps,
            ),
        )

    model.learn(total_timesteps=cfg.total_timesteps, callback=CallbackList(callbacks), progress_bar=True)

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
        deterministic=True,
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
            deterministic=True,
        )
        print(f"[DONE] Saved best.gif from {best_path}")

    print(f"[DONE] Saved final_model.zip and vecnormalize.pkl in {cfg.run_dir}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    # Optional: make torch a bit less aggressive on CPU threads
    torch.set_num_threads(1)
    main()
