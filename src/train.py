"""
PPO training loop for MineBrain using Apple MLX.

Adapted from nums-ai/src/train.py with curriculum integration,
async vectorized environments, and per-stage checkpointing.

Usage:
    python -m src.train                         # Train from Stage 0
    python -m src.train --stage 3               # Start from Stage 3
    python -m src.train --resume models/stage_2  # Resume from checkpoint
    python -m src.train --eval-only --stage 0   # Evaluate Stage 0 model
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.env import AsyncVecEnv, MinecraftEnv
from src.model import MineBrainPolicy, compute_log_probs, compute_entropy
from src.curriculum import (
    STAGES, NUM_STAGES, check_promotion, get_shaping_weight,
)
from src.observations import STACKED_OBS_SIZE
from src.skills import NUM_SKILLS


# ──────────────────────────────────────────────────────────────
# ANSI helpers
# ──────────────────────────────────────────────────────────────

BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
CYAN    = "\033[36m"
MAGENTA = "\033[35m"
CLEAR   = "\033[2K"


def _term_width():
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


# ──────────────────────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────────────────────

DEFAULTS = dict(
    steps_per_stage=2_000_000,
    rollout_steps=2048,
    n_epochs=4,
    batch_size=256,
    gamma=0.995,
    gae_lambda=0.95,
    clip_eps=0.15,
    vf_coef=0.25,
    ent_coef=0.03,
    lr=3e-5,
    max_grad_norm=0.5,
    hidden_size=512,
    n_envs=8,
    eval_interval=100_000,
    eval_episodes=100,
    save_dir="models",
    ws_url="ws://localhost:8765",
    start_stage=0,
)


# ──────────────────────────────────────────────────────────────
# Live dashboard
# ──────────────────────────────────────────────────────────────

class Dashboard:
    """Live-updating terminal dashboard for training progress."""

    TOTAL_LINES = 6

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.t_start = time.time()
        self.last_render = 0
        self.rendered_once = False

        self.total_steps = 0
        self.episodes = 0
        self.stage = cfg.get("start_stage", 0)
        self.best_eval = 0.0

        self.recent_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=20)
        self.recent_goals = deque(maxlen=50)
        self.reward_history = []
        self.episode_results = []

        self.cur_loss = 0.0
        self.cur_pg_loss = 0.0
        self.cur_vf_loss = 0.0
        self.cur_entropy = 0.0
        self.cur_sps = 0
        self.phase = "rollout"

    def _elapsed(self):
        return time.time() - self.t_start

    def _eta(self):
        elapsed = self._elapsed()
        if self.total_steps == 0:
            return "..."
        rate = self.total_steps / elapsed
        remaining = self.cfg["steps_per_stage"] - self.total_steps
        secs = remaining / rate if rate > 0 else 0
        return _fmt_duration(secs)

    def _avg_reward(self):
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def _goal_rate(self):
        if not self.recent_goals:
            return 0.0
        return sum(self.recent_goals) / len(self.recent_goals)

    def _sparkline(self, values, width=25):
        if len(values) < 2:
            return DIM + "..." + RESET
        vals = list(values)[-width:]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx != mn else 1.0
        blocks = " ▁▂▃▄▅▆▇█"
        return "".join(blocks[int((v - mn) / rng * (len(blocks) - 1))] for v in vals)

    def _progress_bar(self, width=35):
        pct = min(self.total_steps / self.cfg["steps_per_stage"], 1.0)
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar, pct

    def render(self, force=False):
        now = time.time()
        if not force and (now - self.last_render) < 0.25:
            return
        self.last_render = now

        if self.rendered_once:
            sys.stdout.write(f"\033[{self.TOTAL_LINES}A")

        stage_cfg = STAGES[self.stage]
        bar, pct = self._progress_bar()
        elapsed_str = _fmt_duration(self._elapsed())
        eta_str = self._eta()
        avg_rwd = self._avg_reward()
        goal_rate = self._goal_rate()
        goal_color = GREEN if goal_rate > 0.8 else (YELLOW if goal_rate > 0.3 else RED)
        loss_str = f"{self.cur_loss:.4f}" if self.cur_loss else "  ..."
        spark = self._sparkline([v for _, v in self.reward_history])

        lines = []
        lines.append(f"{BOLD}MineBrain{RESET} {DIM}Stage {self.stage}: {stage_cfg.name}{RESET} {DIM}│{RESET} PPO on MLX {DIM}│ envs={self.cfg['n_envs']}{RESET}")
        lines.append(f"{bar} {BOLD}{pct*100:5.1f}%{RESET} {DIM}{elapsed_str} elapsed, ETA{RESET} {BOLD}{eta_str}{RESET} {DIM}({self.phase}){RESET}")
        lines.append(f"{DIM}{self.total_steps:,}/{self.cfg['steps_per_stage']:,} steps{RESET} {DIM}│{RESET} {self.cur_sps:,} sps {DIM}│{RESET} {self.episodes:,} episodes")
        lines.append(f"reward {CYAN}{BOLD}{avg_rwd:.2f}{RESET} {DIM}│{RESET} goal {goal_color}{BOLD}{goal_rate*100:.0f}%{RESET} {DIM}│{RESET} loss {CYAN}{loss_str}{RESET} {DIM}[pg {self.cur_pg_loss:.4f} vf {self.cur_vf_loss:.4f} ent {self.cur_entropy:.4f}]{RESET}")
        lines.append(f"trend {spark}")
        lines.append(f"{DIM}promotion: {self._promotion_status()}{RESET}")

        while len(lines) < self.TOTAL_LINES:
            lines.append("")

        output = "\n".join(CLEAR + line for line in lines[:self.TOTAL_LINES])
        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        self.rendered_once = True

    def _promotion_status(self):
        cfg = STAGES[self.stage]
        window = cfg.promotion_window
        if len(self.recent_goals) < window:
            return f"need {window - len(self.recent_goals)} more episodes"
        recent = list(self.recent_goals)[-window:]
        success_rate = sum(recent) / window
        threshold = cfg.promotion_threshold
        if success_rate >= threshold:
            return f"{GREEN}READY ({success_rate*100:.0f}% >= {threshold*100:.0f}%){RESET}"
        return f"{success_rate*100:.0f}% / {threshold*100:.0f}% needed (last {window} episodes)"

    def print_header(self):
        cfg = self.cfg
        stage_cfg = STAGES[self.stage]
        print(f"{BOLD}MineBrain Training{RESET} {DIM}— {stage_cfg.description}{RESET}")
        print(f"{DIM}lr={cfg['lr']} hidden={cfg['hidden_size']} batch={cfg['batch_size']} envs={cfg['n_envs']}{RESET}")
        print()


def _fmt_duration(secs):
    if secs < 0:
        return "..."
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    elif secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    else:
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h {m}m"


# ──────────────────────────────────────────────────────────────
# Rollout buffer
# ──────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self, n_steps: int, n_envs: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, STACKED_OBS_SIZE), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.masks = np.zeros((n_steps, n_envs, NUM_SKILLS), dtype=np.bool_)
        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, dones, values, masks):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.masks[self.ptr] = masks
        self.ptr += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """Compute GAE advantages and returns."""
        advantages = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        returns = advantages + self.values
        return advantages, returns

    def get_batches(self, advantages, returns, batch_size):
        """Yield random minibatches from the buffer."""
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        flat_obs = self.obs.reshape(total, STACKED_OBS_SIZE)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = advantages.reshape(total)
        flat_returns = returns.reshape(total)
        flat_masks = self.masks.reshape(total, NUM_SKILLS)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield (
                mx.array(flat_obs[idx]),
                mx.array(flat_actions[idx]),
                mx.array(flat_log_probs[idx]),
                mx.array(flat_advantages[idx]),
                mx.array(flat_returns[idx]),
                mx.array(flat_masks[idx]),
            )


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train(cfg: dict):
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)

    model = MineBrainPolicy(hidden=cfg["hidden_size"])
    mx.eval(model.parameters())

    # Resume from checkpoint if requested
    resume_path = cfg.get("resume")
    if resume_path:
        p = Path(resume_path)
        model_file = p / "model_best.npz" if p.is_dir() else p
        if model_file.exists():
            model.load_weights(str(model_file))
            mx.eval(model.parameters())
            print(f"  Resumed from {model_file}")
        else:
            print(f"  Warning: {model_file} not found, starting fresh")

    optimizer = optim.Adam(learning_rate=cfg["lr"])

    current_stage = cfg.get("start_stage", 0)

    # Train through curriculum stages
    while current_stage < NUM_STAGES:
        print(f"\n{'='*60}")
        print(f"  STAGE {current_stage}: {STAGES[current_stage].name}")
        print(f"  {STAGES[current_stage].description}")
        print(f"{'='*60}\n")

        model, promoted = train_stage(model, optimizer, current_stage, cfg)

        # Save stage checkpoint
        stage_dir = save_dir / f"stage_{current_stage}"
        stage_dir.mkdir(exist_ok=True)
        save_model(model, stage_dir / "model_promoted.npz")
        print(f"\n  Stage {current_stage} checkpoint saved to {stage_dir}/")

        if promoted:
            current_stage += 1
            # Reset optimizer state for new stage
            optimizer = optim.Adam(learning_rate=cfg["lr"])
        else:
            print(f"\n  Stage {current_stage} training budget exhausted without promotion.")
            print(f"  Continuing to next stage anyway...")
            current_stage += 1

    print(f"\n{GREEN}{BOLD}All stages complete!{RESET}")
    save_model(model, save_dir / "model_final.npz")


def train_stage(model, optimizer, stage: int, cfg: dict):
    """Train a single curriculum stage. Returns (model, promoted)."""

    vec_env = AsyncVecEnv(cfg["n_envs"], cfg["ws_url"], stage)
    obs, masks = vec_env.reset()

    save_dir = Path(cfg["save_dir"]) / f"stage_{stage}"
    save_dir.mkdir(exist_ok=True, parents=True)

    dash = Dashboard(cfg)
    dash.stage = stage
    dash.print_header()

    log_data = []
    t_start = time.time()
    steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
    best_goal_rate = 0.0
    rollout_num = 0

    while rollout_num * steps_per_rollout < cfg["steps_per_stage"]:
        # Anneal reward shaping
        progress = min(rollout_num * steps_per_rollout / cfg["steps_per_stage"], 1.0)
        shaping_w = get_shaping_weight(progress, stage)
        vec_env.set_shaping_weight(shaping_w)

        # Collect rollout
        dash.phase = "rollout"
        buffer = RolloutBuffer(cfg["rollout_steps"], cfg["n_envs"])

        for step in range(cfg["rollout_steps"]):
            if rollout_num == 0 and step == 0:
                dash.phase = "compiling MLX..."
                dash.render(force=True)

            obs_mx = mx.array(obs)
            masks_mx = mx.array(masks)

            logits, values = model(obs_mx, masks_mx)
            mx.eval(logits, values)

            if rollout_num == 0 and step == 0:
                dash.phase = "rollout"

            actions = np.zeros(cfg["n_envs"], dtype=np.int32)
            log_probs_np = np.zeros(cfg["n_envs"], dtype=np.float32)

            logits_np = np.array(logits)
            values_np = np.array(values).squeeze(-1)

            for i in range(cfg["n_envs"]):
                action_idx, log_prob = _sample_action_np(logits_np[i])
                actions[i] = action_idx
                log_probs_np[i] = log_prob

            next_obs, rewards, dones, next_masks, infos = vec_env.step(actions)

            for i in range(cfg["n_envs"]):
                if dones[i]:
                    dash.episodes += 1
                    dash.recent_rewards.append(rewards[i])
                    goal_met = infos[i].get("final_stage_goal_met", False)
                    dash.recent_goals.append(float(goal_met))
                    dash.episode_results.append({"stage_goal_met": goal_met})

            buffer.add(obs, actions, log_probs_np, rewards, dones, values_np, masks)
            obs = next_obs
            masks = next_masks

            if step % 32 == 0:
                dash.total_steps = rollout_num * steps_per_rollout + step * cfg["n_envs"]
                elapsed = time.time() - t_start
                dash.cur_sps = int(dash.total_steps / elapsed) if elapsed > 0 else 0
                dash.render()

        rollout_num += 1
        dash.total_steps = rollout_num * steps_per_rollout

        # Compute advantages
        dash.phase = "update"
        dash.render(force=True)

        _, last_values = model(mx.array(obs), mx.array(masks))
        mx.eval(last_values)
        last_values_np = np.array(last_values).squeeze(-1)

        advantages, returns = buffer.compute_gae(last_values_np, cfg["gamma"], cfg["gae_lambda"])

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # PPO update
        total_loss = 0.0
        total_pg = 0.0
        total_vf = 0.0
        total_ent = 0.0
        n_updates = 0

        loss_and_grad_fn = nn.value_and_grad(model, _ppo_loss)

        for epoch in range(cfg["n_epochs"]):
            for batch in buffer.get_batches(advantages, returns, cfg["batch_size"]):
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_masks = batch

                (loss, (pg_l, vf_l, ent_l)), grads = loss_and_grad_fn(
                    model, b_obs, b_actions, b_old_log_probs,
                    b_advantages, b_returns, b_masks,
                    cfg["clip_eps"], cfg["vf_coef"], cfg["ent_coef"],
                )
                mx.eval(loss, grads)

                loss_val = loss.item()
                if np.isnan(loss_val) or np.isinf(loss_val):
                    continue

                grads = _clip_grad_norm(grads, cfg["max_grad_norm"])
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                n_updates += 1
                total_loss += loss_val
                total_pg += pg_l.item()
                total_vf += vf_l.item()
                total_ent += ent_l.item()

                dash.cur_loss = total_loss / n_updates
                dash.cur_pg_loss = total_pg / n_updates
                dash.cur_vf_loss = total_vf / n_updates
                dash.cur_entropy = total_ent / n_updates
                dash.phase = f"update {epoch+1}/{cfg['n_epochs']}"
                dash.render()

        elapsed = time.time() - t_start
        dash.cur_sps = int(dash.total_steps / elapsed) if elapsed > 0 else 0

        avg_rwd = dash._avg_reward()
        dash.reward_history.append((dash.total_steps, avg_rwd))

        goal_rate = dash._goal_rate()
        if goal_rate > best_goal_rate:
            best_goal_rate = goal_rate
            save_model(model, save_dir / "model_best.npz")

        # Log data
        log_data.append({
            "steps": dash.total_steps,
            "avg_reward": round(avg_rwd, 3),
            "goal_rate": round(goal_rate, 3),
            "episodes": dash.episodes,
            "loss": round(dash.cur_loss, 4) if n_updates > 0 else None,
        })

        dash.render(force=True)

        # Check promotion
        if check_promotion(stage, dash.episode_results):
            print(f"\n\n  {GREEN}{BOLD}PROMOTED!{RESET} Stage {stage} → {stage + 1}")
            with open(save_dir / "training_log.json", "w") as f:
                json.dump(log_data, f, indent=2)
            return model, True

    # Budget exhausted
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    save_model(model, save_dir / "model_best.npz")

    dash.phase = "done"
    dash.render(force=True)
    print(f"\n  Stage {stage} complete: {dash.episodes} episodes in {_fmt_duration(time.time() - t_start)}")
    print(f"  Best goal rate: {best_goal_rate*100:.0f}%")

    return model, False


# ──────────────────────────────────────────────────────────────
# PPO Loss
# ──────────────────────────────────────────────────────────────

def _ppo_loss(
    model, obs, actions, old_log_probs, advantages, returns, masks,
    clip_eps, vf_coef, ent_coef,
):
    """Compute PPO clipped surrogate loss + value loss + entropy bonus."""
    logits, values = model(obs, masks)
    values = values.squeeze(-1)

    new_log_probs = compute_log_probs(logits, actions)
    log_ratio = mx.clip(new_log_probs - old_log_probs, -10.0, 10.0)
    ratio = mx.exp(log_ratio)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = -mx.minimum(ratio * advantages, clipped_ratio * advantages).mean()

    vf_loss = mx.mean(mx.clip((values - returns) ** 2, 0.0, 100.0))

    entropy = compute_entropy(logits).mean()

    total = pg_loss + vf_coef * vf_loss - ent_coef * entropy
    return total, (pg_loss, vf_loss, entropy)


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def _sample_action_np(logits: np.ndarray):
    """Sample action from numpy logits (handles masked actions)."""
    valid = logits > -1e8
    if not valid.any():
        return 0, 0.0

    max_logit = logits[valid].max()
    shifted = np.where(valid, logits - max_logit, -100.0)
    exp_shifted = np.where(valid, np.exp(shifted), 0.0)
    sum_exp = exp_shifted.sum()
    if sum_exp == 0:
        return 0, 0.0
    probs = exp_shifted / sum_exp

    action = np.random.choice(len(probs), p=probs)
    log_prob = np.log(max(probs[action], 1e-10))
    return int(action), float(np.clip(log_prob, -20.0, 0.0))


def _clip_grad_norm(grads, max_norm):
    """Clip gradient norm (tree of arrays)."""
    from mlx.utils import tree_flatten, tree_unflatten
    flat = tree_flatten(grads)
    total_norm_sq = sum(mx.sum(g * g).item() for _, g in flat if g is not None)
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        flat = [(k, g * scale) if g is not None else (k, g) for k, g in flat]
    return tree_unflatten(flat)


def save_model(model, path: Path):
    model.save_weights(str(path))


def load_model(path: Path, hidden: int = 512):
    model = MineBrainPolicy(hidden=hidden)
    model.load_weights(str(path))
    return model


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MineBrain with PPO (MLX)")
    parser.add_argument("--steps-per-stage", type=int, default=DEFAULTS["steps_per_stage"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument("--n-envs", type=int, default=DEFAULTS["n_envs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--rollout-steps", type=int, default=DEFAULTS["rollout_steps"])
    parser.add_argument("--stage", type=int, default=DEFAULTS["start_stage"],
                        help="Starting curriculum stage (0-7)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory or file")
    parser.add_argument("--save-dir", type=str, default=DEFAULTS["save_dir"])
    parser.add_argument("--ws-url", type=str, default=DEFAULTS["ws_url"])
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        from src.evaluate import evaluate_stage
        stage_dir = Path(args.save_dir) / f"stage_{args.stage}"
        model_path = stage_dir / "model_best.npz"
        if not model_path.exists():
            print(f"No model found at {model_path}")
            return
        model = load_model(model_path, hidden=args.hidden)
        evaluate_stage(model, args.stage, args.ws_url)
        return

    cfg = dict(DEFAULTS)
    cfg.update(
        steps_per_stage=args.steps_per_stage,
        lr=args.lr,
        hidden_size=args.hidden,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        start_stage=args.stage,
        save_dir=args.save_dir,
        ws_url=args.ws_url,
        resume=args.resume,
    )

    train(cfg)


if __name__ == "__main__":
    main()
