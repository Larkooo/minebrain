"""
Evaluation harness for MineBrain.

Evaluates trained models per curriculum stage with statistics.
Supports greedy vs stochastic policy comparison.

Usage:
    python -m src.evaluate                      # Evaluate latest model
    python -m src.evaluate --stage 3            # Evaluate Stage 3
    python -m src.evaluate --episodes 500       # More episodes
    python -m src.evaluate --stochastic         # Stochastic policy
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import mlx.core as mx

from src.env import MinecraftEnv
from src.bridge import SyncBridge
from src.model import MineBrainPolicy
from src.curriculum import STAGES, NUM_STAGES
from src.skills import NUM_SKILLS


def evaluate_stage(
    model: MineBrainPolicy,
    stage: int,
    ws_url: str = "ws://localhost:8765",
    n_episodes: int = 100,
    greedy: bool = True,
) -> dict:
    """Evaluate model on a specific curriculum stage.

    Returns stats dict with performance metrics.
    """
    bridge = SyncBridge(ws_url)
    bridge.connect()

    env = MinecraftEnv(env_id=0, bridge=bridge, stage=stage)

    rewards = []
    goals_met = []
    deaths = []
    step_counts = []

    t0 = time.time()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 100000)
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            obs_mx = mx.array(obs[np.newaxis, :])
            mask_mx = mx.array(info["action_mask"][np.newaxis, :])
            logits, _ = model(obs_mx, mask_mx)
            mx.eval(logits)

            logits_np = np.array(logits[0])
            valid = logits_np > -1e8
            if not valid.any():
                break

            if greedy:
                masked = np.where(valid, logits_np, -1e9)
                action = int(np.argmax(masked))
            else:
                shifted = np.where(valid, logits_np - logits_np[valid].max(), -100.0)
                probs = np.where(valid, np.exp(shifted), 0.0)
                total = probs.sum()
                if total == 0:
                    break
                probs /= total
                action = int(np.random.choice(len(probs), p=probs))

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        goals_met.append(info.get("stage_goal_met", False))
        deaths.append(info.get("died", False))
        step_counts.append(steps)

    elapsed = time.time() - t0
    bridge.disconnect()

    return _compute_stats(rewards, goals_met, deaths, step_counts, elapsed, stage)


def _compute_stats(
    rewards: list[float],
    goals_met: list[bool],
    deaths: list[bool],
    step_counts: list[int],
    elapsed: float,
    stage: int,
) -> dict:
    arr = np.array(rewards)
    return {
        "stage": stage,
        "stage_name": STAGES[stage].name,
        "episodes": len(rewards),
        "elapsed": elapsed,
        "reward_avg": float(arr.mean()),
        "reward_std": float(arr.std()),
        "reward_median": float(np.median(arr)),
        "reward_max": float(arr.max()),
        "reward_min": float(arr.min()),
        "goal_rate": sum(goals_met) / len(goals_met),
        "death_rate": sum(deaths) / len(deaths),
        "avg_steps": sum(step_counts) / len(step_counts),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def print_stats(stats: dict):
    print(f"\n{'='*55}")
    print(f"  Stage {stats['stage']}: {stats['stage_name']} — {stats['episodes']} episodes ({stats['elapsed']:.1f}s)")
    print(f"{'='*55}")
    print(f"  Reward:     {stats['reward_avg']:.2f} ± {stats['reward_std']:.2f}")
    print(f"  Median:     {stats['reward_median']:.2f}")
    print(f"  Range:      [{stats['reward_min']:.2f}, {stats['reward_max']:.2f}]")
    print(f"  P25/P75/P90: {stats['p25']:.2f} / {stats['p75']:.2f} / {stats['p90']:.2f}")
    print(f"  Goal rate:  {stats['goal_rate']*100:.1f}%")
    print(f"  Death rate: {stats['death_rate']*100:.1f}%")
    print(f"  Avg steps:  {stats['avg_steps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MineBrain model")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--ws-url", type=str, default="ws://localhost:8765")
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(args.save_dir) / f"stage_{args.stage}" / "model_best.npz"

    if not model_path.exists():
        print(f"No model found at {model_path}")
        print(f"Train first with: python -m src.train --stage {args.stage}")
        return

    print(f"Loading model from {model_path}")
    model = MineBrainPolicy(hidden=args.hidden)
    model.load_weights(str(model_path))

    mode = "stochastic" if args.stochastic else "greedy"
    print(f"Evaluating Stage {args.stage} ({mode}) over {args.episodes} episodes...")

    stats = evaluate_stage(
        model, args.stage, args.ws_url, args.episodes, greedy=not args.stochastic
    )
    print_stats(stats)


if __name__ == "__main__":
    main()
