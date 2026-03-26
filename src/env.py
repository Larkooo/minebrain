"""
Gymnasium environment wrapper for Minecraft via Mineflayer bridge.

Provides MinecraftEnv (single env) and AsyncVecEnv (parallel environments)
with action masking, frame stacking, and curriculum integration.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.bridge import SyncBridge, DEFAULT_WS_URL
from src.curriculum import (
    compute_reward, get_shaping_weight, check_stage_goal, STAGES, NUM_STAGES,
)
from src.observations import (
    OBS_SIZE, STACKED_OBS_SIZE, FRAME_STACK, encode_observation, FrameStacker,
)
from src.skills import NUM_SKILLS, get_stage_mask


class MinecraftEnv(gym.Env):
    """Minecraft environment that communicates with Mineflayer over WebSocket.

    Action space (Discrete 72): macro-action skills with masking.
    Observation space: 930-dim stacked feature vector (310 * 3 frames).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_id: int = 0,
        bridge: SyncBridge | None = None,
        ws_url: str = DEFAULT_WS_URL,
        stage: int = 0,
        render_mode=None,
    ):
        super().__init__()
        self.env_id = env_id
        self.stage = stage
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(NUM_SKILLS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STACKED_OBS_SIZE,), dtype=np.float32
        )

        # Bridge (shared across VecEnv instances or created locally)
        self._bridge = bridge
        self._owns_bridge = bridge is None
        self._ws_url = ws_url

        # State tracking
        self._frame_stacker = FrameStacker(FRAME_STACK, OBS_SIZE)
        self._prev_raw_state: dict = {}
        self._raw_state: dict = {}
        self._step_count = 0
        self._shaping_weight = 0.5
        self._stage_progress = 0.0

    def _get_bridge(self) -> SyncBridge:
        if self._bridge is None:
            self._bridge = SyncBridge(self._ws_url)
            self._bridge.connect()
        return self._bridge

    def set_shaping_weight(self, weight: float):
        """Update reward shaping weight (for annealing during training)."""
        self._shaping_weight = max(0.0, min(1.0, weight))

    def set_stage(self, stage: int):
        """Update curriculum stage."""
        self.stage = min(stage, NUM_STAGES - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        bridge = self._get_bridge()

        game_seed = int(self.np_random.integers(0, 2**31)) if seed is None else seed
        result = bridge.reset(self.env_id, self.stage, game_seed)

        self._raw_state = result.get("raw_state", {})
        self._prev_raw_state = self._raw_state.copy()
        self._step_count = 0

        # Encode observation
        obs = encode_observation(self._raw_state, self.stage)
        stacked = self._frame_stacker.reset(obs)

        # Action mask: combine bot preconditions with stage gating
        bot_mask = np.array(result.get("action_mask", [True] * NUM_SKILLS), dtype=np.bool_)
        stage_mask = get_stage_mask(self.stage)
        combined_mask = bot_mask & stage_mask

        info = {
            "action_mask": combined_mask,
            "stage": self.stage,
            "raw_state": self._raw_state,
        }
        return stacked, info

    def step(self, action: int):
        bridge = self._get_bridge()
        action = int(action)

        self._prev_raw_state = self._raw_state.copy()

        # Execute skill via bridge
        result = bridge.step(self.env_id, action)

        self._raw_state = result.get("raw_state", {})
        skill_result = result.get("skill_result", {"success": False, "skill_name": ""})
        self._step_count += 1

        # Compute reward
        base_reward, shaped_reward = compute_reward(
            self.stage, self._raw_state, self._prev_raw_state, skill_result
        )
        reward = base_reward + self._shaping_weight * shaped_reward

        # Check done conditions
        died = self._raw_state.get("player", {}).get("health", 20) <= 0
        episode_timeout = self._step_count >= self._max_steps()
        dragon_dead = self._raw_state.get("progress", {}).get("milestones", {}).get("dragon_defeated", False)
        done = died or episode_timeout or dragon_dead

        # Encode observation
        obs = encode_observation(self._raw_state, self.stage)
        stacked = self._frame_stacker.push(obs)

        # Action mask
        bot_mask = np.array(result.get("action_mask", [True] * NUM_SKILLS), dtype=np.bool_)
        stage_mask = get_stage_mask(self.stage)
        combined_mask = bot_mask & stage_mask

        # If no valid actions, episode is done
        if not combined_mask.any():
            done = True

        # Check stage goal
        stage_goal_met = check_stage_goal(self.stage, self._raw_state)

        info = {
            "action_mask": combined_mask,
            "stage": self.stage,
            "stage_goal_met": stage_goal_met,
            "skill_result": skill_result,
            "died": died,
            "raw_state": self._raw_state,
        }

        if self.render_mode == "human" and done:
            status = "DIED" if died else ("GOAL!" if stage_goal_met else "timeout")
            print(f"[Env {self.env_id}] Episode done: {status} | steps={self._step_count}")

        return stacked, reward, done, False, info

    def _max_steps(self) -> int:
        """Max steps per episode based on stage config and average skill duration."""
        avg_skill_sec = 5.0
        episode_sec = STAGES[self.stage].episode_length_sec
        return max(10, int(episode_sec / avg_skill_sec))


class AsyncVecEnv:
    """Vectorized environment with parallel stepping via the bridge.

    Steps all environments in parallel (skills execute concurrently
    on the bot server), which is critical since skills take 1-15 seconds.
    """

    def __init__(
        self,
        n_envs: int,
        ws_url: str = DEFAULT_WS_URL,
        stage: int = 0,
    ):
        self.n_envs = n_envs
        self._bridge = SyncBridge(ws_url)
        self._bridge.connect()

        self.envs = [
            MinecraftEnv(env_id=i, bridge=self._bridge, stage=stage)
            for i in range(n_envs)
        ]

    def set_stage(self, stage: int):
        for env in self.envs:
            env.set_stage(stage)

    def set_shaping_weight(self, weight: float):
        for env in self.envs:
            env.set_shaping_weight(weight)

    def reset(self):
        """Reset all environments. Returns (obs, masks) arrays."""
        obs_list = []
        mask_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            mask_list.append(info["action_mask"])
        return np.array(obs_list), np.array(mask_list)

    def step(self, actions: np.ndarray):
        """Step all environments with given actions.

        Uses batch_step for parallel execution on the bot server.

        Returns: (obs, rewards, dones, masks, infos)
        """
        # Execute all skills in parallel via bridge
        action_pairs = [(i, int(actions[i])) for i in range(self.n_envs)]
        results = self._bridge.batch_step(action_pairs)

        obs_list = []
        reward_list = []
        done_list = []
        mask_list = []
        info_list = []

        for i, (env, result) in enumerate(zip(self.envs, results)):
            # Update env state from bridge result
            env._raw_state = result.get("raw_state", {})
            skill_result = result.get("skill_result", {"success": False, "skill_name": ""})
            env._step_count += 1
            env._prev_raw_state = env._raw_state.copy()

            # Compute reward
            base_reward, shaped_reward = compute_reward(
                env.stage, env._raw_state, env._prev_raw_state, skill_result
            )
            reward = base_reward + env._shaping_weight * shaped_reward

            # Check done
            died = env._raw_state.get("player", {}).get("health", 20) <= 0
            episode_timeout = env._step_count >= env._max_steps()
            dragon_dead = env._raw_state.get("progress", {}).get("milestones", {}).get("dragon_defeated", False)
            done = died or episode_timeout or dragon_dead

            # Encode observation
            obs_raw = encode_observation(env._raw_state, env.stage)
            stacked = env._frame_stacker.push(obs_raw)

            # Action mask
            bot_mask = np.array(result.get("action_mask", [True] * NUM_SKILLS), dtype=np.bool_)
            stage_mask = get_stage_mask(env.stage)
            combined_mask = bot_mask & stage_mask

            if not combined_mask.any():
                done = True

            stage_goal_met = check_stage_goal(env.stage, env._raw_state)

            info = {
                "action_mask": combined_mask,
                "stage": env.stage,
                "stage_goal_met": stage_goal_met,
                "skill_result": skill_result,
                "died": died,
            }

            # Auto-reset on done
            if done:
                info["final_step_count"] = env._step_count
                info["final_stage_goal_met"] = stage_goal_met
                reset_obs, reset_info = env.reset()
                stacked = reset_obs
                combined_mask = reset_info["action_mask"]

            obs_list.append(stacked)
            reward_list.append(reward)
            done_list.append(done)
            mask_list.append(combined_mask)
            info_list.append(info)

        return (
            np.array(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            np.array(mask_list),
            info_list,
        )
