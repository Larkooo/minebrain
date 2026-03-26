"""
Curriculum manager for MineBrain.

Defines 8 progressive stages from punching trees to slaying the Ender Dragon.
Each stage has its own reward function, skill unlocks, promotion criteria,
and reward shaping with annealing (adapted from nums-ai).
"""

from dataclasses import dataclass, field
import numpy as np

from src.skills import NUM_SKILLS, get_stage_mask

NUM_STAGES = 8


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    description: str
    episode_length_sec: int
    shaping_start: float = 0.5
    shaping_end: float = 0.05
    shaping_anneal_frac: float = 0.7
    promotion_window: int = 5       # check last N episodes
    promotion_threshold: float = 1.0  # fraction that must succeed (1.0 = all)


STAGES: list[StageConfig] = [
    StageConfig(
        name="Punch Trees",
        description="Gather wood, craft planks, sticks, crafting table, wooden tools",
        episode_length_sec=300,
        promotion_threshold=1.0,
    ),
    StageConfig(
        name="Stone Age",
        description="Mine stone/coal, craft stone tools and furnace",
        episode_length_sec=600,
        promotion_threshold=1.0,
    ),
    StageConfig(
        name="Iron Age",
        description="Mine/smelt iron, craft iron gear, basic survival (eat, shelter, sleep)",
        episode_length_sec=900,
        promotion_threshold=1.0,
    ),
    StageConfig(
        name="Diamond Hunting",
        description="Mine diamonds deep underground, craft diamond gear, combat",
        episode_length_sec=1200,
        promotion_threshold=0.6,
    ),
    StageConfig(
        name="Nether Prep",
        description="Collect obsidian, build and light nether portal",
        episode_length_sec=1500,
        promotion_threshold=0.6,
    ),
    StageConfig(
        name="Nether Conquest",
        description="Find fortress, kill blazes, collect ender pearls",
        episode_length_sec=2400,
        promotion_threshold=0.4,
    ),
    StageConfig(
        name="Stronghold",
        description="Locate stronghold, activate and enter end portal",
        episode_length_sec=1800,
        promotion_threshold=0.4,
    ),
    StageConfig(
        name="Dragon Slayer",
        description="Destroy crystals, defeat the Ender Dragon",
        episode_length_sec=3600,
        shaping_start=0.3,
        promotion_threshold=0.2,
    ),
]


def compute_reward(stage: int, raw_state: dict, prev_state: dict, skill_result: dict) -> tuple[float, float]:
    """Compute (base_reward, shaped_reward) for a given stage.

    Args:
        stage: Current curriculum stage (0-7).
        raw_state: Current game state dict from bot.
        prev_state: Previous game state dict.
        skill_result: Result of the executed skill {success, skill_name, ...}.

    Returns:
        (base_reward, shaped_reward) — shaped reward gets multiplied by annealing weight.
    """
    success = skill_result.get("success", False)
    skill_name = skill_result.get("skill_name", "")
    died = raw_state.get("player", {}).get("health", 20) <= 0

    base = 0.0
    shaped = 0.0

    # Small penalty for failed skills (agent learns to avoid futile actions)
    if not success:
        shaped -= 0.1

    # Death penalty (scales with stage)
    if died:
        base -= (1.0 + stage * 0.5)

    # Stage-specific rewards
    if stage == 0:
        base, shaped = _stage0_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 1:
        base, shaped = _stage1_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 2:
        base, shaped = _stage2_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 3:
        base, shaped = _stage3_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 4:
        base, shaped = _stage4_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 5:
        base, shaped = _stage5_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 6:
        base, shaped = _stage6_rewards(raw_state, prev_state, skill_result, base, shaped)
    elif stage == 7:
        base, shaped = _stage7_rewards(raw_state, prev_state, skill_result, base, shaped)

    return base, shaped


def _inv_count(state, item):
    return state.get("inventory", {}).get("resources", {}).get(item, 0)


def _has_tool(state, tool):
    return state.get("inventory", {}).get("tools", {}).get(tool, False)


def _inv_delta(cur, prev, item):
    return _inv_count(cur, item) - _inv_count(prev, item)


def _newly_has(cur, prev, tool):
    return _has_tool(cur, tool) and not _has_tool(prev, tool)


# ── Stage 0: Punch Trees ──

def _stage0_rewards(cur, prev, result, base, shaped):
    # +1.0 per log collected
    log_delta = _inv_delta(cur, prev, "log")
    if log_delta > 0:
        base += 1.0 * log_delta

    # +2.0 for crafting first crafting table
    if _newly_has(cur, prev, "has_crafting_table"):
        base += 2.0

    # +3.0 for first wooden pickaxe
    if _newly_has(cur, prev, "has_wooden_pickaxe"):
        base += 3.0

    # Shaped: proximity encouragement for item pickups
    if result.get("items_collected", 0) > 0:
        shaped += 0.1

    return base, shaped


# ── Stage 1: Stone Age ──

def _stage1_rewards(cur, prev, result, base, shaped):
    # Include Stage 0 rewards (cumulative)
    base, shaped = _stage0_rewards(cur, prev, result, base, shaped)

    cobble_delta = _inv_delta(cur, prev, "cobblestone")
    if cobble_delta > 0:
        base += 0.1 * cobble_delta

    if _newly_has(cur, prev, "has_stone_pickaxe"):
        base += 3.0
    if _inv_delta(cur, prev, "coal") > 0:
        shaped += 0.2
    if _newly_has(cur, prev, "has_furnace"):
        base += 2.0

    return base, shaped


# ── Stage 2: Iron Age + Survival ──

def _stage2_rewards(cur, prev, result, base, shaped):
    base, shaped = _stage1_rewards(cur, prev, result, base, shaped)

    iron_delta = _inv_delta(cur, prev, "iron_ingot")
    if iron_delta > 0:
        base += 3.0 * iron_delta

    if _newly_has(cur, prev, "has_iron_pickaxe"):
        base += 5.0
    if _newly_has(cur, prev, "has_iron_sword"):
        base += 3.0

    # Eating reward
    food_delta = cur.get("player", {}).get("food", 20) - prev.get("player", {}).get("food", 20)
    if food_delta > 0 and result.get("skill_name") == "eat_food":
        shaped += 0.5

    # Survival bonus
    shaped += 0.02  # small per-step survival bonus

    return base, shaped


# ── Stage 3: Diamond Hunting ──

def _stage3_rewards(cur, prev, result, base, shaped):
    base, shaped = _stage2_rewards(cur, prev, result, base, shaped)

    diamond_delta = _inv_delta(cur, prev, "diamond")
    if diamond_delta > 0:
        base += 10.0 * diamond_delta

    if _newly_has(cur, prev, "has_diamond_pickaxe"):
        base += 15.0
    if _newly_has(cur, prev, "has_diamond_sword"):
        base += 10.0

    # Depth exploration bonus
    y = cur.get("player", {}).get("position", {}).get("y", 64)
    if y < 16:
        shaped += 0.1  # reward for being at mining depth

    # Combat reward
    if result.get("skill_name") == "attack_nearest_hostile" and result.get("success"):
        shaped += 0.5

    return base, shaped


# ── Stage 4: Nether Preparation ──

def _stage4_rewards(cur, prev, result, base, shaped):
    base, shaped = _stage3_rewards(cur, prev, result, base, shaped)

    obsidian_delta = _inv_delta(cur, prev, "obsidian")
    if obsidian_delta > 0:
        base += 5.0 * obsidian_delta

    # Portal built milestone
    milestones = cur.get("progress", {}).get("milestones", {})
    prev_milestones = prev.get("progress", {}).get("milestones", {})
    if milestones.get("nether_portal_built") and not prev_milestones.get("nether_portal_built"):
        base += 10.0

    return base, shaped


# ── Stage 5: Nether Conquest ──

def _stage5_rewards(cur, prev, result, base, shaped):
    milestones = cur.get("progress", {}).get("milestones", {})
    prev_milestones = prev.get("progress", {}).get("milestones", {})

    # Entering nether for first time
    if milestones.get("entered_nether") and not prev_milestones.get("entered_nether"):
        base += 5.0

    # Finding fortress
    if milestones.get("found_fortress") and not prev_milestones.get("found_fortress"):
        base += 10.0

    # Blaze rods
    rod_delta = _inv_delta(cur, prev, "blaze_rod")
    if rod_delta > 0:
        base += 5.0 * rod_delta

    # Ender pearls
    pearl_delta = _inv_delta(cur, prev, "ender_pearl")
    if pearl_delta > 0:
        base += 5.0 * pearl_delta

    # Bonus for collecting enough
    if milestones.get("blaze_rods_enough") and not prev_milestones.get("blaze_rods_enough"):
        base += 20.0
    if milestones.get("ender_pearls_enough") and not prev_milestones.get("ender_pearls_enough"):
        base += 20.0

    # Shaped: progress toward goals
    rods = _inv_count(cur, "blaze_rod")
    pearls = _inv_count(cur, "ender_pearl")
    shaped += 0.05 * min(rods / 7, 1.0)
    shaped += 0.05 * min(pearls / 12, 1.0)

    return base, shaped


# ── Stage 6: Stronghold ──

def _stage6_rewards(cur, prev, result, base, shaped):
    milestones = cur.get("progress", {}).get("milestones", {})
    prev_milestones = prev.get("progress", {}).get("milestones", {})

    if milestones.get("located_stronghold") and not prev_milestones.get("located_stronghold"):
        base += 15.0

    if milestones.get("end_portal_activated") and not prev_milestones.get("end_portal_activated"):
        base += 10.0

    if milestones.get("entered_end") and not prev_milestones.get("entered_end"):
        base += 20.0

    # Shaped: eye of ender usage moves us closer
    if result.get("skill_name") == "throw_eye_of_ender" and result.get("success"):
        shaped += 1.0

    return base, shaped


# ── Stage 7: Dragon Slayer ──

def _stage7_rewards(cur, prev, result, base, shaped):
    milestones = cur.get("progress", {}).get("milestones", {})
    prev_milestones = prev.get("progress", {}).get("milestones", {})

    # Crystal destruction
    cur_frac = milestones.get("crystals_destroyed_frac", 0.0)
    prev_frac = prev_milestones.get("crystals_destroyed_frac", 0.0)
    if cur_frac > prev_frac:
        base += 10.0 * (cur_frac - prev_frac) * 10  # ~10 per crystal

    # Dragon health reduction
    cur_dragon_hp = cur.get("nearby_entities", {}).get("special", {}).get("dragon_health", 200)
    prev_dragon_hp = prev.get("nearby_entities", {}).get("special", {}).get("dragon_health", 200)
    hp_reduced = prev_dragon_hp - cur_dragon_hp
    if hp_reduced > 0:
        shaped += hp_reduced / 20.0  # scaled

    # Dragon defeated!
    if milestones.get("dragon_defeated") and not prev_milestones.get("dragon_defeated"):
        base += 200.0

    return base, shaped


def check_promotion(stage: int, episode_results: list[dict]) -> bool:
    """Check if the agent should be promoted to the next stage.

    Args:
        stage: Current stage (0-7).
        episode_results: List of recent episode result dicts.

    Returns:
        True if promotion criteria are met.
    """
    if stage >= NUM_STAGES - 1:
        return False

    cfg = STAGES[stage]
    if len(episode_results) < cfg.promotion_window:
        return False

    recent = episode_results[-cfg.promotion_window:]
    successes = sum(1 for ep in recent if ep.get("stage_goal_met", False))
    success_rate = successes / cfg.promotion_window

    return success_rate >= cfg.promotion_threshold


def check_stage_goal(stage: int, raw_state: dict) -> bool:
    """Check if the current episode has met the stage's goal condition."""
    inv = raw_state.get("inventory", {})
    tools = inv.get("tools", {})
    resources = inv.get("resources", {})
    milestones = raw_state.get("progress", {}).get("milestones", {})

    if stage == 0:
        return tools.get("has_wooden_pickaxe", False)
    elif stage == 1:
        return tools.get("has_stone_pickaxe", False) and inv.get("has_furnace", False)
    elif stage == 2:
        return tools.get("has_iron_pickaxe", False) and inv.get("armor", {}).get("chestplate", "none") != "none"
    elif stage == 3:
        return tools.get("has_diamond_pickaxe", False)
    elif stage == 4:
        return milestones.get("nether_portal_built", False)
    elif stage == 5:
        return milestones.get("blaze_rods_enough", False) and milestones.get("ender_pearls_enough", False)
    elif stage == 6:
        return milestones.get("entered_end", False)
    elif stage == 7:
        return milestones.get("dragon_defeated", False)

    return False


def get_shaping_weight(stage_progress: float, stage: int) -> float:
    """Compute reward shaping weight with annealing.

    Args:
        stage_progress: Progress within current stage [0, 1].
        stage: Current stage index.

    Returns:
        Shaping weight to multiply shaped rewards by.
    """
    cfg = STAGES[stage]
    if stage_progress < cfg.shaping_anneal_frac:
        t = stage_progress / cfg.shaping_anneal_frac
        return cfg.shaping_start + (cfg.shaping_end - cfg.shaping_start) * t
    return cfg.shaping_end
