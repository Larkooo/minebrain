"""
Skill registry for MineBrain.

Defines all 72 macro-action skills the agent can execute, organized by category.
Each skill has preconditions (for action masking) and curriculum stage gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import numpy as np


NUM_SKILLS = 72


class SkillCategory(IntEnum):
    GATHERING = 0
    CRAFTING = 1
    SMELTING = 2
    SURVIVAL = 3
    COMBAT = 4
    NAVIGATION = 5
    END_GAME = 6
    UTILITY = 7


@dataclass(frozen=True)
class SkillDef:
    """Definition of a single macro-action skill."""
    id: int
    name: str
    category: SkillCategory
    min_stage: int  # earliest curriculum stage where this skill is available
    timeout_ms: int = 15000  # max execution time before auto-fail
    description: str = ""


# ──────────────────────────────────────────────────────────────
# Skill definitions (72 total)
# ──────────────────────────────────────────────────────────────

SKILLS: list[SkillDef] = [
    # === Gathering (0-11) ===
    SkillDef(0,  "mine_nearest_log",         SkillCategory.GATHERING, 0, 10000, "Pathfind to nearest tree, chop, collect drops"),
    SkillDef(1,  "mine_nearest_stone",       SkillCategory.GATHERING, 1, 10000, "Mine stone blocks (requires wooden pickaxe+)"),
    SkillDef(2,  "mine_nearest_coal_ore",    SkillCategory.GATHERING, 1, 10000, "Mine coal ore (requires wooden pickaxe+)"),
    SkillDef(3,  "mine_nearest_iron_ore",    SkillCategory.GATHERING, 2, 12000, "Mine iron ore (requires stone pickaxe+)"),
    SkillDef(4,  "mine_nearest_gold_ore",    SkillCategory.GATHERING, 3, 12000, "Mine gold ore (requires iron pickaxe+)"),
    SkillDef(5,  "mine_nearest_diamond_ore", SkillCategory.GATHERING, 3, 15000, "Mine diamond ore (requires iron pickaxe+)"),
    SkillDef(6,  "mine_nearest_obsidian",    SkillCategory.GATHERING, 4, 15000, "Mine obsidian (requires diamond pickaxe)"),
    SkillDef(7,  "collect_gravel",           SkillCategory.GATHERING, 4, 8000,  "Dig gravel for flint"),
    SkillDef(8,  "collect_sand",             SkillCategory.GATHERING, 2, 8000,  "Dig sand"),
    SkillDef(9,  "collect_nearest_item_drop",SkillCategory.GATHERING, 0, 5000,  "Pathfind to nearest item entity on ground"),
    SkillDef(10, "mine_nether_quartz",       SkillCategory.GATHERING, 5, 10000, "Mine nether quartz (in nether only)"),
    SkillDef(11, "harvest_crop",             SkillCategory.GATHERING, 2, 8000,  "Harvest nearest crop"),

    # === Crafting (12-27) ===
    SkillDef(12, "craft_planks",             SkillCategory.CRAFTING, 0, 3000, "Craft logs into planks"),
    SkillDef(13, "craft_sticks",             SkillCategory.CRAFTING, 0, 3000, "Craft planks into sticks"),
    SkillDef(14, "craft_crafting_table",     SkillCategory.CRAFTING, 0, 3000, "Craft a crafting table"),
    SkillDef(15, "craft_wooden_pickaxe",     SkillCategory.CRAFTING, 0, 5000, "Craft wooden pickaxe (needs table)"),
    SkillDef(16, "craft_wooden_sword",       SkillCategory.CRAFTING, 0, 5000, "Craft wooden sword (needs table)"),
    SkillDef(17, "craft_stone_pickaxe",      SkillCategory.CRAFTING, 1, 5000, "Craft stone pickaxe"),
    SkillDef(18, "craft_stone_sword",        SkillCategory.CRAFTING, 1, 5000, "Craft stone sword"),
    SkillDef(19, "craft_iron_pickaxe",       SkillCategory.CRAFTING, 2, 5000, "Craft iron pickaxe (needs iron ingots)"),
    SkillDef(20, "craft_iron_sword",         SkillCategory.CRAFTING, 2, 5000, "Craft iron sword"),
    SkillDef(21, "craft_diamond_pickaxe",    SkillCategory.CRAFTING, 3, 5000, "Craft diamond pickaxe"),
    SkillDef(22, "craft_diamond_sword",      SkillCategory.CRAFTING, 3, 5000, "Craft diamond sword"),
    SkillDef(23, "craft_furnace",            SkillCategory.CRAFTING, 1, 5000, "Craft a furnace"),
    SkillDef(24, "craft_bucket",             SkillCategory.CRAFTING, 2, 5000, "Craft iron bucket"),
    SkillDef(25, "craft_iron_armor_set",     SkillCategory.CRAFTING, 2, 8000, "Craft affordable iron armor pieces"),
    SkillDef(26, "craft_diamond_armor_set",  SkillCategory.CRAFTING, 3, 8000, "Craft affordable diamond armor pieces"),
    SkillDef(27, "craft_eye_of_ender",       SkillCategory.CRAFTING, 5, 3000, "Craft eye of ender (blaze powder + pearl)"),

    # === Smelting (28-30) ===
    SkillDef(28, "smelt_iron_ore",           SkillCategory.SMELTING, 2, 12000, "Smelt iron ore in furnace"),
    SkillDef(29, "smelt_gold_ore",           SkillCategory.SMELTING, 3, 12000, "Smelt gold ore in furnace"),
    SkillDef(30, "smelt_food",               SkillCategory.SMELTING, 2, 12000, "Cook raw meat in furnace"),

    # === Survival (31-37) ===
    SkillDef(31, "eat_food",                 SkillCategory.SURVIVAL, 2, 5000,  "Eat best available food item"),
    SkillDef(32, "sleep_in_bed",             SkillCategory.SURVIVAL, 2, 8000,  "Place bed if needed, sleep through night"),
    SkillDef(33, "place_crafting_table",     SkillCategory.SURVIVAL, 1, 3000,  "Place crafting table nearby"),
    SkillDef(34, "place_furnace",            SkillCategory.SURVIVAL, 1, 3000,  "Place furnace nearby"),
    SkillDef(35, "build_simple_shelter",     SkillCategory.SURVIVAL, 2, 15000, "Build 3x3x3 cobblestone shelter"),
    SkillDef(36, "retreat_from_danger",      SkillCategory.SURVIVAL, 3, 8000,  "Run away from nearest hostile mob"),
    SkillDef(37, "equip_best_gear",          SkillCategory.SURVIVAL, 2, 3000,  "Auto-equip best armor + weapon"),

    # === Combat (38-45) ===
    SkillDef(38, "attack_nearest_hostile",   SkillCategory.COMBAT, 3, 15000, "Melee attack nearest hostile mob"),
    SkillDef(39, "hunt_nearest_animal",      SkillCategory.COMBAT, 2, 10000, "Kill nearest animal for food drops"),
    SkillDef(40, "fight_blaze",              SkillCategory.COMBAT, 5, 15000, "Engage blaze with ranged/melee"),
    SkillDef(41, "fight_enderman",           SkillCategory.COMBAT, 5, 15000, "Careful enderman combat for pearl"),
    SkillDef(42, "trade_with_villager",      SkillCategory.COMBAT, 5, 10000, "Trade with villager for useful items"),
    SkillDef(43, "kill_end_crystal",         SkillCategory.COMBAT, 7, 10000, "Destroy an end crystal"),
    SkillDef(44, "attack_ender_dragon",      SkillCategory.COMBAT, 7, 15000, "Melee dragon when perched"),
    SkillDef(45, "use_bed_on_dragon",        SkillCategory.COMBAT, 7, 10000, "Bed explosion strategy on dragon"),

    # === Navigation (46-55) ===
    SkillDef(46, "explore_randomly",         SkillCategory.NAVIGATION, 0, 10000, "Wander in unexplored direction"),
    SkillDef(47, "explore_for_cave",         SkillCategory.NAVIGATION, 3, 12000, "Find cave entrance nearby"),
    SkillDef(48, "go_to_nearest_village",    SkillCategory.NAVIGATION, 4, 15000, "Navigate to nearest village"),
    SkillDef(49, "descend_to_mining_level",  SkillCategory.NAVIGATION, 3, 15000, "Dig staircase down to y=11"),
    SkillDef(50, "ascend_to_surface",        SkillCategory.NAVIGATION, 3, 15000, "Pillar/staircase up to surface"),
    SkillDef(51, "build_nether_portal",      SkillCategory.NAVIGATION, 4, 30000, "Build nether portal with obsidian"),
    SkillDef(52, "enter_nether_portal",      SkillCategory.NAVIGATION, 5, 10000, "Step into nether portal"),
    SkillDef(53, "find_nether_fortress",     SkillCategory.NAVIGATION, 5, 30000, "Navigate in nether toward fortress"),
    SkillDef(54, "throw_eye_of_ender",       SkillCategory.NAVIGATION, 6, 8000,  "Throw eye to locate stronghold"),
    SkillDef(55, "go_to_stronghold",         SkillCategory.NAVIGATION, 6, 30000, "Navigate to stronghold location"),

    # === End Game (56-61) ===
    SkillDef(56, "activate_end_portal",      SkillCategory.END_GAME, 6, 10000, "Place eyes in end portal frames"),
    SkillDef(57, "enter_end_portal",         SkillCategory.END_GAME, 6, 8000,  "Jump into end portal"),
    SkillDef(58, "destroy_all_crystals",     SkillCategory.END_GAME, 7, 60000, "Systematic crystal destruction"),
    SkillDef(59, "fight_dragon_cycle",       SkillCategory.END_GAME, 7, 60000, "Full dragon fight loop"),
    SkillDef(60, "pillar_to_crystal",        SkillCategory.END_GAME, 7, 15000, "Tower up to caged crystals"),
    SkillDef(61, "collect_dragon_egg",       SkillCategory.END_GAME, 7, 10000, "Victory lap - collect egg"),

    # === Utility (62-71) ===
    SkillDef(62, "wait_idle",                SkillCategory.UTILITY, 0, 5000,  "Do nothing for a few seconds"),
    SkillDef(63, "look_around",              SkillCategory.UTILITY, 0, 3000,  "Scan surroundings, update observations"),
    SkillDef(64, "drop_junk_items",          SkillCategory.UTILITY, 2, 3000,  "Clear inventory of useless items"),
    SkillDef(65, "organize_inventory",       SkillCategory.UTILITY, 2, 5000,  "Sort inventory for efficiency"),
    SkillDef(66, "place_torch",              SkillCategory.UTILITY, 1, 3000,  "Place torch to light up area"),
    SkillDef(67, "fill_bucket_with_water",   SkillCategory.UTILITY, 4, 8000,  "Fill bucket from water source"),
    SkillDef(68, "fill_bucket_with_lava",    SkillCategory.UTILITY, 4, 8000,  "Fill bucket from lava source"),
    SkillDef(69, "create_infinite_water",    SkillCategory.UTILITY, 4, 10000, "Create infinite water source"),
    SkillDef(70, "dig_down_one",             SkillCategory.UTILITY, 1, 5000,  "Controlled single block dig below"),
    SkillDef(71, "pillar_up_one",            SkillCategory.UTILITY, 1, 5000,  "Place block below feet to go up"),
]

# Validate IDs are contiguous 0..71
assert len(SKILLS) == NUM_SKILLS
assert all(s.id == i for i, s in enumerate(SKILLS))

# Lookup by name
SKILL_BY_NAME: dict[str, SkillDef] = {s.name: s for s in SKILLS}

# Pre-compute stage masks: stage_masks[stage] is a bool array of which skills are available
NUM_STAGES = 8
STAGE_SKILL_MASKS = np.zeros((NUM_STAGES, NUM_SKILLS), dtype=np.bool_)
for stage in range(NUM_STAGES):
    for skill in SKILLS:
        if skill.min_stage <= stage:
            STAGE_SKILL_MASKS[stage, skill.id] = True


def get_skills_for_stage(stage: int) -> list[SkillDef]:
    """Return all skills available at a given curriculum stage."""
    return [s for s in SKILLS if s.min_stage <= stage]


def get_stage_mask(stage: int) -> np.ndarray:
    """Return the stage-level action mask (72-element bool array)."""
    return STAGE_SKILL_MASKS[min(stage, NUM_STAGES - 1)].copy()
