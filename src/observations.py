"""
Observation feature vector encoding for MineBrain.

Converts raw Minecraft game state (from Mineflayer via bridge) into a
normalized ~310-feature vector, following the same philosophy as nums-ai's
83-feature observation: hand-engineered, well-documented, all in [0, 1].
"""

import numpy as np

OBS_SIZE = 310
FRAME_STACK = 3
STACKED_OBS_SIZE = OBS_SIZE * FRAME_STACK

# ──────────────────────────────────────────────────────────────
# Feature index documentation
# ──────────────────────────────────────────────────────────────
#
# Section 1: Player Vitals [0:8]
#   0  health / 20
#   1  food / 20
#   2  saturation / 20
#   3  armor_value / 20
#   4  experience_level / 50
#   5  is_on_fire (binary)
#   6  is_in_water (binary)
#   7  is_raining (binary)
#
# Section 2: Position & World [8:20]
#   8  x / 10000 (clamped to [-1, 1], shifted to [0, 1])
#   9  y / 256
#   10 z / 10000 (clamped, shifted)
#   11 time_of_day / 24000
#   12 is_daytime (binary)
#   13 is_in_nether (binary)
#   14 is_in_end (binary)
#   15 is_in_overworld (binary)
#   16 light_level / 15
#   17 biome_temperature (0-1)
#   18 distance_from_spawn / 10000 (clamped)
#   19 y_relative_to_sea_level: (y - 63) / 128 + 0.5 (clamped [0,1])
#
# Section 3: Inventory Summary [20:60]
#   20:30 Tool tier indicators (10 binary features)
#   30:50 Resource counts (20 features, each count / 64 clamped to 1.0)
#   50:60 Armor & equipment (10 features)
#
# Section 4: Nearby Blocks [60:100]
#   60:80 Block type densities in 16-block radius (20 features)
#   80:100 Structural features (20 features)
#
# Section 5: Nearby Entities [100:130]
#   100:110 Hostile mob encoding (10 features)
#   110:120 Passive mob encoding (10 features)
#   120:130 Special entities (10 features)
#
# Section 6: Curriculum & Progress [130:150]
#   130:138 Curriculum stage one-hot (8 features)
#   138:150 Milestone flags + progress (12 features)
#
# Section 7: Derived Features [150:180]
#   150:180 Computed signals (30 features)
#
# Section 8: Spatial Grid [180:310]
#   180:305 5x5x5 coarse block categories (125 features)
#   305:310 Padding / reserved (5 features)
# ──────────────────────────────────────────────────────────────

# Resource names for inventory encoding (indices 30:50)
TRACKED_RESOURCES = [
    "log", "planks", "stick", "cobblestone", "iron_ore",
    "iron_ingot", "gold_ingot", "diamond", "coal", "food",
    "wool", "obsidian", "ender_pearl", "blaze_rod", "blaze_powder",
    "eye_of_ender", "flint", "gravel", "bucket", "string",
]

# Block types for density encoding (indices 60:80)
TRACKED_BLOCKS = [
    "log", "stone", "iron_ore", "diamond_ore", "coal_ore",
    "gold_ore", "water", "lava", "sand", "gravel",
    "dirt", "grass_block", "nether_portal", "obsidian", "nether_bricks",
    "soul_sand", "end_stone", "bedrock", "wheat", "chest",
]

# Spatial grid block categories
BLOCK_CATEGORY_AIR = 0
BLOCK_CATEGORY_SOLID = 1
BLOCK_CATEGORY_LIQUID = 2
BLOCK_CATEGORY_ORE = 3
BLOCK_CATEGORY_ENTITY = 4
NUM_BLOCK_CATEGORIES = 5


def encode_observation(raw_state: dict, stage: int = 0) -> np.ndarray:
    """Encode raw game state dict into a 310-element float32 observation vector.

    Args:
        raw_state: Dict from Mineflayer bot with game state fields.
        stage: Current curriculum stage (0-7).

    Returns:
        np.ndarray of shape (310,) with values in [0, 1].
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # --- Section 1: Player Vitals [0:8] ---
    player = raw_state.get("player", {})
    obs[0] = _clamp01(player.get("health", 20) / 20)
    obs[1] = _clamp01(player.get("food", 20) / 20)
    obs[2] = _clamp01(player.get("saturation", 5) / 20)
    obs[3] = _clamp01(player.get("armor", 0) / 20)
    obs[4] = _clamp01(player.get("xp_level", 0) / 50)
    obs[5] = float(player.get("is_on_fire", False))
    obs[6] = float(player.get("is_in_water", False))
    obs[7] = float(raw_state.get("is_raining", False))

    # --- Section 2: Position & World [8:20] ---
    pos = player.get("position", {"x": 0, "y": 64, "z": 0})
    obs[8] = _clamp01((pos.get("x", 0) / 10000) * 0.5 + 0.5)
    obs[9] = _clamp01(pos.get("y", 64) / 256)
    obs[10] = _clamp01((pos.get("z", 0) / 10000) * 0.5 + 0.5)

    world = raw_state.get("world", {})
    obs[11] = _clamp01(world.get("time_of_day", 0) / 24000)
    obs[12] = float(world.get("is_daytime", True))

    dimension = world.get("dimension", "overworld")
    obs[13] = float(dimension == "the_nether")
    obs[14] = float(dimension == "the_end")
    obs[15] = float(dimension == "overworld")
    obs[16] = _clamp01(world.get("light_level", 15) / 15)
    obs[17] = _clamp01(world.get("biome_temperature", 0.5))

    spawn = raw_state.get("spawn_point", {"x": 0, "y": 64, "z": 0})
    dx = pos.get("x", 0) - spawn.get("x", 0)
    dz = pos.get("z", 0) - spawn.get("z", 0)
    dist_spawn = (dx**2 + dz**2) ** 0.5
    obs[18] = _clamp01(dist_spawn / 10000)
    obs[19] = _clamp01((pos.get("y", 64) - 63) / 128 + 0.5)

    # --- Section 3: Inventory Summary [20:60] ---
    inv = raw_state.get("inventory", {})

    # Tool tier indicators [20:30]
    tools = inv.get("tools", {})
    obs[20] = float(tools.get("has_wooden_pickaxe", False))
    obs[21] = float(tools.get("has_stone_pickaxe", False))
    obs[22] = float(tools.get("has_iron_pickaxe", False))
    obs[23] = float(tools.get("has_diamond_pickaxe", False))
    obs[24] = float(tools.get("has_wooden_sword", False))
    obs[25] = float(tools.get("has_stone_sword", False))
    obs[26] = float(tools.get("has_iron_sword", False))
    obs[27] = float(tools.get("has_diamond_sword", False))
    obs[28] = float(tools.get("has_axe", False))
    obs[29] = float(tools.get("has_shovel", False))

    # Resource counts [30:50]
    resources = inv.get("resources", {})
    for i, res_name in enumerate(TRACKED_RESOURCES):
        obs[30 + i] = _clamp01(resources.get(res_name, 0) / 64)

    # Armor & equipment [50:60]
    armor = inv.get("armor", {})
    tier_map = {"none": 0.0, "leather": 0.25, "iron": 0.5, "diamond": 0.75, "netherite": 1.0}
    obs[50] = tier_map.get(armor.get("helmet", "none"), 0.0)
    obs[51] = tier_map.get(armor.get("chestplate", "none"), 0.0)
    obs[52] = tier_map.get(armor.get("leggings", "none"), 0.0)
    obs[53] = tier_map.get(armor.get("boots", "none"), 0.0)
    obs[54] = float(armor.get("has_shield", False))
    obs[55] = _clamp01(armor.get("total_armor_points", 0) / 20)
    obs[56] = _clamp01(inv.get("empty_slots", 36) / 36)
    obs[57] = float(inv.get("has_crafting_table", False))
    obs[58] = float(inv.get("has_furnace", False))
    obs[59] = float(inv.get("has_bed", False))

    # --- Section 4: Nearby Blocks [60:100] ---
    blocks = raw_state.get("nearby_blocks", {})

    # Block densities [60:80]
    densities = blocks.get("densities", {})
    for i, block_name in enumerate(TRACKED_BLOCKS):
        obs[60 + i] = _clamp01(densities.get(block_name, 0.0))

    # Structural features [80:100]
    structure = blocks.get("structure", {})
    obs[80] = _clamp01(structure.get("nearest_tree_dist", 32) / 32)
    obs[81] = _clamp01(structure.get("nearest_cave_dist", 32) / 32)
    obs[82] = _clamp01(structure.get("nearest_water_dist", 32) / 32)
    obs[83] = _clamp01(structure.get("nearest_lava_dist", 32) / 32)
    obs[84] = _clamp01(structure.get("blocks_below_to_void", 64) / 64)
    obs[85] = float(structure.get("open_sky_above", True))
    obs[86] = float(structure.get("is_underground", False))
    obs[87] = _clamp01(structure.get("nearest_village_dist", 200) / 200)
    obs[88] = _clamp01(structure.get("nearest_fortress_dist", 100) / 100)
    obs[89] = float(structure.get("stronghold_located", False))

    # Directional composition [90:95] (what block type is ahead/left/right/above/below)
    directions = structure.get("directional_blocks", [0.5] * 5)
    for i, v in enumerate(directions[:5]):
        obs[90 + i] = _clamp01(v)

    # Directional danger [95:100] (lava/void/mob proximity)
    danger = structure.get("directional_danger", [0.0] * 5)
    for i, v in enumerate(danger[:5]):
        obs[95 + i] = _clamp01(v)

    # --- Section 5: Nearby Entities [100:130] ---
    entities = raw_state.get("nearby_entities", {})

    # Hostile mobs [100:110]
    hostile = entities.get("hostile", {})
    obs[100] = _clamp01(hostile.get("nearest_dist", 32) / 32)
    obs[101] = _clamp01(hostile.get("nearest_health", 0) / 20)
    obs[102] = _clamp01(hostile.get("count_within_8", 0) / 10)
    obs[103] = _clamp01(hostile.get("count_within_16", 0) / 20)
    obs[104] = float(hostile.get("zombie_nearby", False))
    obs[105] = float(hostile.get("skeleton_nearby", False))
    obs[106] = float(hostile.get("creeper_nearby", False))
    obs[107] = float(hostile.get("enderman_nearby", False))
    obs[108] = float(hostile.get("blaze_nearby", False))
    obs[109] = float(hostile.get("ghast_nearby", False))

    # Passive mobs [110:120]
    passive = entities.get("passive", {})
    obs[110] = _clamp01(passive.get("nearest_animal_dist", 32) / 32)
    obs[111] = _clamp01(passive.get("animal_count_within_16", 0) / 20)
    obs[112] = float(passive.get("cow_nearby", False))
    obs[113] = float(passive.get("pig_nearby", False))
    obs[114] = float(passive.get("sheep_nearby", False))
    obs[115] = float(passive.get("chicken_nearby", False))
    obs[116] = float(passive.get("horse_nearby", False))
    obs[117] = float(passive.get("rabbit_nearby", False))
    obs[118] = _clamp01(passive.get("nearest_villager_dist", 32) / 32)
    obs[119] = _clamp01(passive.get("villager_count", 0) / 10)

    # Special entities [120:130]
    special = entities.get("special", {})
    obs[120] = float(special.get("ender_dragon_alive", False))
    obs[121] = _clamp01(special.get("dragon_health", 0) / 200)
    obs[122] = _clamp01(special.get("dragon_distance", 100) / 100)
    obs[123] = _clamp01(special.get("end_crystal_count", 0) / 10)
    obs[124] = _clamp01(special.get("nearest_crystal_dist", 100) / 100)
    obs[125] = _clamp01(special.get("nearest_item_drop_dist", 16) / 16)
    obs[126] = _clamp01(special.get("item_drops_count", 0) / 20)
    obs[127] = _clamp01(special.get("nearest_player_dist", 64) / 64)
    obs[128] = _clamp01(special.get("nearest_blaze_spawner_dist", 32) / 32)
    obs[129] = float(special.get("near_end_portal", False))

    # --- Section 6: Curriculum & Progress [130:150] ---
    # Stage one-hot [130:138]
    if 0 <= stage < 8:
        obs[130 + stage] = 1.0

    progress = raw_state.get("progress", {})
    obs[138] = _clamp01(progress.get("stage_progress", 0.0))
    obs[139] = _clamp01(stage / 7)

    # Milestone flags [140:150]
    milestones = progress.get("milestones", {})
    obs[140] = float(milestones.get("entered_nether", False))
    obs[141] = float(milestones.get("found_fortress", False))
    obs[142] = float(milestones.get("blaze_rods_enough", False))
    obs[143] = float(milestones.get("ender_pearls_enough", False))
    obs[144] = float(milestones.get("located_stronghold", False))
    obs[145] = float(milestones.get("entered_end", False))
    obs[146] = _clamp01(milestones.get("crystals_destroyed_frac", 0.0))
    obs[147] = float(milestones.get("nether_portal_built", False))
    obs[148] = float(milestones.get("end_portal_activated", False))
    obs[149] = float(milestones.get("dragon_defeated", False))

    # --- Section 7: Derived Features [150:180] ---
    derived = raw_state.get("derived", {})
    obs[150] = float(derived.get("can_craft_anything", False))
    obs[151] = float(derived.get("can_smelt_anything", False))

    # Tool/weapon tier as scalar
    pickaxe_tier = 0
    if obs[23]:
        pickaxe_tier = 4
    elif obs[22]:
        pickaxe_tier = 3
    elif obs[21]:
        pickaxe_tier = 2
    elif obs[20]:
        pickaxe_tier = 1
    obs[152] = pickaxe_tier / 4

    sword_tier = 0
    if obs[27]:
        sword_tier = 4
    elif obs[26]:
        sword_tier = 3
    elif obs[25]:
        sword_tier = 2
    elif obs[24]:
        sword_tier = 1
    obs[153] = sword_tier / 4

    # Combat power estimate
    obs[154] = _clamp01((sword_tier * 2 + obs[55] * 20 + obs[0] * 20) / 50)

    obs[155] = _clamp01(derived.get("time_since_progress", 0) / 300)
    obs[156] = _clamp01(derived.get("valid_actions_count", 0) / 72)
    obs[157] = 1.0 if obs[1] < 0.3 else (0.5 if obs[1] < 0.6 else 0.0)  # food urgency
    obs[158] = 1.0 if obs[0] < 0.3 else (0.5 if obs[0] < 0.6 else 0.0)  # health urgency

    # Night danger: high if nighttime + on surface + no shelter
    is_night = obs[12] < 0.5
    on_surface = not obs[86]
    obs[159] = float(is_night and on_surface) * (1.0 - obs[16])  # worse with low light

    # Skill success recency [160:170]
    recency = derived.get("skill_recency", [0.0] * 10)
    for i, v in enumerate(recency[:10]):
        obs[160 + i] = _clamp01(v)

    # Reserved [170:180]
    # Left as zeros for future expansion

    # --- Section 8: Spatial Grid [180:310] ---
    # 5x5x5 grid = 125 values, each is a block category (0-4) normalized to [0, 1]
    grid = raw_state.get("spatial_grid", [])
    for i, v in enumerate(grid[:125]):
        obs[180 + i] = _clamp01(v / NUM_BLOCK_CATEGORIES)

    # Reserved [305:310] - zeros

    return obs


def _clamp01(v: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, float(v)))


class FrameStacker:
    """Maintains a rolling window of observations for frame stacking."""

    def __init__(self, n_frames: int = FRAME_STACK, obs_size: int = OBS_SIZE):
        self.n_frames = n_frames
        self.obs_size = obs_size
        self.frames = np.zeros((n_frames, obs_size), dtype=np.float32)

    def reset(self, initial_obs: np.ndarray) -> np.ndarray:
        """Reset with initial observation (duplicated across all frames)."""
        for i in range(self.n_frames):
            self.frames[i] = initial_obs
        return self.get()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Add new observation, return stacked frames."""
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = obs
        return self.get()

    def get(self) -> np.ndarray:
        """Return flattened stacked observation."""
        return self.frames.flatten()
