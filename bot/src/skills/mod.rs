//! Skill registry and execution.
//!
//! Maps action IDs (0-71) to skill implementations.
//! Each skill is an async function that operates the bot via Azalea's API.

pub mod gathering;
pub mod crafting;
pub mod combat;
pub mod survival;
pub mod navigation;
pub mod nether;
pub mod end;

use azalea::prelude::*;
use serde::Serialize;
use std::time::{Duration, Instant};

pub const CORE_SKILL_COUNT: usize = 72;

/// Minimum curriculum stage for each skill (mirrors src/skills.py).
const SKILL_MIN_STAGE: [u8; CORE_SKILL_COUNT] = [
    // Gathering (0-11)
    0, 1, 1, 2, 3, 3, 4, 4, 2, 0, 5, 2,
    // Crafting (12-27)
    0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 2, 3, 5,
    // Smelting (28-30)
    2, 3, 2,
    // Survival (31-37)
    2, 2, 1, 1, 2, 3, 2,
    // Combat (38-45)
    3, 2, 5, 5, 5, 7, 7, 7,
    // Navigation (46-55)
    0, 3, 4, 3, 3, 4, 5, 5, 6, 6,
    // End game (56-61)
    6, 6, 7, 7, 7, 7,
    // Utility (62-71)
    0, 0, 2, 2, 1, 4, 4, 4, 1, 1,
];

/// Skill names for each action ID.
const SKILL_NAMES: [&str; CORE_SKILL_COUNT] = [
    // Gathering
    "mine_nearest_log", "mine_nearest_stone", "mine_nearest_coal_ore",
    "mine_nearest_iron_ore", "mine_nearest_gold_ore", "mine_nearest_diamond_ore",
    "mine_nearest_obsidian", "collect_gravel", "collect_sand",
    "collect_nearest_item_drop", "mine_nether_quartz", "harvest_crop",
    // Crafting
    "craft_planks", "craft_sticks", "craft_crafting_table",
    "craft_wooden_pickaxe", "craft_wooden_sword", "craft_stone_pickaxe",
    "craft_stone_sword", "craft_iron_pickaxe", "craft_iron_sword",
    "craft_diamond_pickaxe", "craft_diamond_sword", "craft_furnace",
    "craft_bucket", "craft_iron_armor_set", "craft_diamond_armor_set",
    "craft_eye_of_ender",
    // Smelting
    "smelt_iron_ore", "smelt_gold_ore", "smelt_food",
    // Survival
    "eat_food", "sleep_in_bed", "place_crafting_table", "place_furnace",
    "build_simple_shelter", "retreat_from_danger", "equip_best_gear",
    // Combat
    "attack_nearest_hostile", "hunt_nearest_animal", "fight_blaze",
    "fight_enderman", "trade_with_villager", "kill_end_crystal",
    "attack_ender_dragon", "use_bed_on_dragon",
    // Navigation
    "explore_randomly", "explore_for_cave", "go_to_nearest_village",
    "descend_to_mining_level", "ascend_to_surface", "build_nether_portal",
    "enter_nether_portal", "find_nether_fortress", "throw_eye_of_ender",
    "go_to_stronghold",
    // End game
    "activate_end_portal", "enter_end_portal", "destroy_all_crystals",
    "fight_dragon_cycle", "pillar_to_crystal", "collect_dragon_egg",
    // Utility
    "wait_idle", "look_around", "drop_junk_items", "organize_inventory",
    "place_torch", "fill_bucket_with_water", "fill_bucket_with_lava",
    "create_infinite_water", "dig_down_one", "pillar_up_one",
];

#[derive(Debug, Clone, Serialize)]
pub struct SkillResult {
    pub success: bool,
    pub skill_name: String,
    pub reason: String,
    pub duration_ms: u64,
    pub items_collected: u32,
}

impl SkillResult {
    pub fn ok(name: &str) -> Self {
        Self {
            success: true,
            skill_name: name.to_string(),
            reason: "ok".to_string(),
            duration_ms: 0,
            items_collected: 0,
        }
    }

    pub fn failure(name: &str, reason: &str) -> Self {
        Self {
            success: false,
            skill_name: name.to_string(),
            reason: reason.to_string(),
            duration_ms: 0,
            items_collected: 0,
        }
    }
}

/// Execute a skill by action ID.
pub async fn execute_skill(bot: &Client, action: u32, stage: u8) -> SkillResult {
    let action = action as usize;

    if action >= CORE_SKILL_COUNT {
        return SkillResult::failure("unknown", "action ID out of range");
    }

    // Stage gating
    if SKILL_MIN_STAGE[action] > stage {
        return SkillResult::failure(
            SKILL_NAMES[action],
            &format!("locked until stage {}", SKILL_MIN_STAGE[action]),
        );
    }

    let start = Instant::now();
    let timeout = Duration::from_secs(15);

    let result = tokio::time::timeout(timeout, dispatch_skill(bot, action)).await;

    let mut skill_result = match result {
        Ok(r) => r,
        Err(_) => SkillResult::failure(SKILL_NAMES[action], "skill timeout"),
    };

    skill_result.duration_ms = start.elapsed().as_millis() as u64;
    skill_result
}

/// Dispatch to the correct skill implementation.
async fn dispatch_skill(bot: &Client, action: usize) -> SkillResult {
    match action {
        // Gathering
        0 => gathering::mine_nearest_log(bot).await,
        1 => gathering::mine_nearest_stone(bot).await,
        2 => gathering::mine_nearest_coal_ore(bot).await,
        3 => gathering::mine_nearest_iron_ore(bot).await,
        4 => gathering::mine_nearest_gold_ore(bot).await,
        5 => gathering::mine_nearest_diamond_ore(bot).await,
        6 => gathering::mine_nearest_obsidian(bot).await,
        7 => gathering::collect_gravel(bot).await,
        8 => gathering::collect_sand(bot).await,
        9 => gathering::collect_nearest_item_drop(bot).await,
        10 => gathering::mine_nether_quartz(bot).await,
        11 => gathering::harvest_crop(bot).await,
        // Crafting
        12 => crafting::craft_planks(bot).await,
        13 => crafting::craft_sticks(bot).await,
        14 => crafting::craft_crafting_table(bot).await,
        15 => crafting::craft_wooden_pickaxe(bot).await,
        16 => crafting::craft_wooden_sword(bot).await,
        17 => crafting::craft_stone_pickaxe(bot).await,
        18 => crafting::craft_stone_sword(bot).await,
        19 => crafting::craft_iron_pickaxe(bot).await,
        20 => crafting::craft_iron_sword(bot).await,
        21 => crafting::craft_diamond_pickaxe(bot).await,
        22 => crafting::craft_diamond_sword(bot).await,
        23 => crafting::craft_furnace(bot).await,
        24 => crafting::craft_bucket(bot).await,
        25 => crafting::craft_iron_armor_set(bot).await,
        26 => crafting::craft_diamond_armor_set(bot).await,
        27 => crafting::craft_eye_of_ender(bot).await,
        // Smelting
        28 => crafting::smelt_iron_ore(bot).await,
        29 => crafting::smelt_gold_ore(bot).await,
        30 => crafting::smelt_food(bot).await,
        // Survival
        31 => survival::eat_food(bot).await,
        32 => survival::sleep_in_bed(bot).await,
        33 => survival::place_crafting_table(bot).await,
        34 => survival::place_furnace(bot).await,
        35 => survival::build_simple_shelter(bot).await,
        36 => survival::retreat_from_danger(bot).await,
        37 => survival::equip_best_gear(bot).await,
        // Combat
        38 => combat::attack_nearest_hostile(bot).await,
        39 => combat::hunt_nearest_animal(bot).await,
        40 => combat::fight_blaze(bot).await,
        41 => combat::fight_enderman(bot).await,
        42 => combat::trade_with_villager(bot).await,
        43 => end::kill_end_crystal(bot).await,
        44 => end::attack_ender_dragon(bot).await,
        45 => end::use_bed_on_dragon(bot).await,
        // Navigation
        46 => navigation::explore_randomly(bot).await,
        47 => navigation::explore_for_cave(bot).await,
        48 => navigation::go_to_nearest_village(bot).await,
        49 => navigation::descend_to_mining_level(bot).await,
        50 => navigation::ascend_to_surface(bot).await,
        51 => nether::build_nether_portal(bot).await,
        52 => nether::enter_nether_portal(bot).await,
        53 => nether::find_nether_fortress(bot).await,
        54 => navigation::throw_eye_of_ender(bot).await,
        55 => navigation::go_to_stronghold(bot).await,
        // End game
        56 => end::activate_end_portal(bot).await,
        57 => end::enter_end_portal(bot).await,
        58 => end::destroy_all_crystals(bot).await,
        59 => end::fight_dragon_cycle(bot).await,
        60 => end::pillar_to_crystal(bot).await,
        61 => end::collect_dragon_egg(bot).await,
        // Utility
        62 => survival::wait_idle(bot).await,
        63 => survival::look_around(bot).await,
        64 => survival::drop_junk_items(bot).await,
        65 => survival::organize_inventory(bot).await,
        66 => survival::place_torch(bot).await,
        67 => navigation::fill_bucket_with_water(bot).await,
        68 => navigation::fill_bucket_with_lava(bot).await,
        69 => navigation::create_infinite_water(bot).await,
        70 => navigation::dig_down_one(bot).await,
        71 => navigation::pillar_up_one(bot).await,
        _ => SkillResult::failure("unknown", "invalid action"),
    }
}

/// Compute action mask based on bot state and curriculum stage.
pub fn get_action_mask(bot: &Client, stage: u8) -> Vec<bool> {
    let mut mask = vec![false; CORE_SKILL_COUNT];

    let menu = bot.menu();
    let slots = menu.slots();

    // Build inventory lookup
    let mut inv: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for slot in slots.iter() {
        if let azalea::inventory::ItemStack::Present(item) = slot {
            let name = format!("{:?}", item.kind).to_lowercase();
            *inv.entry(name).or_insert(0) += item.count as u32;
        }
    }

    let has = |name: &str| -> bool {
        inv.iter().any(|(k, v)| k.contains(name) && *v > 0)
    };
    let count = |name: &str| -> u32 {
        inv.iter()
            .filter(|(k, _)| k.contains(name))
            .map(|(_, v)| v)
            .sum()
    };

    let pickaxe_tier = if has("diamond_pickaxe") || has("netherite_pickaxe") {
        4
    } else if has("iron_pickaxe") {
        3
    } else if has("stone_pickaxe") {
        2
    } else if has("wooden_pickaxe") {
        1
    } else {
        0
    };

    let world_name = bot.component::<azalea::world::InstanceName>().to_string();
    let is_overworld = !world_name.contains("nether") && !world_name.contains("end");
    let is_nether = world_name.contains("nether");
    let is_end = world_name.contains("end");

    // Gathering (0-11)
    mask[0] = is_overworld;
    mask[1] = pickaxe_tier >= 1;
    mask[2] = pickaxe_tier >= 1;
    mask[3] = pickaxe_tier >= 2;
    mask[4] = pickaxe_tier >= 3;
    mask[5] = pickaxe_tier >= 3;
    mask[6] = pickaxe_tier >= 4;
    mask[7] = true;
    mask[8] = true;
    mask[9] = true;
    mask[10] = is_nether && pickaxe_tier >= 1;
    mask[11] = is_overworld;

    // Crafting (12-27)
    mask[12] = has("log");
    let total_planks = count("planks");
    mask[13] = total_planks >= 2;
    mask[14] = total_planks >= 4;
    mask[15] = total_planks >= 3 && count("stick") >= 2;
    mask[16] = total_planks >= 2 && count("stick") >= 1;
    mask[17] = count("cobblestone") >= 3 && count("stick") >= 2;
    mask[18] = count("cobblestone") >= 2 && count("stick") >= 1;
    mask[19] = count("iron_ingot") >= 3 && count("stick") >= 2;
    mask[20] = count("iron_ingot") >= 2 && count("stick") >= 1;
    mask[21] = count("diamond") >= 3 && count("stick") >= 2;
    mask[22] = count("diamond") >= 2 && count("stick") >= 1;
    mask[23] = count("cobblestone") >= 8;
    mask[24] = count("iron_ingot") >= 3;
    mask[25] = count("iron_ingot") >= 4;
    mask[26] = count("diamond") >= 4;
    mask[27] = count("blaze_powder") >= 1 && count("ender_pearl") >= 1;

    // Smelting (28-30)
    let has_fuel = count("coal") > 0 || has("planks");
    mask[28] = has("raw_iron") && has_fuel;
    mask[29] = has("raw_gold") && has_fuel;
    mask[30] = (has("beef") || has("porkchop") || has("chicken") || has("potato")) && has_fuel;

    // Survival (31-37)
    mask[31] = has("bread") || has("apple") || has("cooked") || has("carrot") || has("rotten_flesh");
    mask[32] = is_overworld;
    mask[33] = has("crafting_table");
    mask[34] = has("furnace");
    mask[35] = count("cobblestone") >= 20;
    mask[36] = true;
    mask[37] = true;

    // Combat (38-45)
    mask[38] = true;
    mask[39] = true;
    mask[40] = is_nether;
    mask[41] = true;
    mask[42] = true;
    mask[43] = is_end;
    mask[44] = is_end;
    mask[45] = is_end && has("bed");

    // Navigation (46-55)
    mask[46] = true;
    mask[47] = is_overworld;
    mask[48] = is_overworld;
    mask[49] = is_overworld;
    mask[50] = true;
    mask[51] = count("obsidian") >= 10;
    mask[52] = true;
    mask[53] = is_nether;
    mask[54] = has("ender_eye") || has("eye_of_ender");
    mask[55] = is_overworld;

    // End game (56-61)
    mask[56] = count("ender_eye") + count("eye_of_ender") >= 12;
    mask[57] = true;
    mask[58] = is_end;
    mask[59] = is_end;
    mask[60] = is_end;
    mask[61] = is_end;

    // Utility (62-71)
    mask[62] = true;
    mask[63] = true;
    mask[64] = true;
    mask[65] = true;
    mask[66] = has("torch");
    mask[67] = has("bucket") && is_overworld;
    mask[68] = has("bucket");
    mask[69] = count("water_bucket") >= 2;
    mask[70] = pickaxe_tier >= 1;
    mask[71] = count("cobblestone") + count("dirt") >= 1;

    // Apply stage gating
    for i in 0..CORE_SKILL_COUNT {
        if SKILL_MIN_STAGE[i] > stage {
            mask[i] = false;
        }
    }

    mask
}
