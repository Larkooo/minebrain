//! Observation extraction from Azalea bot state.
//!
//! Collects raw game state and packages it as JSON that the Python
//! observation encoder converts into the 310-feature vector.

use azalea::prelude::*;
use azalea::BlockPos;
use serde_json::{json, Value};

/// Collect the full observation state from a bot.
pub fn collect_observation(bot: &Client, stage: u8) -> Value {
    json!({
        "player": player_state(bot),
        "world": world_state(bot),
        "inventory": inventory_state(bot),
        "nearby_blocks": nearby_blocks(bot),
        "nearby_entities": nearby_entities(bot),
        "spawn_point": spawn_point(bot),
        "progress": progress_state(bot, stage),
        "derived": derived_features(bot),
        "spatial_grid": spatial_grid(bot),
    })
}

fn player_state(bot: &Client) -> Value {
    let pos = bot.position();
    let health = bot.health();
    let hunger = bot.hunger();

    json!({
        "health": health,
        "food": hunger.food,
        "saturation": hunger.saturation,
        "armor": 0, // computed from inventory
        "xp_level": 0,  // Experience not tracked in azalea 0.15.1
        "is_on_fire": false,
        "is_in_water": false,
        "position": {
            "x": pos.x,
            "y": pos.y,
            "z": pos.z,
        }
    })
}

fn world_state(bot: &Client) -> Value {
    let world_name = bot.component::<azalea::world::InstanceName>().to_string();
    let dimension = if world_name.contains("nether") {
        "the_nether"
    } else if world_name.contains("end") {
        "the_end"
    } else {
        "overworld"
    };

    json!({
        "time_of_day": 6000, // would need packet tracking for precise time
        "is_daytime": true,
        "dimension": dimension,
        "light_level": 15,
        "biome_temperature": 0.5,
        "is_raining": false,
    })
}

fn inventory_state(bot: &Client) -> Value {
    let menu = bot.menu();
    let slots = menu.slots();

    let mut tools = json!({
        "has_wooden_pickaxe": false, "has_stone_pickaxe": false,
        "has_iron_pickaxe": false, "has_diamond_pickaxe": false,
        "has_wooden_sword": false, "has_stone_sword": false,
        "has_iron_sword": false, "has_diamond_sword": false,
        "has_axe": false, "has_shovel": false,
    });

    let mut resources: std::collections::HashMap<&str, u32> = [
        ("log", 0), ("planks", 0), ("stick", 0), ("cobblestone", 0),
        ("iron_ore", 0), ("iron_ingot", 0), ("gold_ingot", 0), ("diamond", 0),
        ("coal", 0), ("food", 0), ("wool", 0), ("obsidian", 0),
        ("ender_pearl", 0), ("blaze_rod", 0), ("blaze_powder", 0),
        ("eye_of_ender", 0), ("flint", 0), ("gravel", 0), ("bucket", 0),
        ("string", 0),
    ]
    .into_iter()
    .collect();

    let mut empty_slots = 0u32;
    let mut has_crafting_table = false;
    let mut has_furnace = false;
    let mut has_bed = false;

    for slot in slots.iter() {
        match slot {
            azalea::inventory::ItemStack::Present(item) => {
                let name = format!("{:?}", item.kind).to_lowercase();
                let count = item.count as u32;

                // Tools
                if name.contains("wooden_pickaxe") { tools["has_wooden_pickaxe"] = json!(true); }
                if name.contains("stone_pickaxe") { tools["has_stone_pickaxe"] = json!(true); }
                if name.contains("iron_pickaxe") { tools["has_iron_pickaxe"] = json!(true); }
                if name.contains("diamond_pickaxe") { tools["has_diamond_pickaxe"] = json!(true); }
                if name.contains("wooden_sword") { tools["has_wooden_sword"] = json!(true); }
                if name.contains("stone_sword") { tools["has_stone_sword"] = json!(true); }
                if name.contains("iron_sword") { tools["has_iron_sword"] = json!(true); }
                if name.contains("diamond_sword") { tools["has_diamond_sword"] = json!(true); }
                if name.contains("_axe") { tools["has_axe"] = json!(true); }
                if name.contains("_shovel") { tools["has_shovel"] = json!(true); }

                // Resources
                if name.contains("log") || name.contains("wood") {
                    *resources.get_mut("log").unwrap() += count;
                } else if name.contains("plank") {
                    *resources.get_mut("planks").unwrap() += count;
                } else if name == "stick" {
                    *resources.get_mut("stick").unwrap() += count;
                } else if name == "cobblestone" {
                    *resources.get_mut("cobblestone").unwrap() += count;
                } else if name.contains("raw_iron") {
                    *resources.get_mut("iron_ore").unwrap() += count;
                } else if name == "iron_ingot" {
                    *resources.get_mut("iron_ingot").unwrap() += count;
                } else if name == "gold_ingot" {
                    *resources.get_mut("gold_ingot").unwrap() += count;
                } else if name == "diamond" {
                    *resources.get_mut("diamond").unwrap() += count;
                } else if name == "coal" || name == "charcoal" {
                    *resources.get_mut("coal").unwrap() += count;
                } else if name.contains("obsidian") {
                    *resources.get_mut("obsidian").unwrap() += count;
                } else if name == "ender_pearl" {
                    *resources.get_mut("ender_pearl").unwrap() += count;
                } else if name == "blaze_rod" {
                    *resources.get_mut("blaze_rod").unwrap() += count;
                } else if name == "blaze_powder" {
                    *resources.get_mut("blaze_powder").unwrap() += count;
                } else if name.contains("eye") && name.contains("ender") {
                    *resources.get_mut("eye_of_ender").unwrap() += count;
                } else if name == "flint" {
                    *resources.get_mut("flint").unwrap() += count;
                } else if name == "gravel" {
                    *resources.get_mut("gravel").unwrap() += count;
                } else if name.contains("bucket") {
                    *resources.get_mut("bucket").unwrap() += count;
                } else if name == "string" {
                    *resources.get_mut("string").unwrap() += count;
                }

                // Food items
                if is_food(&name) {
                    *resources.get_mut("food").unwrap() += count;
                }

                // Special items
                if name == "crafting_table" { has_crafting_table = true; }
                if name == "furnace" { has_furnace = true; }
                if name.contains("bed") { has_bed = true; }
            }
            azalea::inventory::ItemStack::Empty => {
                empty_slots += 1;
            }
        }
    }

    let resources_json: serde_json::Map<String, Value> = resources
        .iter()
        .map(|(k, v)| (k.to_string(), json!(*v)))
        .collect();

    json!({
        "tools": tools,
        "resources": resources_json,
        "armor": {
            "helmet": "none", "chestplate": "none",
            "leggings": "none", "boots": "none",
            "has_shield": false, "total_armor_points": 0,
        },
        "empty_slots": empty_slots,
        "has_crafting_table": has_crafting_table,
        "has_furnace": has_furnace,
        "has_bed": has_bed,
    })
}

fn nearby_blocks(bot: &Client) -> Value {
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    let world = bot.world();
    let world_lock = world.read();

    let mut densities: std::collections::HashMap<&str, f64> = [
        ("log", 0.0), ("stone", 0.0), ("iron_ore", 0.0), ("diamond_ore", 0.0),
        ("coal_ore", 0.0), ("gold_ore", 0.0), ("water", 0.0), ("lava", 0.0),
        ("sand", 0.0), ("gravel", 0.0), ("dirt", 0.0), ("grass_block", 0.0),
        ("nether_portal", 0.0), ("obsidian", 0.0), ("nether_bricks", 0.0),
        ("soul_sand", 0.0), ("end_stone", 0.0), ("bedrock", 0.0),
        ("wheat", 0.0), ("chest", 0.0),
    ]
    .into_iter()
    .collect();

    let mut total_blocks = 0u32;
    let mut nearest_tree: f64 = 32.0;
    let mut nearest_water: f64 = 32.0;
    let mut nearest_lava: f64 = 32.0;
    let radius = 16i32;
    let step = 2i32;

    for dx in (-radius..=radius).step_by(step as usize) {
        for dy in (-radius..=radius).step_by(step as usize) {
            for dz in (-radius..=radius).step_by(step as usize) {
                let dist = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                if dist > radius as f64 {
                    continue;
                }

                let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                if let Some(block_state) = world_lock.get_block_state(bpos) {
                    total_blocks += 1;
                    let name = format!("{block_state:?}").to_lowercase();

                    for key in densities.keys().copied().collect::<Vec<_>>() {
                        if name.contains(key) {
                            *densities.get_mut(key).unwrap() += 1.0;
                            break;
                        }
                    }

                    if name.contains("log") && dist < nearest_tree {
                        nearest_tree = dist;
                    }
                    if name.contains("water") && dist < nearest_water {
                        nearest_water = dist;
                    }
                    if name.contains("lava") && dist < nearest_lava {
                        nearest_lava = dist;
                    }
                }
            }
        }
    }

    if total_blocks > 0 {
        for v in densities.values_mut() {
            *v = (*v / total_blocks as f64).min(1.0);
        }
    }

    let density_json: serde_json::Map<String, Value> = densities
        .iter()
        .map(|(k, v)| (k.to_string(), json!(*v)))
        .collect();

    json!({
        "densities": density_json,
        "structure": {
            "nearest_tree_dist": nearest_tree,
            "nearest_cave_dist": 32.0,
            "nearest_water_dist": nearest_water,
            "nearest_lava_dist": nearest_lava,
            "blocks_below_to_void": 64,
            "open_sky_above": true,
            "is_underground": center.y < 50,
            "nearest_village_dist": 200,
            "nearest_fortress_dist": 100,
            "stronghold_located": false,
            "directional_blocks": [0.5, 0.5, 0.5, 0.5, 0.5],
            "directional_danger": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    })
}

fn nearby_entities(_bot: &Client) -> Value {
    // Query nearby entities using Azalea's ECS
    // Simplified: return defaults; full implementation would use
    // bot.nearest_entity_by with component filters
    json!({
        "hostile": {
            "nearest_dist": 32, "nearest_health": 0,
            "count_within_8": 0, "count_within_16": 0,
            "zombie_nearby": false, "skeleton_nearby": false,
            "creeper_nearby": false, "enderman_nearby": false,
            "blaze_nearby": false, "ghast_nearby": false,
        },
        "passive": {
            "nearest_animal_dist": 32, "animal_count_within_16": 0,
            "cow_nearby": false, "pig_nearby": false,
            "sheep_nearby": false, "chicken_nearby": false,
            "horse_nearby": false, "rabbit_nearby": false,
            "nearest_villager_dist": 32, "villager_count": 0,
        },
        "special": {
            "ender_dragon_alive": false, "dragon_health": 0,
            "dragon_distance": 100, "end_crystal_count": 0,
            "nearest_crystal_dist": 100, "nearest_item_drop_dist": 16,
            "item_drops_count": 0, "nearest_player_dist": 64,
            "nearest_blaze_spawner_dist": 32, "near_end_portal": false,
        }
    })
}

fn spawn_point(_bot: &Client) -> Value {
    json!({"x": 0, "y": 64, "z": 0})
}

fn progress_state(bot: &Client, _stage: u8) -> Value {
    let inv = inventory_state(bot);
    let resources = inv.get("inventory")
        .and_then(|i| i.get("resources"))
        .cloned()
        .unwrap_or(json!({}));

    let blaze_rods = resources.get("blaze_rod").and_then(|v| v.as_u64()).unwrap_or(0);
    let ender_pearls = resources.get("ender_pearl").and_then(|v| v.as_u64()).unwrap_or(0);

    let world_name = bot.component::<azalea::world::InstanceName>().to_string();

    json!({
        "stage_progress": 0,
        "milestones": {
            "entered_nether": world_name.contains("nether"),
            "found_fortress": false,
            "blaze_rods_enough": blaze_rods >= 7,
            "ender_pearls_enough": ender_pearls >= 12,
            "located_stronghold": false,
            "entered_end": world_name.contains("end"),
            "crystals_destroyed_frac": 0,
            "nether_portal_built": false,
            "end_portal_activated": false,
            "dragon_defeated": false,
        }
    })
}

fn derived_features(_bot: &Client) -> Value {
    json!({
        "can_craft_anything": false,
        "can_smelt_anything": false,
        "time_since_progress": 0,
        "valid_actions_count": 0,
        "skill_recency": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
}

fn spatial_grid(bot: &Client) -> Value {
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    let world = bot.world();
    let world_lock = world.read();

    let mut grid = Vec::with_capacity(125);

    for dy in -2..=2 {
        for dx in -2..=2 {
            for dz in -2..=2 {
                let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                let category = if let Some(state) = world_lock.get_block_state(bpos) {
                    let name = format!("{state:?}").to_lowercase();
                    if name.contains("air") {
                        0 // AIR
                    } else if name.contains("water") || name.contains("lava") {
                        2 // LIQUID
                    } else if name.contains("ore") {
                        3 // ORE
                    } else {
                        1 // SOLID
                    }
                } else {
                    0
                };
                grid.push(category);
            }
        }
    }

    json!(grid)
}

fn is_food(name: &str) -> bool {
    matches!(
        name,
        "bread" | "apple" | "golden_apple" | "cooked_beef" | "cooked_porkchop"
            | "cooked_chicken" | "cooked_mutton" | "cooked_rabbit" | "cooked_salmon"
            | "cooked_cod" | "baked_potato" | "pumpkin_pie" | "cookie" | "melon_slice"
            | "dried_kelp" | "sweet_berries" | "carrot" | "beetroot" | "beef"
            | "porkchop" | "chicken" | "mutton" | "rabbit" | "rotten_flesh"
    )
}
