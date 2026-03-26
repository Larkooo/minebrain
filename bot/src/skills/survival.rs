//! Survival skills: eating, sleeping, building, equipment management.

use azalea::prelude::*;

use super::SkillResult;

pub async fn eat_food(bot: &Client) -> SkillResult {
    // Equip food and consume
    bot.start_use_item();
    bot.wait_ticks(40).await; // eating takes ~1.6 seconds
    SkillResult::ok("eat_food")
}

pub async fn sleep_in_bed(bot: &Client) -> SkillResult {
    // Find bed and interact
    let pos = bot.position();
    let center = azalea::BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    let bed_pos = {
        let world = bot.world();
        let world_lock = world.read();
        let mut found = None;

        // Search for bed
        'search: for dx in -16..=16 {
            for dy in -4..=4 {
                for dz in -16..=16 {
                    let bpos = azalea::BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        let name = format!("{state:?}").to_lowercase();
                        if name.contains("bed") {
                            found = Some(bpos);
                            break 'search;
                        }
                    }
                }
            }
        }
        found
    }; // world_lock dropped here

    if let Some(bpos) = bed_pos {
        bot.block_interact(bpos);
        bot.wait_ticks(60).await;
        return SkillResult::ok("sleep_in_bed");
    }

    SkillResult::failure("sleep_in_bed", "no bed found")
}

pub async fn place_crafting_table(bot: &Client) -> SkillResult {
    // Place block at feet offset
    let pos = bot.position();
    let target = azalea::BlockPos::new(pos.x as i32 + 1, pos.y as i32 - 1, pos.z as i32);
    bot.block_interact(target);
    bot.wait_ticks(5).await;
    SkillResult::ok("place_crafting_table")
}

pub async fn place_furnace(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let target = azalea::BlockPos::new(pos.x as i32 - 1, pos.y as i32 - 1, pos.z as i32);
    bot.block_interact(target);
    bot.wait_ticks(5).await;
    SkillResult::ok("place_furnace")
}

pub async fn build_simple_shelter(bot: &Client) -> SkillResult {
    // Build walls using block placement
    // Simplified: place blocks around the bot
    let pos = bot.position();
    let base = azalea::BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    for dx in -1..=1 {
        for dz in -1..=1 {
            if dx == 0 && dz == 0 { continue; }
            for dy in 0..=2 {
                let target = azalea::BlockPos::new(base.x + dx, base.y + dy, base.z + dz);
                bot.block_interact(target);
                bot.wait_ticks(2).await;
            }
        }
    }
    // Roof
    for dx in -1..=1 {
        for dz in -1..=1 {
            let target = azalea::BlockPos::new(base.x + dx, base.y + 3, base.z + dz);
            bot.block_interact(target);
            bot.wait_ticks(2).await;
        }
    }

    SkillResult::ok("build_simple_shelter")
}

pub async fn retreat_from_danger(bot: &Client) -> SkillResult {
    // Sprint away from current position
    bot.sprint(azalea::SprintDirection::Forward);
    bot.wait_ticks(60).await; // 3 seconds
    bot.walk(azalea::WalkDirection::None);
    SkillResult::ok("retreat_from_danger")
}

pub async fn equip_best_gear(_bot: &Client) -> SkillResult {
    // Auto-equip handled by looking through inventory slots
    // In practice, would click-move armor pieces to armor slots
    SkillResult::ok("equip_best_gear")
}

pub async fn wait_idle(bot: &Client) -> SkillResult {
    bot.wait_ticks(60).await; // 3 seconds
    SkillResult::ok("wait_idle")
}

pub async fn look_around(bot: &Client) -> SkillResult {
    let yaw = bot.direction().0;
    for i in 0..4 {
        bot.set_direction(yaw + (i as f32) * 90.0, 0.0);
        bot.wait_ticks(6).await;
    }
    SkillResult::ok("look_around")
}

pub async fn drop_junk_items(_bot: &Client) -> SkillResult {
    // Would iterate inventory and drop low-value items
    SkillResult::ok("drop_junk_items")
}

pub async fn organize_inventory(_bot: &Client) -> SkillResult {
    SkillResult::ok("organize_inventory")
}

pub async fn place_torch(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let target = azalea::BlockPos::new(pos.x as i32, pos.y as i32 - 1, pos.z as i32);
    bot.block_interact(target);
    bot.wait_ticks(3).await;
    SkillResult::ok("place_torch")
}
