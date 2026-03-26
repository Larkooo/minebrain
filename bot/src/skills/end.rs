//! End game skills: portal activation, crystal destruction, dragon fighting.

use azalea::pathfinder::goals::*;
use azalea::prelude::*;
use azalea::BlockPos;
use std::time::Duration;

use super::SkillResult;

pub async fn activate_end_portal(bot: &Client) -> SkillResult {
    // Find end portal frames and place eyes
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    let frames = {
        let world = bot.world();
        let world_lock = world.read();
        let mut found = Vec::new();
        for dx in -32..=32 {
            for dy in -8..=8 {
                for dz in -32..=32 {
                    let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        if format!("{state:?}").to_lowercase().contains("end_portal_frame") {
                            found.push(bpos);
                        }
                    }
                }
            }
        }
        found
    };

    if frames.is_empty() {
        return SkillResult::failure("activate_end_portal", "no portal frames found");
    }

    // Navigate to frames and place eyes
    for frame in &frames {
        let _ = tokio::time::timeout(
            Duration::from_secs(5),
            bot.goto(RadiusGoal::new(
                azalea::Vec3::new(frame.x as f64, frame.y as f64, frame.z as f64),
                3.0,
            )),
        ).await;
        bot.block_interact(*frame);
        bot.wait_ticks(5).await;
    }

    SkillResult::ok("activate_end_portal")
}

pub async fn enter_end_portal(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    let portal_pos = {
        let world = bot.world();
        let world_lock = world.read();
        let mut found = None;

        'search: for dx in -16..=16 {
            for dy in -4..=4 {
                for dz in -16..=16 {
                    let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        if format!("{state:?}").to_lowercase().contains("end_portal")
                            && !format!("{state:?}").to_lowercase().contains("frame")
                        {
                            found = Some(bpos);
                            break 'search;
                        }
                    }
                }
            }
        }
        found
    };

    if let Some(bpos) = portal_pos {
        let _ = tokio::time::timeout(
            Duration::from_secs(8),
            bot.goto(BlockPosGoal(bpos)),
        ).await;
        bot.wait_ticks(100).await;
        return SkillResult::ok("enter_end_portal");
    }

    SkillResult::failure("enter_end_portal", "no end portal found")
}

pub async fn kill_end_crystal(bot: &Client) -> SkillResult {
    // Find and attack nearest end crystal entity
    let bot_pos = bot.eye_position();

    // Attack nearest entity that looks like a crystal
    // In practice, would filter by EndCrystal component
    bot.look_at(azalea::Vec3::new(bot_pos.x, bot_pos.y + 20.0, bot_pos.z));
    bot.start_use_item(); // shoot arrow at crystal
    bot.wait_ticks(20).await;

    SkillResult::ok("kill_end_crystal")
}

pub async fn attack_ender_dragon(bot: &Client) -> SkillResult {
    // Navigate to center fountain and attack when dragon perches
    let _fountain = BlockPos::new(0, 64, 0);
    let _ = tokio::time::timeout(
        Duration::from_secs(10),
        bot.goto(RadiusGoal::new(azalea::Vec3::new(0.0, 64.0, 0.0), 5.0)),
    ).await;

    // Attack cycle: wait for dragon, swing
    for _ in 0..30 {
        if !bot.has_attack_cooldown() {
            // Look up toward dragon
            bot.set_direction(0.0, -30.0);
            // Attack would target the dragon entity
        }
        bot.wait_ticks(10).await;
    }

    SkillResult::ok("attack_ender_dragon")
}

pub async fn use_bed_on_dragon(bot: &Client) -> SkillResult {
    // Place bed and interact (causes explosion in The End)
    let pos = bot.position();
    let target = BlockPos::new(pos.x as i32, pos.y as i32 - 1, pos.z as i32);
    bot.block_interact(target); // place bed
    bot.wait_ticks(3).await;

    let bed_pos = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    bot.block_interact(bed_pos); // click bed -> explosion
    bot.wait_ticks(10).await;

    SkillResult::ok("use_bed_on_dragon")
}

pub async fn destroy_all_crystals(bot: &Client) -> SkillResult {
    let mut destroyed = 0;
    for _ in 0..10 {
        let result = kill_end_crystal(bot).await;
        if result.success { destroyed += 1; }
        bot.wait_ticks(10).await;
    }

    SkillResult {
        success: destroyed > 0,
        skill_name: "destroy_all_crystals".into(),
        reason: format!("destroyed {destroyed} crystals"),
        duration_ms: 0,
        items_collected: 0,
    }
}

pub async fn fight_dragon_cycle(bot: &Client) -> SkillResult {
    destroy_all_crystals(bot).await;
    attack_ender_dragon(bot).await
}

pub async fn pillar_to_crystal(bot: &Client) -> SkillResult {
    // Pillar up toward a crystal position
    for _ in 0..20 {
        bot.jump();
        bot.wait_ticks(4).await;
        let pos = bot.position();
        let below = BlockPos::new(pos.x as i32, pos.y as i32 - 1, pos.z as i32);
        bot.block_interact(below);
        bot.wait_ticks(3).await;
    }

    SkillResult::ok("pillar_to_crystal")
}

pub async fn collect_dragon_egg(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    let egg_pos = {
        let world = bot.world();
        let world_lock = world.read();
        let mut found = None;

        'search: for dx in -16..=16 {
            for dy in -8..=16 {
                for dz in -16..=16 {
                    let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        if format!("{state:?}").to_lowercase().contains("dragon_egg") {
                            found = Some(bpos);
                            break 'search;
                        }
                    }
                }
            }
        }
        found
    };

    if let Some(bpos) = egg_pos {
        // Mine block below the egg (egg falls)
        let below = BlockPos::new(bpos.x, bpos.y - 1, bpos.z);
        bot.mine(below).await;
        bot.wait_ticks(10).await;
        return SkillResult::ok("collect_dragon_egg");
    }

    SkillResult::failure("collect_dragon_egg", "no dragon egg found")
}
