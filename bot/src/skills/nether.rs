//! Nether skills: portal building, nether navigation, fortress finding.

use azalea::pathfinder::goals::*;
use azalea::prelude::*;
use azalea::BlockPos;
use std::time::Duration;

use super::SkillResult;

pub async fn build_nether_portal(bot: &Client) -> SkillResult {
    // Place obsidian in a portal frame pattern
    let pos = bot.position();
    let base = BlockPos::new(pos.x as i32 + 2, pos.y as i32, pos.z as i32);

    // Portal frame offsets (4x5, needs 10 obsidian minimum)
    let frame: [(i32, i32); 10] = [
        // Bottom
        (0, 0), (1, 0), (2, 0), (3, 0),
        // Left pillar
        (0, 1), (0, 2), (0, 3),
        // Right pillar
        (3, 1), (3, 2), (3, 3),
    ];

    for (dx, dy) in frame {
        let target = BlockPos::new(base.x + dx, base.y + dy, base.z);
        bot.block_interact(target);
        bot.wait_ticks(3).await;
    }

    // Light portal with flint and steel
    let inside = BlockPos::new(base.x + 1, base.y + 1, base.z);
    bot.block_interact(inside);
    bot.wait_ticks(10).await;

    SkillResult::ok("build_nether_portal")
}

pub async fn enter_nether_portal(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    let portal_pos = {
        let world = bot.world();
        let world_lock = world.read();
        let mut found = None;

        'search: for dx in -32..=32 {
            for dy in -8..=8 {
                for dz in -32..=32 {
                    let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        if format!("{state:?}").to_lowercase().contains("nether_portal") {
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
            Duration::from_secs(10),
            bot.goto(BlockPosGoal(bpos)),
        ).await;
        bot.wait_ticks(100).await; // Wait for teleport
        return SkillResult::ok("enter_nether_portal");
    }

    SkillResult::failure("enter_nether_portal", "no portal found")
}

pub async fn find_nether_fortress(bot: &Client) -> SkillResult {
    // Search for nether bricks (fortress indicator)
    let pos = bot.position();

    let fortress_pos = {
        let world = bot.world();
        let world_lock = world.read();
        let mut found = None;

        'search: for dx in -64..=64 {
            for dy in -16..=16 {
                for dz in -64..=64 {
                    if (dx * dx + dy * dy + dz * dz) > 64 * 64 { continue; }
                    let bpos = BlockPos::new(
                        pos.x as i32 + dx,
                        pos.y as i32 + dy,
                        pos.z as i32 + dz,
                    );
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        if format!("{state:?}").to_lowercase().contains("nether_brick") {
                            found = Some(bpos);
                            break 'search;
                        }
                    }
                }
            }
        }
        found
    };

    if let Some(bpos) = fortress_pos {
        let _ = tokio::time::timeout(
            Duration::from_secs(30),
            bot.goto(BlockPosGoal(bpos)),
        ).await;
        return SkillResult::ok("find_nether_fortress");
    }

    // Not found, explore in +Z direction (fortresses generate along Z axis)
    let target = BlockPos::new(
        pos.x as i32,
        pos.y as i32,
        pos.z as i32 + 50,
    );
    let _ = tokio::time::timeout(
        Duration::from_secs(15),
        bot.goto(BlockPosGoal(target)),
    ).await;

    SkillResult::failure("find_nether_fortress", "still searching")
}
