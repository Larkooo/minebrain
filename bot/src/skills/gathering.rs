//! Gathering skills: mining and collecting resources.

use azalea::pathfinder::goals::*;
use azalea::prelude::*;
use azalea::BlockPos;

use super::SkillResult;

pub async fn mine_nearest_log(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "log", "mine_nearest_log").await
}

pub async fn mine_nearest_stone(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "stone", "mine_nearest_stone").await
}

pub async fn mine_nearest_coal_ore(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "coal_ore", "mine_nearest_coal_ore").await
}

pub async fn mine_nearest_iron_ore(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "iron_ore", "mine_nearest_iron_ore").await
}

pub async fn mine_nearest_gold_ore(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "gold_ore", "mine_nearest_gold_ore").await
}

pub async fn mine_nearest_diamond_ore(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "diamond_ore", "mine_nearest_diamond_ore").await
}

pub async fn mine_nearest_obsidian(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "obsidian", "mine_nearest_obsidian").await
}

pub async fn collect_gravel(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "gravel", "collect_gravel").await
}

pub async fn collect_sand(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "sand", "collect_sand").await
}

pub async fn mine_nether_quartz(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "nether_quartz_ore", "mine_nether_quartz").await
}

pub async fn harvest_crop(bot: &Client) -> SkillResult {
    mine_nearest_block(bot, "wheat", "harvest_crop").await
}

pub async fn collect_nearest_item_drop(bot: &Client) -> SkillResult {
    // Walk toward nearby area to pick up items
    let pos = bot.position();
    let target = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    bot.goto(RadiusGoal::new(azalea::Vec3::new(target.x as f64, target.y as f64, target.z as f64), 5.0)).await;
    bot.wait_ticks(20).await;
    SkillResult::ok("collect_nearest_item_drop")
}

/// Find the nearest block matching `name` and mine it.
async fn mine_nearest_block(bot: &Client, block_name: &str, skill_name: &str) -> SkillResult {
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    let world = bot.world();

    // Search for the block in expanding radius
    let mut best_pos: Option<BlockPos> = None;
    let mut best_dist = f64::MAX;

    {
        let world_lock = world.read();
        for dx in -32..=32 {
            for dy in -16..=16 {
                for dz in -32..=32 {
                    let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                    if let Some(state) = world_lock.get_block_state(bpos) {
                        let name = format!("{state:?}").to_lowercase();
                        if name.contains(block_name) {
                            let dist = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                            if dist < best_dist {
                                best_dist = dist;
                                best_pos = Some(bpos);
                            }
                        }
                    }
                }
            }
        }
    }

    let Some(target) = best_pos else {
        return SkillResult::failure(skill_name, &format!("no {block_name} nearby"));
    };

    // Pathfind to the block
    if let Err(_) = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        bot.goto(BlockPosGoal(target)),
    ).await {
        return SkillResult::failure(skill_name, "pathfind timeout");
    }

    // Mine the block
    bot.mine(target).await;
    bot.wait_ticks(5).await;

    SkillResult::ok(skill_name)
}
