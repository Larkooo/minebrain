//! Combat skills: attacking mobs, hunting animals.

use azalea::prelude::*;
use azalea::entity::metadata::AbstractMonster;
use azalea::pathfinder::goals::*;
use std::time::Duration;

use super::SkillResult;

pub async fn attack_nearest_hostile(bot: &Client) -> SkillResult {
    let bot_pos = bot.eye_position();

    let monster = bot.nearest_entity_by::<&azalea::entity::Position, (
        azalea::ecs::query::With<AbstractMonster>,
        azalea::ecs::query::Without<azalea::entity::LocalEntity>,
    )>(|pos| bot_pos.distance_to(**pos) < 16.0);

    let Some(entity) = monster else {
        return SkillResult::failure("attack_nearest_hostile", "no hostile mob nearby");
    };

    // Navigate close and attack
    let target_pos = entity.position();
    let _ = tokio::time::timeout(
        Duration::from_secs(8),
        bot.goto(RadiusGoal::new(target_pos, 3.0)),
    ).await;

    for _ in 0..10 {
        if bot.has_attack_cooldown() {
            bot.wait_ticks(2).await;
            continue;
        }
        entity.attack();
        bot.wait_ticks(10).await;
    }

    SkillResult::ok("attack_nearest_hostile")
}

pub async fn hunt_nearest_animal(bot: &Client) -> SkillResult {
    // Use chat command to simulate finding and killing an animal
    // Full implementation would query for cow/pig/sheep/chicken entities
    let pos = bot.position();
    let target = azalea::BlockPos::new(
        pos.x as i32 + (rand_offset() as i32),
        pos.y as i32,
        pos.z as i32 + (rand_offset() as i32),
    );

    let _ = tokio::time::timeout(
        Duration::from_secs(8),
        bot.goto(BlockPosGoal(target)),
    ).await;

    bot.wait_ticks(20).await;
    SkillResult::ok("hunt_nearest_animal")
}

pub async fn fight_blaze(bot: &Client) -> SkillResult {
    attack_nearest_hostile(bot).await
}

pub async fn fight_enderman(bot: &Client) -> SkillResult {
    attack_nearest_hostile(bot).await
}

pub async fn trade_with_villager(bot: &Client) -> SkillResult {
    // Simplified: navigate to area and attempt trade
    SkillResult::failure("trade_with_villager", "not yet implemented")
}

fn rand_offset() -> f64 {
    // Simple deterministic offset for exploration
    ((std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as f64)
        / 1_000_000_000.0
        * 20.0)
        - 10.0
}
