//! Navigation skills: exploring, mining descent/ascent, portal finding.

use azalea::pathfinder::goals::*;
use azalea::prelude::*;
use azalea::BlockPos;
use std::time::Duration;

use super::SkillResult;

pub async fn explore_randomly(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let angle = rand_angle();
    let distance = 20.0 + rand_frac() * 30.0;

    let target = BlockPos::new(
        (pos.x + angle.cos() * distance) as i32,
        pos.y as i32,
        (pos.z + angle.sin() * distance) as i32,
    );

    let _ = tokio::time::timeout(
        Duration::from_secs(10),
        bot.goto(BlockPosGoal(target)),
    ).await;

    SkillResult::ok("explore_randomly")
}

pub async fn explore_for_cave(bot: &Client) -> SkillResult {
    // Search for air below surface level
    let pos = bot.position();
    let world = bot.world();
    let world_lock = world.read();

    for radius in [10, 20, 30] {
        for i in 0..8 {
            let angle = (i as f64 / 8.0) * std::f64::consts::TAU;
            let x = pos.x as i32 + (angle.cos() * radius as f64) as i32;
            let z = pos.z as i32 + (angle.sin() * radius as f64) as i32;

            for y in ((pos.y as i32 - 30).max(5))..=(pos.y as i32 - 5) {
                let bpos = BlockPos::new(x, y, z);
                if let Some(state) = world_lock.get_block_state(bpos) {
                    let name = format!("{state:?}").to_lowercase();
                    if name.contains("air") || name.contains("cave") {
                        drop(world_lock);
                        let _ = tokio::time::timeout(
                            Duration::from_secs(12),
                            bot.goto(BlockPosGoal(bpos)),
                        ).await;
                        return SkillResult::ok("explore_for_cave");
                    }
                }
            }
        }
    }

    SkillResult::failure("explore_for_cave", "no cave found")
}

pub async fn go_to_nearest_village(bot: &Client) -> SkillResult {
    // Explore in a direction hoping to find a village
    explore_randomly(bot).await;
    SkillResult { skill_name: "go_to_nearest_village".into(), ..SkillResult::ok("") }
}

pub async fn descend_to_mining_level(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let target_y = 11;

    if (pos.y as i32) <= target_y + 3 {
        return SkillResult::ok("descend_to_mining_level");
    }

    // Dig staircase: mine block ahead and below repeatedly
    let mut current = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    let direction = bot.direction();
    let dx = -(direction.y_rot().to_radians().sin()) as i32;
    let dz = -(direction.y_rot().to_radians().cos()) as i32;

    for _ in 0..50 {
        if current.y <= target_y { break; }

        let next = BlockPos::new(current.x + dx.signum(), current.y - 1, current.z + dz.signum());
        let above = BlockPos::new(next.x, next.y + 1, next.z);

        bot.mine(next).await;
        bot.mine(above).await;

        let _ = tokio::time::timeout(
            Duration::from_secs(3),
            bot.goto(BlockPosGoal(next)),
        ).await;

        current = BlockPos::new(
            bot.position().x as i32,
            bot.position().y as i32,
            bot.position().z as i32,
        );
    }

    SkillResult::ok("descend_to_mining_level")
}

pub async fn ascend_to_surface(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let mut current = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);

    for _ in 0..80 {
        // Check if at surface (light above)
        let above2 = BlockPos::new(current.x, current.y + 2, current.z);
        let world = bot.world();
        let is_air = {
            let lock = world.read();
            lock.get_block_state(above2)
                .map(|s| format!("{s:?}").to_lowercase().contains("air"))
                .unwrap_or(true)
        };

        if is_air {
            // Mine up
            let above = BlockPos::new(current.x, current.y + 2, current.z);
            bot.mine(above).await;
            bot.jump();
            bot.wait_ticks(5).await;
        } else {
            bot.mine(BlockPos::new(current.x, current.y + 2, current.z)).await;
        }

        current = BlockPos::new(
            bot.position().x as i32,
            bot.position().y as i32,
            bot.position().z as i32,
        );

        // Check if reached surface
        if current.y > 60 { break; }
    }

    SkillResult::ok("ascend_to_surface")
}

pub async fn throw_eye_of_ender(bot: &Client) -> SkillResult {
    bot.set_direction(bot.direction().y_rot(), -15.0); // Look slightly up
    bot.start_use_item();
    bot.wait_ticks(40).await;
    SkillResult::ok("throw_eye_of_ender")
}

pub async fn go_to_stronghold(bot: &Client) -> SkillResult {
    // Throw eye and move in that direction
    throw_eye_of_ender(bot).await;

    let pos = bot.position();
    let dir = bot.direction();
    let yaw_rad = dir.y_rot().to_radians();
    let target = BlockPos::new(
        (pos.x - (yaw_rad.sin() as f64) * 100.0) as i32,
        pos.y as i32,
        (pos.z - (yaw_rad.cos() as f64) * 100.0) as i32,
    );

    let _ = tokio::time::timeout(
        Duration::from_secs(30),
        bot.goto(BlockPosGoal(target)),
    ).await;

    SkillResult::ok("go_to_stronghold")
}

pub async fn fill_bucket_with_water(bot: &Client) -> SkillResult {
    // Find water and right-click with bucket
    let pos = bot.position();
    let center = BlockPos::new(pos.x as i32, pos.y as i32, pos.z as i32);
    let world = bot.world();
    let world_lock = world.read();

    for dx in -16..=16 {
        for dy in -8..=8 {
            for dz in -16..=16 {
                let bpos = BlockPos::new(center.x + dx, center.y + dy, center.z + dz);
                if let Some(state) = world_lock.get_block_state(bpos) {
                    if format!("{state:?}").to_lowercase().contains("water") {
                        drop(world_lock);
                        let _ = tokio::time::timeout(
                            Duration::from_secs(8),
                            bot.goto(RadiusGoal::new(
                                azalea::Vec3::new(bpos.x as f64, bpos.y as f64, bpos.z as f64),
                                3.0,
                            )),
                        ).await;
                        bot.block_interact(bpos);
                        bot.wait_ticks(5).await;
                        return SkillResult::ok("fill_bucket_with_water");
                    }
                }
            }
        }
    }

    SkillResult::failure("fill_bucket_with_water", "no water nearby")
}

pub async fn fill_bucket_with_lava(bot: &Client) -> SkillResult {
    SkillResult::failure("fill_bucket_with_lava", "not yet implemented")
}

pub async fn create_infinite_water(bot: &Client) -> SkillResult {
    SkillResult::failure("create_infinite_water", "not yet implemented")
}

pub async fn dig_down_one(bot: &Client) -> SkillResult {
    let pos = bot.position();
    let below = BlockPos::new(pos.x as i32, pos.y as i32 - 1, pos.z as i32);
    bot.mine(below).await;
    SkillResult::ok("dig_down_one")
}

pub async fn pillar_up_one(bot: &Client) -> SkillResult {
    bot.jump();
    bot.wait_ticks(4).await;
    let pos = bot.position();
    let below = BlockPos::new(pos.x as i32, pos.y as i32 - 1, pos.z as i32);
    bot.block_interact(below);
    bot.wait_ticks(3).await;
    SkillResult::ok("pillar_up_one")
}

fn rand_angle() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as f64;
    (nanos / 1_000_000_000.0) * std::f64::consts::TAU
}

fn rand_frac() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as f64;
    (nanos % 1_000_000.0) / 1_000_000.0
}
