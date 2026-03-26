//! MineBrain Bot Server
//!
//! WebSocket server that manages Azalea Minecraft bots for RL training.
//! Python training code communicates via JSON messages over WebSocket.

mod observation;
mod skills;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use azalea::prelude::*;
use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use observation::collect_observation;
use skills::{execute_skill, get_action_mask, SkillResult, CORE_SKILL_COUNT};

/// Shared state for a single bot environment.
struct BotEnv {
    client: Client,
    stage: u8,
}

/// All bot environments keyed by env_id.
type Bots = Arc<RwLock<HashMap<u32, BotEnv>>>;

// ── WebSocket Protocol ──

#[derive(Deserialize)]
struct WsRequest {
    #[serde(rename = "type")]
    msg_type: String,
    env_id: Option<u32>,
    action: Option<u32>,
    stage: Option<u8>,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct ResetResult {
    #[serde(rename = "type")]
    msg_type: String,
    env_id: u32,
    raw_state: serde_json::Value,
    action_mask: Vec<bool>,
}

#[derive(Serialize)]
struct StepResult {
    #[serde(rename = "type")]
    msg_type: String,
    env_id: u32,
    raw_state: serde_json::Value,
    action_mask: Vec<bool>,
    skill_result: SkillResult,
}

#[derive(Serialize)]
struct SkillsResult {
    #[serde(rename = "type")]
    msg_type: String,
    total_actions: usize,
    core_skills: usize,
}

#[derive(Serialize)]
struct ErrorResult {
    #[serde(rename = "type")]
    msg_type: String,
    error: String,
    env_id: u32,
    raw_state: serde_json::Value,
    action_mask: Vec<bool>,
    skill_result: SkillResult,
}

// ── Bot State for Azalea Handler ──

#[derive(Clone, Component, Default)]
struct BotState {
    env_id: u32,
}

/// Azalea event handler for each bot.
async fn handle_bot_event(
    bot: Client,
    event: Event,
    _state: BotState,
) -> eyre::Result<()> {
    match event {
        Event::Spawn => {
            info!("Bot spawned: {}", bot.username());
        }
        Event::Death(_) => {
            info!("Bot died: {}", bot.username());
        }
        _ => {}
    }
    Ok(())
}

// ── Server ──

const DEFAULT_PORT: u16 = 8765;
const MC_HOST: &str = "localhost";
const MC_PORT: u16 = 25565;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let mc_host = std::env::var("MC_HOST").unwrap_or_else(|_| MC_HOST.to_string());
    let mc_port: u16 = std::env::var("MC_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(MC_PORT);

    let bots: Bots = Arc::new(RwLock::new(HashMap::new()));

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(&addr).await?;

    info!("MineBrain bot server listening on ws://0.0.0.0:{port}");
    info!("Connecting bots to MC server at {mc_host}:{mc_port}");

    while let Ok((stream, peer)) = listener.accept().await {
        info!("Python client connected from {peer}");
        let bots = bots.clone();
        let mc_host = mc_host.clone();
        tokio::spawn(handle_connection(stream, bots, mc_host, mc_port));
    }

    Ok(())
}

async fn handle_connection(
    stream: TcpStream,
    bots: Bots,
    mc_host: String,
    mc_port: u16,
) {
    let ws = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("WebSocket handshake failed: {e}");
            return;
        }
    };

    let (mut tx, mut rx) = ws.split();

    while let Some(msg) = rx.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Err(e) => {
                warn!("WebSocket error: {e}");
                break;
            }
            _ => continue,
        };

        let req: WsRequest = match serde_json::from_str(&msg) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx
                    .send(Message::Text(
                        serde_json::json!({"error": format!("Invalid JSON: {e}")}).to_string(),
                    ))
                    .await;
                continue;
            }
        };

        let env_id = req.env_id.unwrap_or(0);
        let response = match req.msg_type.as_str() {
            "reset" => {
                handle_reset(
                    &bots,
                    env_id,
                    req.stage.unwrap_or(0),
                    req.seed,
                    &mc_host,
                    mc_port,
                )
                .await
            }
            "step" => {
                handle_step(&bots, env_id, req.action.unwrap_or(0)).await
            }
            "get_state" => handle_get_state(&bots, env_id).await,
            "get_skills" => {
                serde_json::to_string(&SkillsResult {
                    msg_type: "skills_result".into(),
                    total_actions: CORE_SKILL_COUNT,
                    core_skills: CORE_SKILL_COUNT,
                })
                .unwrap()
            }
            other => {
                serde_json::json!({"error": format!("Unknown type: {other}")}).to_string()
            }
        };

        if tx.send(Message::Text(response)).await.is_err() {
            break;
        }
    }

    info!("Python client disconnected");
}

async fn handle_reset(
    bots: &Bots,
    env_id: u32,
    stage: u8,
    _seed: Option<u64>,
    mc_host: &str,
    mc_port: u16,
) -> String {
    // Create bot if it doesn't exist, or reset existing one
    let needs_create = {
        let lock = bots.read();
        !lock.contains_key(&env_id)
    };

    if needs_create {
        let username = format!("minebrain_{env_id}");
        let account = Account::offline(&username);
        let addr = format!("{mc_host}:{}", mc_port + env_id as u16);

        match ClientBuilder::new()
            .set_handler(handle_bot_event)
            .start(account, &addr)
            .await
        {
            AppExit::Success => {}
            AppExit::Error(e) => {
                return serde_json::to_string(&ErrorResult {
                    msg_type: "reset_result".into(),
                    error: format!("Failed to connect bot: {e:?}"),
                    env_id,
                    raw_state: serde_json::json!({}),
                    action_mask: vec![false; CORE_SKILL_COUNT],
                    skill_result: SkillResult::failure("connect", "bot creation failed"),
                })
                .unwrap();
            }
        }
    }

    // Reset via chat commands if bot exists
    let lock = bots.read();
    if let Some(env) = lock.get(&env_id) {
        let bot = &env.client;
        bot.chat("/tp @s 0 64 0");
        bot.chat("/clear @s");
        bot.chat("/effect clear @s");
        bot.chat("/gamemode survival @s");
        bot.chat("/time set day");
        bot.chat("/weather clear");
        bot.chat("/kill @e[type=!player,distance=..100]");
        bot.chat("/effect give @s minecraft:instant_health 1 10");
        bot.chat("/effect give @s minecraft:saturation 1 10");

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let raw_state = collect_observation(bot, stage);
        let action_mask = get_action_mask(bot, stage);

        return serde_json::to_string(&ResetResult {
            msg_type: "reset_result".into(),
            env_id,
            raw_state,
            action_mask,
        })
        .unwrap();
    }
    drop(lock);

    // Fallback: return empty state
    serde_json::to_string(&ResetResult {
        msg_type: "reset_result".into(),
        env_id,
        raw_state: serde_json::json!({}),
        action_mask: vec![false; CORE_SKILL_COUNT],
    })
    .unwrap()
}

async fn handle_step(bots: &Bots, env_id: u32, action: u32) -> String {
    let bot_client = {
        let lock = bots.read();
        lock.get(&env_id).map(|env| (env.client.clone(), env.stage))
    };

    let Some((bot, stage)) = bot_client else {
        return serde_json::to_string(&StepResult {
            msg_type: "step_result".into(),
            env_id,
            raw_state: serde_json::json!({}),
            action_mask: vec![false; CORE_SKILL_COUNT],
            skill_result: SkillResult::failure("unknown", "bot not found"),
        })
        .unwrap();
    };

    let skill_result = execute_skill(&bot, action, stage).await;
    let raw_state = collect_observation(&bot, stage);
    let action_mask = get_action_mask(&bot, stage);

    serde_json::to_string(&StepResult {
        msg_type: "step_result".into(),
        env_id,
        raw_state,
        action_mask,
        skill_result,
    })
    .unwrap()
}

async fn handle_get_state(bots: &Bots, env_id: u32) -> String {
    let lock = bots.read();
    let raw_state = if let Some(env) = lock.get(&env_id) {
        collect_observation(&env.client, env.stage)
    } else {
        serde_json::json!({})
    };
    serde_json::json!({"type": "state_result", "env_id": env_id, "raw_state": raw_state})
        .to_string()
}
