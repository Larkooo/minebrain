//! MineBrain Bot Server
//!
//! WebSocket server that manages Azalea Minecraft bots for RL training.
//! Python training code communicates via JSON messages over WebSocket.
//!
//! Architecture: Azalea requires `spawn_local` (Bevy ECS), so the main
//! tokio runtime uses `current_thread` flavor with a `LocalSet`. The WS
//! server runs on a separate OS thread with its own multi-thread runtime.
//! Communication between the two uses `tokio::sync::mpsc` channels.

mod observation;
mod skills;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use azalea::prelude::*;
use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
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

#[derive(Deserialize, Debug)]
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
struct ErrorResult {
    #[serde(rename = "type")]
    msg_type: String,
    error: String,
    env_id: u32,
    raw_state: serde_json::Value,
    action_mask: Vec<bool>,
    skill_result: SkillResult,
}

// ── Command types sent from WS thread to Azalea thread ──

enum BotCommand {
    Reset {
        env_id: u32,
        stage: u8,
        mc_host: String,
        mc_port: u16,
        reply: oneshot::Sender<String>,
    },
    Step {
        env_id: u32,
        action: u32,
        reply: oneshot::Sender<String>,
    },
    GetState {
        env_id: u32,
        reply: oneshot::Sender<String>,
    },
    GetSkills {
        reply: oneshot::Sender<String>,
    },
}

const DEFAULT_PORT: u16 = 8765;
const MC_HOST: &str = "localhost";
const MC_PORT: u16 = 25565;

fn main() {
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

    info!("MineBrain bot server starting");
    info!("  WS port:   {port}");
    info!("  MC server:  {mc_host}:{mc_port}");

    // Channel: WS thread sends commands, Azalea thread processes them
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<BotCommand>();

    // Start WebSocket server on a separate OS thread (multi-threaded runtime)
    let ws_mc_host = mc_host.clone();
    let ws_mc_port = mc_port;
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build WS runtime");

        rt.block_on(ws_server(port, cmd_tx, ws_mc_host, ws_mc_port));
    });

    // Run Azalea bot logic on main thread with LocalSet (required by Bevy ECS)
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build Azalea runtime");

    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, bot_command_loop(cmd_rx, mc_host, mc_port));
}

// ── Bot command processor (runs in LocalSet) ──

async fn bot_command_loop(
    mut cmd_rx: mpsc::UnboundedReceiver<BotCommand>,
    mc_host: String,
    mc_port: u16,
) {
    let bots: Bots = Arc::new(RwLock::new(HashMap::new()));

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            BotCommand::Reset {
                env_id,
                stage,
                mc_host: host,
                mc_port: port,
                reply,
            } => {
                let response = handle_reset(&bots, env_id, stage, &host, port).await;
                let _ = reply.send(response);
            }
            BotCommand::Step {
                env_id,
                action,
                reply,
            } => {
                let response = handle_step(&bots, env_id, action).await;
                let _ = reply.send(response);
            }
            BotCommand::GetState { env_id, reply } => {
                let response = handle_get_state(&bots, env_id).await;
                let _ = reply.send(response);
            }
            BotCommand::GetSkills { reply } => {
                let response = serde_json::json!({
                    "type": "skills_result",
                    "total_actions": CORE_SKILL_COUNT,
                    "core_skills": CORE_SKILL_COUNT,
                })
                .to_string();
                let _ = reply.send(response);
            }
        }
    }
}

// ── WebSocket server (runs on separate thread) ──

async fn ws_server(
    port: u16,
    cmd_tx: mpsc::UnboundedSender<BotCommand>,
    mc_host: String,
    mc_port: u16,
) {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind WS port");

    info!("WebSocket server listening on ws://0.0.0.0:{port}");

    while let Ok((stream, peer)) = listener.accept().await {
        info!("Python client connected from {peer}");
        let cmd_tx = cmd_tx.clone();
        let mc_host = mc_host.clone();
        tokio::spawn(handle_ws_connection(stream, cmd_tx, mc_host, mc_port));
    }
}

async fn handle_ws_connection(
    stream: tokio::net::TcpStream,
    cmd_tx: mpsc::UnboundedSender<BotCommand>,
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
        let text = match msg {
            Ok(tokio_tungstenite::tungstenite::Message::Text(t)) => t,
            Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => break,
            Err(e) => {
                warn!("WebSocket error: {e}");
                break;
            }
            _ => continue,
        };

        let req: WsRequest = match serde_json::from_str(&text) {
            Ok(r) => r,
            Err(e) => {
                let err = serde_json::json!({"error": format!("Invalid JSON: {e}")}).to_string();
                let _ = tx
                    .send(tokio_tungstenite::tungstenite::Message::Text(err))
                    .await;
                continue;
            }
        };

        let env_id = req.env_id.unwrap_or(0);
        let (reply_tx, reply_rx) = oneshot::channel();

        let cmd = match req.msg_type.as_str() {
            "reset" => BotCommand::Reset {
                env_id,
                stage: req.stage.unwrap_or(0),
                mc_host: mc_host.clone(),
                mc_port,
                reply: reply_tx,
            },
            "step" => BotCommand::Step {
                env_id,
                action: req.action.unwrap_or(0),
                reply: reply_tx,
            },
            "get_state" => BotCommand::GetState {
                env_id,
                reply: reply_tx,
            },
            "get_skills" => BotCommand::GetSkills { reply: reply_tx },
            other => {
                let err = serde_json::json!({"error": format!("Unknown type: {other}")}).to_string();
                let _ = tx
                    .send(tokio_tungstenite::tungstenite::Message::Text(err))
                    .await;
                continue;
            }
        };

        if cmd_tx.send(cmd).is_err() {
            break;
        }

        // Wait for reply from the Azalea thread
        match reply_rx.await {
            Ok(response) => {
                if tx
                    .send(tokio_tungstenite::tungstenite::Message::Text(response))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Err(_) => {
                warn!("Bot command handler dropped");
                break;
            }
        }
    }

    info!("Python client disconnected");
}

// ── Bot handlers (run in LocalSet context) ──

async fn handle_reset(
    bots: &Bots,
    env_id: u32,
    stage: u8,
    mc_host: &str,
    mc_port: u16,
) -> String {
    let needs_create = {
        let lock = bots.read();
        !lock.contains_key(&env_id)
    };

    if needs_create {
        let username = format!("minebrain_{env_id}");
        let account = Account::offline(&username);
        let addr = format!("{mc_host}:{mc_port}");

        info!("Creating bot {username} connecting to {addr}");

        match Client::join(account, addr.as_str()).await {
            Ok((client, _rx)) => {
                info!("Bot {username} connected successfully");
                // Wait for spawn
                client.wait_ticks(20).await;

                let mut lock = bots.write();
                lock.insert(env_id, BotEnv { client, stage });
            }
            Err(e) => {
                error!("Failed to connect bot: {e:?}");
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

    // Reset via chat commands
    let bot = {
        let lock = bots.read();
        lock.get(&env_id).map(|env| env.client.clone())
    };

    if let Some(bot) = bot {
        bot.chat("/tp @s 0 64 0");
        bot.chat("/clear @s");
        bot.chat("/effect clear @s");
        bot.chat("/gamemode survival @s");
        bot.chat("/time set day");
        bot.chat("/weather clear");
        bot.chat("/kill @e[type=!player,distance=..100]");
        bot.chat("/effect give @s minecraft:instant_health 1 10");
        bot.chat("/effect give @s minecraft:saturation 1 10");

        bot.wait_ticks(10).await;

        let raw_state = collect_observation(&bot, stage);
        let action_mask = get_action_mask(&bot, stage);

        return serde_json::to_string(&ResetResult {
            msg_type: "reset_result".into(),
            env_id,
            raw_state,
            action_mask,
        })
        .unwrap();
    }

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
