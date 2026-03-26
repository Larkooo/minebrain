# MineBrain

A neural network that learns to play Minecraft from scratch and beat the Ender Dragon.

MineBrain uses **reinforcement learning** (PPO) to train an agent through an 8-stage curriculum — from punching trees to slaying the dragon. The agent doesn't see pixels; it reads structured game state and selects high-level skills ("mine nearest iron ore", "craft diamond pickaxe", "fight blaze") which the bot executes autonomously.

```
┌─────────────────────────┐     WebSocket      ┌──────────────────────────┐
│     Python (MLX)        │◄──── JSON ────────►│     Rust (Azalea)        │
│                         │                     │                          │
│  Actor-Critic Network   │    observations     │  Minecraft Bot Client    │
│  PPO Training Loop      │◄───────────────────│  72 Skill Executors      │
│  Curriculum Manager     │    action masks     │  Game State Observer     │
│  Reward Shaping         │───────────────────►│  Pathfinding / Mining    │
│                         │    skill actions    │                          │
└─────────────────────────┘                     └──────────┬───────────────┘
                                                           │ MC Protocol
                                                           ▼
                                                ┌──────────────────────┐
                                                │  Minecraft Server    │
                                                │  (Paper/Fabric)      │
                                                └──────────────────────┘
```

## How It Works

**The agent sees** a 310-feature observation vector: player health/hunger/position, inventory contents, nearby block densities, hostile mob distances, curriculum progress milestones — all normalized to [0,1].

**The agent chooses** from 72 discrete macro-action skills. Invalid skills are masked out (can't mine diamond without an iron pickaxe). The action space grows as the curriculum progresses.

**The agent learns** via Proximal Policy Optimization (PPO) with reward shaping that anneals over training — dense guidance early, sparse rewards later. Same approach that beat a [number-placement strategy game](https://github.com/Larkooo/nums-ai) by 14%.

## Curriculum

| Stage | Name | Goal | Skills Unlocked |
|-------|------|------|----------------|
| 0 | Punch Trees | Craft wooden pickaxe | 10 |
| 1 | Stone Age | Stone pickaxe + furnace | 20 |
| 2 | Iron Age | Iron gear + survive | 36 |
| 3 | Diamond Hunting | Diamond pickaxe | 47 |
| 4 | Nether Prep | Build nether portal | 54 |
| 5 | Nether Conquest | 7 blaze rods + 12 pearls | 61 |
| 6 | Stronghold | Enter The End | 65 |
| 7 | Dragon Slayer | Defeat the Ender Dragon | 72 |

Each stage has its own reward function, promotion criteria, and reward shaping that decays from 0.5 to 0.05 over 70% of stage training (same annealing pattern as nums-ai).

## Architecture

### Neural Network

```
Input (930 = 310 features × 3 frame stack)
  → Linear(930, 512) + ReLU
  → Linear(512, 512) + ReLU
  → Linear(512, 512) + ReLU

Actor: → Linear(512, 256) + ReLU → Linear(256, 72)  + action masking
Critic: → Linear(512, 256) + ReLU → Linear(256, 1)
```

- **1.28M parameters** on Apple MLX (Apple Silicon optimized)
- Action masking prevents invalid skill selection
- Frame stacking provides short-term memory

### Bot (Rust + Azalea)

The Minecraft client is written in **Rust** using [Azalea](https://github.com/azalea-rs/azalea), a high-performance Minecraft bot library built on Bevy ECS. This gives us:

- Native-speed pathfinding and block operations
- Type-safe game state access via ECS components
- Async skill execution with proper timeout handling
- Memory-efficient compared to JVM-based alternatives

### Skill System

Skills are macro-actions that abstract away low-level control:

| Category | Examples | Count |
|----------|----------|-------|
| Gathering | mine_nearest_log, mine_diamond_ore, collect_item_drop | 12 |
| Crafting | craft_planks, craft_iron_pickaxe, craft_eye_of_ender | 16 |
| Smelting | smelt_iron_ore, smelt_food | 3 |
| Survival | eat_food, build_shelter, equip_best_gear | 7 |
| Combat | attack_hostile, fight_blaze, attack_ender_dragon | 8 |
| Navigation | explore, descend_to_mine, find_fortress, go_to_stronghold | 10 |
| End Game | activate_end_portal, destroy_crystals, fight_dragon_cycle | 6 |
| Utility | wait, look_around, place_torch, dig_down, pillar_up | 10 |

Each skill has preconditions (checked via action masking) and a curriculum stage gate.

## Quick Start

### Prerequisites

- **Rust** (1.75+) — `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python 3.10+** with MLX — `pip install mlx`
- **Minecraft Java Server** (1.20.4) — [Paper](https://papermc.io/) recommended

### Setup

```bash
git clone https://github.com/Larkooo/minebrain.git
cd minebrain

# Python dependencies
pip install -e .

# Build the Rust bot (first build takes ~2 min)
cd bot && cargo build --release && cd ..

# Start a Minecraft server on localhost:25565
# (with op permissions for the bot, or use a plugin for /craft /smelt commands)
```

### Train

```bash
# Terminal 1: Start the bot server
cd bot && cargo run --release

# Terminal 2: Train from Stage 0
python -m src.train

# Resume from a specific stage
python -m src.train --stage 3 --resume models/stage_2

# Evaluate
python -m src.evaluate --stage 0 --episodes 100
```

### Training Dashboard

```
MineBrain Stage 3: Diamond Hunting │ PPO on MLX │ envs=8
████████████████░░░░░░░░░░░░░░░░░░░  48.2%  21h elapsed, ETA 22h (rollout)
983,040/2,000,000 steps │ 12 sps │ 4,312 episodes
reward 8.42 │ goal 34% │ loss 0.0234 [pg -0.0082 vf 0.0156 ent 0.0312]
trend ▁▂▃▃▄▅▅▆▅▆▇▆▇█▇██
promotion: 34% / 60% needed (last 5 episodes)
```

## Project Structure

```
minebrain/
├── src/                        # Python — ML training
│   ├── model.py                # Actor-Critic MLP (MLX, 1.28M params)
│   ├── train.py                # PPO loop + live dashboard
│   ├── env.py                  # Gymnasium wrapper + AsyncVecEnv
│   ├── bridge.py               # WebSocket client to Rust bot
│   ├── curriculum.py           # 8 stages, rewards, promotion logic
│   ├── skills.py               # 72 skill definitions + stage masks
│   ├── observations.py         # 310-feature encoder + frame stacker
│   └── evaluate.py             # Per-stage evaluation harness
├── bot/                        # Rust — Minecraft bot (Azalea)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             # WebSocket server + bot management
│       ├── observation.rs      # ECS game state → JSON
│       └── skills/
│           ├── mod.rs          # Skill registry + action mask
│           ├── gathering.rs    # Mining and resource collection
│           ├── crafting.rs     # Crafting and smelting
│           ├── combat.rs       # Mob fighting
│           ├── survival.rs     # Eating, shelter, equipment
│           ├── navigation.rs   # Pathfinding, mining descent
│           ├── nether.rs       # Portal, fortress, blazes
│           └── end.rs          # Dragon fight
├── models/                     # Saved checkpoints per stage
├── pyproject.toml
└── CLAUDE.md
```

## PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-5 | Slower than nums-ai (noisier MC gradients) |
| Rollout steps | 2,048 | Shorter (MC skills take 1-15 seconds) |
| Epochs per update | 4 | |
| Batch size | 256 | |
| Gamma | 0.995 | |
| GAE lambda | 0.95 | |
| Clip epsilon | 0.15 | Conservative updates |
| Entropy coeff | 0.03 | |
| Parallel envs | 8 | Each on its own MC server port |
| Frame stack | 3 | 310 × 3 = 930 input features |

## Extending with Custom Skills

Add new skills by implementing them in `bot/src/skills/`:

1. Add your async skill function to the appropriate module (or create a new one)
2. Register it in `bot/src/skills/mod.rs` (add to `SKILL_NAMES`, `SKILL_MIN_STAGE`, and `dispatch_skill`)
3. Add the corresponding `SkillDef` in `src/skills.py`
4. Update `NUM_SKILLS` in both Python and Rust
5. Add action mask preconditions

The model automatically adjusts to the new action space size on next training run.

## Acknowledgments

- Training approach adapted from [nums-ai](https://github.com/Larkooo/nums-ai) — PPO on MLX for game-playing
- Minecraft bot powered by [Azalea](https://github.com/azalea-rs/azalea) — Rust Minecraft client library
- Neural network framework: [Apple MLX](https://github.com/ml-explore/mlx)

## License

MIT
