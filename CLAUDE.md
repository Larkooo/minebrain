# MineBrain

RL agent that learns to play and beat Minecraft using PPO on Apple MLX.

## Architecture

- **Python** (src/): MLX neural network, PPO training, Gymnasium environment wrapper
- **Rust** (bot/): Azalea 0.26.1 bot server, skill execution, observation extraction
- **Bridge**: Python <-> Rust via WebSocket (JSON protocol on port 8765)

## Key patterns (from nums-ai)

- Actor-Critic MLP with action masking
- PPO with GAE advantage estimation
- Reward shaping with annealing per curriculum stage
- Vectorized async environments

## Running

```bash
# Install Python dependencies
pip install -e .

# Build and run the Rust bot server
cd bot && cargo run --release

# In another terminal, start training
python -m src.train

# Evaluate a trained model
python -m src.evaluate --stage 0
```

## Project structure

- `src/model.py` - MLX Actor-Critic network (1.2M params)
- `src/train.py` - PPO training loop + live dashboard
- `src/env.py` - MinecraftEnv Gymnasium wrapper + AsyncVecEnv
- `src/bridge.py` - Python WebSocket client
- `src/curriculum.py` - 8-stage curriculum with rewards and promotion
- `src/skills.py` - Skill registry (72 skills) with action masking
- `src/observations.py` - Observation feature vector (~310 features)
- `bot/src/main.rs` - Azalea WebSocket server + bot management
- `bot/src/observation.rs` - Game state extraction from Azalea ECS
- `bot/src/skills/` - Skill implementations (gathering, crafting, combat, etc.)
