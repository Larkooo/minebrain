#!/bin/bash
# Start everything: MC server, bot server, and training
# Each runs in its own terminal pane (requires tmux)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check tmux
if ! command -v tmux &>/dev/null; then
    echo "tmux not found. Starting components manually..."
    echo ""
    echo "Open 3 terminals and run:"
    echo "  1) ./scripts/start-server.sh   # Minecraft server"
    echo "  2) ./scripts/start-bot.sh      # Rust bot server"
    echo "  3) ./scripts/train.sh          # Python training"
    echo ""
    echo "Or install tmux: brew install tmux"
    exit 1
fi

SESSION="minebrain"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with MC server
tmux new-session -d -s "$SESSION" -n "mc-server" "$SCRIPT_DIR/start-server.sh"

# Wait for server to start
echo "Waiting 15s for MC server to start..."
sleep 15

# Bot server pane
tmux new-window -t "$SESSION" -n "bot" "$SCRIPT_DIR/start-bot.sh"

# Wait for bot to build and connect
echo "Waiting 10s for bot server to build..."
sleep 10

# Training pane
tmux new-window -t "$SESSION" -n "train" "$SCRIPT_DIR/train.sh"

# Attach
echo "All components started! Attaching to tmux session..."
echo "  Switch panes: Ctrl-B then number (0=mc, 1=bot, 2=train)"
echo "  Detach: Ctrl-B then D"
echo ""
tmux attach -t "$SESSION"
