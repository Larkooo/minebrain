#!/bin/bash
# Build and start the Rust bot server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOT_DIR="$SCRIPT_DIR/../bot"

cd "$BOT_DIR"

echo "Building MineBrain bot server..."
cargo build --release 2>&1 | tail -5

echo ""
echo "Starting bot server on ws://localhost:8765"
echo "  MC server: localhost:25565"
echo ""

exec cargo run --release
