#!/bin/bash
# Start training with sensible defaults for single-machine setup
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

cd "$PROJECT_DIR"

# Default to 1 env for easier local testing
N_ENVS="${N_ENVS:-1}"
STAGE="${STAGE:-0}"

echo "Starting MineBrain training"
echo "  Stage: $STAGE"
echo "  Envs:  $N_ENVS"
echo "  Make sure the MC server and bot server are running first!"
echo ""

exec python3 -m src.train \
    --n-envs "$N_ENVS" \
    --stage "$STAGE" \
    "$@"
