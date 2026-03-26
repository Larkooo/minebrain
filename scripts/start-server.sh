#!/bin/bash
# Start the Minecraft Paper server for training
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/../server"

cd "$SERVER_DIR"

echo "Starting Minecraft Paper server..."
echo "  Port: 25565"
echo "  Mode: offline (no auth)"
echo "  Seed: minebrain42"
echo ""

# Use Homebrew Java if system Java is missing
JAVA="java"
if ! command -v java &>/dev/null || java -version 2>&1 | grep -q "Unable to locate"; then
    if [ -f "/opt/homebrew/opt/openjdk@21/bin/java" ]; then
        JAVA="/opt/homebrew/opt/openjdk@21/bin/java"
    elif [ -f "/opt/homebrew/opt/openjdk/bin/java" ]; then
        JAVA="/opt/homebrew/opt/openjdk/bin/java"
    else
        echo "ERROR: Java not found. Install with: brew install openjdk@21"
        exit 1
    fi
fi

exec "$JAVA" \
    -Xmx2G -Xms1G \
    -XX:+UseG1GC \
    -XX:+ParallelRefProcEnabled \
    -XX:MaxGCPauseMillis=200 \
    -jar paper.jar \
    --nogui
