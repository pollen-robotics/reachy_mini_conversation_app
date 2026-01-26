#!/bin/bash
# Setup and run web-agent for browser automation
# Repository: https://github.com/BrowserOperator/web-agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEB_AGENT_DIR="$PROJECT_DIR/web-agent"
LOCAL_DIR="$WEB_AGENT_DIR/deployments/local"
REPO_URL="https://github.com/BrowserOperator/web-agent.git"

echo "=== Web-Agent Setup ==="

# Check prerequisites
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Clone or pull latest
if [ -d "$WEB_AGENT_DIR" ]; then
    echo "Updating existing web-agent..."
    cd "$WEB_AGENT_DIR"
    git pull
else
    echo "Cloning web-agent repository..."
    git clone "$REPO_URL" "$WEB_AGENT_DIR"
fi

cd "$LOCAL_DIR"

# Build and run using the project's Makefile (init is called by build)
echo ""
echo "Building web-agent (this may take ~30 min on first run)..."
make build

echo ""
echo "Starting web-agent container..."
make run

echo ""
echo "=== Web-Agent Started ==="
echo "Service running at: http://localhost:8080"
echo "WebRTC browser view: http://localhost:8000"
echo ""
echo "To use with reachy_mini_conversation_app, set in .env:"
echo "  WEB_AGENT_ENDPOINT=http://localhost:8080"
echo "  WEB_AGENT_PATH=$WEB_AGENT_DIR"
