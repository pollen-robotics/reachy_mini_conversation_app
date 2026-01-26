#!/bin/bash
# Run the reachy_mini_conversation_app with this external profile
#
# This script demonstrates how to load a profile from outside the library
# using the 3 environment variables:
#   - REACHY_MINI_CUSTOM_PROFILE: The profile folder name (e.g., "script")
#   - PROFILES_DIRECTORY: Path to the directory containing the profile folder
#   - TOOLS_DIRECTORY: (optional) Path to additional shared tools
#
# Usage:
#   ./run.sh
#
# Or manually export and run:
#   export REACHY_MINI_CUSTOM_PROFILE=custom_profile
#   export PROFILES_DIRECTORY=/path/to/custom_profiles_and_tools
#   export TOOLS_DIRECTORY=/path/to/custom_profiles_and_tools/custom_tools
#   reachy-mini-conversation

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the environment variables
# PROFILES_DIRECTORY: points to custom_profiles_and_tools (contains custom_profile/)
# REACHY_MINI_CUSTOM_PROFILE: the profile folder name (custom_profile)
# TOOLS_DIRECTORY: points to custom_tool/ (contains new_custom_tool.py)

export REACHY_MINI_CUSTOM_PROFILE=custom_profile # BE CAREFUL TO NOT OVERWRITE THEM INTO YOUR .env FILE
export PROFILES_DIRECTORY="$PROFILE_DIR" # BE CAREFUL TO NOT OVERWRITE THEM INTO YOUR .env FILE
export TOOLS_DIRECTORY="$PROFILE_DIR/custom_tools" # BE CAREFUL TO NOT OVERWRITE THEM INTO YOUR .env FILE

echo "=== Running with external profile ==="
echo "REACHY_MINI_CUSTOM_PROFILE: $REACHY_MINI_CUSTOM_PROFILE"
echo "PROFILES_DIRECTORY: $PROFILES_DIRECTORY"
echo "TOOLS_DIRECTORY: ${TOOLS_DIRECTORY:-<not set>}"
echo "===================================="
echo ""

# Run the conversation app (we assume the reachy daemon is running)
uv run reachy-mini-conversation-app
