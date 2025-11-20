# RMScript Branch: Core File Changes Summary

This document explains the three core file modifications required to support the rmscript scripting language integration.

## Overview

The rmscript branch adds a natural-language-inspired scripting language for robot programming. 
While most changes are new files in `src/reachy_mini_conversation_app/rmscript/`, three core files required modification to integrate rmscript tools into the existing conversation app architecture.

## 1. Tool Loading System (`core_tools.py`)

**Lines changed:** +109 lines in `_load_profile_tools()`

**Why:** Enable loading `.rmscript` files as conversation tools alongside Python-based tools.

**Logic:**
- Extended tool discovery with a three-tier strategy:
  1. Look for `{tool_name}.rmscript` files in the profile directory
  2. Fall back to `{tool_name}.py` modules in the profile directory
  3. Fall back to shared tools library
- When a `.rmscript` file is found:
  - Compile it using `ReachyMiniScriptCompiler`
  - Dynamically generate a `Tool` subclass using `type()`
  - Register it for LLM function calling
- This allows users to define custom robot behaviors in simple rmscript syntax without writing Python tool boilerplate

## 2. Movement System (`moves.py`)

**Lines changed:** +9 lines in `_get_primary_pose()`

**Why:** Support "hold moves" that maintain the robot's pose while executing non-motion actions.

**Problem:**
- RMScript tools can queue actions like `picture` (camera capture) or `play mysound` (audio playback)
- These actions take time but involve no robot motion
- Without special handling, the robot would jump to neutral pose during these actions

**Solution:**
- Added detection for moves that return `(None, None, None)` for all pose components
- When detected, freeze the robot at its last commanded pose until the action completes
- Used by rmscript's `SoundQueueMove` and `PictureQueueMove` classes

## 3. OpenAI Realtime Handler (`openai_realtime.py`)

**Lines changed:** +5 lines (field filtering) + 36 lines restructured (image handling)

### Change 1: Internal Field Filtering

**Why:** RMScript-compiled tools return internal metadata (e.g., `_move_queue`) that shouldn't pollute LLM conversation context.

**Logic:**
- Filter out fields starting with `_` before sending tool results to OpenAI
- Keeps conversation context clean while allowing tools to return debugging info

### Change 2: Generalized Image Handling

**Why:** Originally, only the `camera` tool could return images. RMScript's `picture` command also needs this capability.

**Logic:**
- Changed from `if tool_name == "camera"` to `if "b64_im" in tool_result`
- Any tool (camera, rmscript picture, future tools) can now return base64-encoded images
- Two-step process:
  1. Add image to OpenAI conversation for vision analysis
  2. Decode and display in Gradio UI
- Changed implementation to decode from base64 instead of fetching from camera worker (more generic)

## Why These Changes Enable RMScript

Together, these changes create the infrastructure for rmscript tools to:
1. **Load dynamically** from simple `.rmscript` files (core_tools)
2. **Queue robot movements and actions** while maintaining pose continuity (moves)
3. **Return rich results** (images, internal state) that integrate cleanly with the conversation system (openai_realtime)

The changes are minimal and preserve backward compatibility—existing Python tools and camera functionality remain unchanged.
