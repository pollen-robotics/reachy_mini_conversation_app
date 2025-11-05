import os
import logging
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Check if .env file exists
env_file = Path(".env")
if not env_file.exists():
    raise RuntimeError(
        ".env file not found. Please create one based on .env.example:\n"
        "  cp .env.example .env\n"
        "Then add your OPENAI_API_KEY to the .env file.",
    )

# Load .env and verify it was loaded successfully
if not load_dotenv():
    raise RuntimeError(
        "Failed to load .env file. Please ensure the file is readable and properly formatted.",
    )

logger.info("Configuration loaded from .env file")


class Config:
    """Configuration class for the conversation app."""

    # OpenAI Configuration (required for OpenAI agent)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Note: Validation happens at runtime when OpenAI agent is selected

    # ElevenLabs Configuration (required for ElevenLabs agent)
    ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # Optional for public agents
    
    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")


config = Config()
