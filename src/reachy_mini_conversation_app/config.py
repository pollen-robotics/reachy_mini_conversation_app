import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    HANDLER_TYPE = os.getenv("HANDLER_TYPE")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_REALTIME_MODEL_NAME = os.getenv("OPENAI_REALTIME_MODEL_NAME")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_LIVE_MODEL_NAME = os.getenv("GEMINI_LIVE_MODEL_NAME")

    # Validate API keys based on handler type
    if HANDLER_TYPE == "openai":
        if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
            raise RuntimeError(
                "OPENAI_API_KEY is missing or empty for HANDLER_TYPE=openai.\n"
                "Either:\n"
                "  1. Create a .env file with: OPENAI_API_KEY=your_api_key_here (recommended)\n"
                "  2. Set environment variable: export OPENAI_API_KEY=your_api_key_here"
            )
    elif HANDLER_TYPE == "gemini":
        if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
            raise RuntimeError(
                "GEMINI_API_KEY is missing or empty for HANDLER_TYPE=gemini.\n"
                "Either:\n"
                "  1. Create a .env file with: GEMINI_API_KEY=your_api_key_here (recommended)\n"
                "  2. Set environment variable: export GEMINI_API_KEY=your_api_key_here"
            )
    else:
        raise RuntimeError(
            f"Unknown HANDLER_TYPE: {HANDLER_TYPE}\n"
            "Valid options: 'openai' or 'gemini'"
        )

    # Optional vision configuration
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Handler: {HANDLER_TYPE})")
    logger.debug(f"Model: {OPENAI_REALTIME_MODEL_NAME if HANDLER_TYPE == 'openai' else GEMINI_LIVE_MODEL_NAME}")
    logger.debug(f"HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

config = Config()
