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

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    # Anthropic (Claude API) - for Linus developer profile
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # GitHub API - for Linus developer profile
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_DEFAULT_OWNER = os.getenv("GITHUB_DEFAULT_OWNER")
    GITHUB_OWNER_EMAIL = os.getenv("GITHUB_OWNER_EMAIL")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass


def set_config_value(key: str, value: str | None) -> bool:
    """Update a configuration value at runtime.

    Updates both the config object and the environment variable.
    Returns True if the update was successful.
    """
    try:
        # Update config object
        if hasattr(config, key):
            setattr(config, key, value)

        # Update environment variable
        if value is not None and value != "":
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

        return True
    except Exception as e:
        logger.error(f"Failed to set config value {key}: {e}")
        return False


def reload_config() -> None:
    """Reload configuration from the .env file.

    This re-reads the .env file and updates the config object
    with fresh values from the environment.
    """
    # Re-find and load .env
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Configuration reloaded from {dotenv_path}")

    # Update config object with fresh values
    config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    config.MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    config.HF_HOME = os.getenv("HF_HOME", "./cache")
    config.LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    config.HF_TOKEN = os.getenv("HF_TOKEN")
    config.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    config.ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    config.GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    config.GITHUB_DEFAULT_OWNER = os.getenv("GITHUB_DEFAULT_OWNER")
    config.GITHUB_OWNER_EMAIL = os.getenv("GITHUB_OWNER_EMAIL")
    config.REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
