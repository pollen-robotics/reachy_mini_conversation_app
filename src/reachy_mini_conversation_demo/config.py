import os

from dotenv import load_dotenv


load_dotenv()


def getenv_bool(key: str, default: bool = False) -> bool:
    """Read env var as a Python bool (case-insensitive)."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"true", "1", "yes", "on"}


class Config:
    """Configuration class for the conversation demo."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set


config = Config()
