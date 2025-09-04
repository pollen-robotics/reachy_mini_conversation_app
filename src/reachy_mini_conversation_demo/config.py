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
    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-realtime-preview")
    SIM = getenv_bool("SIM", False)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
    VISION_ENABLED = getenv_bool("VISION_ENABLED", False)
    HEAD_TRACKING = getenv_bool("HEAD_TRACKING", False)

config = Config()