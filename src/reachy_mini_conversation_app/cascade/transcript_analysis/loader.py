"""Load transcript reaction configurations from demo modules."""

from __future__ import annotations
import os
import logging
import importlib
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def get_demo_reactions() -> Optional[Dict[str, Any]]:
    """Load reactions configuration from demo module.

    Looks for a 'reactions' dict attribute in the demo module specified
    by the DEMO environment variable.

    The reactions dict should have this structure:
        {
            "keywords": {
                "pizza": async_callback_function,
                ...
            },
            "entities": {
                "food": async_callback_function,
                ...
            },
            "gliner_model": "urchade/gliner_small-v2.1"  # Optional
        }

    Returns:
        Dict with reactions config, or None if no demo or no reactions

    Example:
        >>> os.environ["DEMO"] = "food"
        >>> reactions = get_demo_reactions()
        >>> if reactions:
        ...     keywords = reactions.get("keywords", {})
        ...     entities = reactions.get("entities", {})

    """
    demo = os.getenv("DEMO")
    if not demo:
        logger.debug("No DEMO environment variable set, transcript reactions disabled")
        return None

    try:
        # Import demo module (same pattern as get_session_instructions)
        module = importlib.import_module(f"demos.{demo}")

        # Look for reactions attribute
        reactions = getattr(module, "reactions", None)

        if isinstance(reactions, dict):
            logger.info(f"Loaded reactions from demo '{demo}'")

            # Log what was loaded
            keyword_count = len(reactions.get("keywords", {}))
            entity_count = len(reactions.get("entities", {}))
            gliner_model = reactions.get("gliner_model", "default")

            logger.info(
                f"  Keywords: {keyword_count}, "
                f"Entities: {entity_count}, "
                f"GLiNER model: {gliner_model}"
            )

            return reactions

        logger.info(f"Demo '{demo}' has no 'reactions' attribute, transcript reactions disabled")
        return None

    except ModuleNotFoundError:
        logger.warning(f"Demo module '{demo}' not found, transcript reactions disabled")
        return None

    except Exception as e:
        logger.warning(f"Failed to load reactions from demo '{demo}': {e}")
        return None
