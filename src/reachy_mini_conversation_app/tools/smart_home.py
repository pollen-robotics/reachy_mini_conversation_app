"""Smart home control tool using Home Assistant REST API."""

import logging
import os
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# Valid actions per domain
DOMAIN_ACTIONS = {
    "light": ["on", "off", "toggle", "brightness", "color", "list"],
    "media_player": ["play", "pause", "stop", "volume", "mute", "list"],
}


class SmartHome(Tool):
    """Control Home Assistant devices including lights and media players."""

    name = "smart_home"
    description = (
        "Control smart home devices via Home Assistant. "
        "Use domain='light' for lights (on, off, toggle, brightness, color, list). "
        "Use domain='media_player' for media (play, pause, stop, volume, mute, list). "
        "Use action='list' to discover available devices in a domain."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "enum": ["light", "media_player"],
                "description": "Device domain: 'light' or 'media_player'",
            },
            "action": {
                "type": "string",
                "description": "Action to perform. Lights: on, off, toggle, brightness, color, list. Media: play, pause, stop, volume, mute, list",
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID (e.g., 'light.living_room', 'media_player.tv'). Required for all actions except 'list'",
            },
            "brightness": {
                "type": "integer",
                "description": "Brightness level 0-100 (for lights with action='brightness' or 'on')",
            },
            "color": {
                "type": "string",
                "description": "Color name or hex code (for lights with action='color')",
            },
            "volume": {
                "type": "integer",
                "description": "Volume level 0-100 (for media_player with action='volume')",
            },
        },
        "required": ["domain", "action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute smart home command via Home Assistant API."""
        domain = kwargs.get("domain")
        action = kwargs.get("action")
        entity_id = kwargs.get("entity_id")
        brightness = kwargs.get("brightness")
        color = kwargs.get("color")
        volume = kwargs.get("volume")

        # Validate required params
        if not domain or not action:
            return {"error": "domain and action are required"}

        # Validate action for domain
        valid_actions = DOMAIN_ACTIONS.get(domain, [])
        if action not in valid_actions:
            return {
                "error": f"Invalid action '{action}' for domain '{domain}'. Valid: {valid_actions}"
            }

        # Require entity_id for non-list actions
        if action != "list" and not entity_id:
            return {"error": f"entity_id is required for action '{action}'"}

        # Get Home Assistant configuration
        base_url = os.getenv("HOME_ASSISTANT_URL")
        token = os.getenv("HOME_ASSISTANT_TOKEN")

        if not base_url or not token:
            logger.error("HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN not configured")
            return {"error": "Home Assistant not configured. Set HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN."}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        logger.info("Smart home: domain=%s, action=%s, entity_id=%s", domain, action, entity_id)

        try:
            async with httpx.AsyncClient() as client:
                # Handle list action - get all entities in domain
                if action == "list":
                    return await self._list_entities(client, base_url, headers, domain)

                # Handle domain-specific actions
                if domain == "light":
                    return await self._control_light(
                        client, base_url, headers, action, entity_id, brightness, color
                    )
                elif domain == "media_player":
                    return await self._control_media(
                        client, base_url, headers, action, entity_id, volume
                    )

        except httpx.HTTPStatusError as e:
            logger.exception("Home Assistant HTTP error")
            return {"error": f"Home Assistant API error: {e.response.status_code}"}
        except httpx.TimeoutException:
            logger.exception("Home Assistant timeout")
            return {"error": "Home Assistant request timed out"}
        except Exception as e:
            logger.exception("Smart home control failed")
            return {"error": f"Smart home control failed: {str(e)}"}

    async def _list_entities(
        self, client: httpx.AsyncClient, base_url: str, headers: dict, domain: str
    ) -> Dict[str, Any]:
        """List all entities in a domain."""
        response = await client.get(
            f"{base_url}/api/states",
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        states = response.json()

        # Filter by domain
        entities = []
        for state in states:
            if state["entity_id"].startswith(f"{domain}."):
                entities.append({
                    "entity_id": state["entity_id"],
                    "name": state["attributes"].get("friendly_name", state["entity_id"]),
                    "state": state["state"],
                })

        logger.info("Found %d %s entities", len(entities), domain)
        return {
            "status": "success",
            "domain": domain,
            "entities": entities,
        }

    async def _control_light(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict,
        action: str,
        entity_id: str,
        brightness: int | None,
        color: str | None,
    ) -> Dict[str, Any]:
        """Control a light entity."""
        service_data = {"entity_id": entity_id}

        # Map actions to Home Assistant services
        if action == "on":
            service = "turn_on"
            if brightness is not None:
                # Home Assistant uses 0-255 for brightness
                service_data["brightness"] = int(brightness * 2.55)
        elif action == "off":
            service = "turn_off"
        elif action == "toggle":
            service = "toggle"
        elif action == "brightness":
            service = "turn_on"
            if brightness is not None:
                service_data["brightness"] = int(brightness * 2.55)
            else:
                return {"error": "brightness value required for brightness action"}
        elif action == "color":
            service = "turn_on"
            if color:
                # Try to parse color - support named colors and hex
                if color.startswith("#"):
                    # Hex color
                    hex_color = color.lstrip("#")
                    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                    service_data["rgb_color"] = list(rgb)
                else:
                    # Named color - let Home Assistant handle it
                    service_data["color_name"] = color
            else:
                return {"error": "color value required for color action"}
        else:
            return {"error": f"Unknown light action: {action}"}

        # Call Home Assistant service
        response = await client.post(
            f"{base_url}/api/services/light/{service}",
            headers=headers,
            json=service_data,
            timeout=10.0,
        )
        response.raise_for_status()

        logger.info("Light %s: %s", action, entity_id)
        return {
            "status": "success",
            "domain": "light",
            "action": action,
            "entity_id": entity_id,
        }

    async def _control_media(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        headers: dict,
        action: str,
        entity_id: str,
        volume: int | None,
    ) -> Dict[str, Any]:
        """Control a media player entity."""
        service_data = {"entity_id": entity_id}

        # Map actions to Home Assistant services
        if action == "play":
            service = "media_play"
        elif action == "pause":
            service = "media_pause"
        elif action == "stop":
            service = "media_stop"
        elif action == "volume":
            service = "volume_set"
            if volume is not None:
                # Home Assistant uses 0.0-1.0 for volume
                service_data["volume_level"] = volume / 100.0
            else:
                return {"error": "volume value required for volume action"}
        elif action == "mute":
            service = "volume_mute"
            service_data["is_volume_muted"] = True
        else:
            return {"error": f"Unknown media_player action: {action}"}

        # Call Home Assistant service
        response = await client.post(
            f"{base_url}/api/services/media_player/{service}",
            headers=headers,
            json=service_data,
            timeout=10.0,
        )
        response.raise_for_status()

        logger.info("Media player %s: %s", action, entity_id)
        return {
            "status": "success",
            "domain": "media_player",
            "action": action,
            "entity_id": entity_id,
        }
