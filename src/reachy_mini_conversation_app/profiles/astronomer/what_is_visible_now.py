import logging
from typing import Any, Dict
from datetime import UTC, datetime

from .astronomer import get_catalog, find_celestial_angles
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class WhatIsVisibleNow(Tool):
    """List classic celestial objects currently visible above the horizon."""

    name = "what_is_visible_now"
    description = "Get a list of the most interesting celestial objects (stars, planets, deep sky objects) currently visible above the horizon."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute what is visible now tool."""

        time = datetime.now(UTC)

        catalog = get_catalog()

        visible_objects = []

        for obj in catalog.objects:
            if not obj.classic:
                continue

            # Calculate position using high-level API
            result = find_celestial_angles(obj.name, time = time)

            if not result['found']:
                continue

            # Only include objects above the horizon
            if result['altitude'] > 0:
                visible_objects.append({
                    'name': obj.name,
                    'type': obj.type,
                    'altitude': result['altitude'],
                    'azimuth': result['azimuth']
                })

        # Sort by altitude (highest first)
        visible_objects.sort(key=lambda x: x['altitude'], reverse=True)

        # Format results
        if not visible_objects:
            status = "No classic celestial objects are currently visible above the horizon."
        else:
            status_lines = [f"Currently visible classic objects ({len(visible_objects)} total):"]

            # Group by type
            by_type = {}
            for obj in visible_objects:
                obj_type = obj['type']
                if obj_type not in by_type:
                    by_type[obj_type] = []
                by_type[obj_type].append(obj)

            # Solar system objects first
            if 'star' in by_type and any(obj['name'] in ['Sun', 'Moon'] for obj in by_type.get('star', []) + by_type.get('satellite', [])):
                solar_system = [obj for obj in visible_objects if obj['name'] in ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']]
                if solar_system:
                    status_lines.append("\nSolar System:")
                    for obj in solar_system:
                        status_lines.append(f"  - {obj['name']} (alt: {obj['altitude']:.0f}°)")

            # Planets
            planets = [obj for obj in visible_objects if obj['type'] == 'planet']
            if planets:
                status_lines.append("\nPlanets:")
                for obj in planets:
                    status_lines.append(f"  - {obj['name']} (alt: {obj['altitude']:.0f}°)")

            # Bright stars
            stars = [obj for obj in visible_objects if obj['type'] == 'star' and obj['name'] not in ['Sun', 'Moon']]
            if stars:
                status_lines.append("\nBright Stars:")
                for obj in stars[:10]:  # Limit to top 10 stars
                    status_lines.append(f"  - {obj['name']} (alt: {obj['altitude']:.0f}°)")

            # Constellations
            constellations = [obj for obj in visible_objects if obj['type'] == 'constellation']
            if constellations:
                status_lines.append("\nConstellations:")
                for obj in constellations[:8]:  # Limit to 8 constellations
                    status_lines.append(f"  - {obj['name']} (alt: {obj['altitude']:.0f}°)")

            # Deep sky objects
            deep_sky = [obj for obj in visible_objects if obj['type'] in ['galaxy', 'nebula', 'cluster']]
            if deep_sky:
                status_lines.append("\nDeep Sky Objects:")
                for obj in deep_sky:
                    status_lines.append(f"  - {obj['name']} ({obj['type']}, alt: {obj['altitude']:.0f}°)")

            status = "\n".join(status_lines)

        logger.info("Tool call: what_is_visible_now found %d visible classic objects", len(visible_objects))

        return {"status": status}
