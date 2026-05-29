"""Closing-the-loop observer (Tier 0).

Read-side of the robot event log written by
``robot_comic.telemetry.JsonlEventExporter``. Lets an autonomous agent
(Claude Code / Hermes) ask *"what did the robot do recently?"*.

See ``docs/closing-the-loop.md`` for the full design.
"""
