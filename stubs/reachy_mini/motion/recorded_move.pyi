"""Type stubs for reachy_mini.motion.recorded_move module."""

from reachy_mini.motion.move import Move

class RecordedMoves:
    """Collection of pre-recorded moves."""

    def __init__(self, repo_id: str = ..., **kwargs: object) -> None: ...
    def get(self, name: str) -> Move:
        """Get a recorded move by name.

        Args:
            name: Name of the move to retrieve.

        Returns:
            The requested Move object.

        """
        ...

    def list_moves(self) -> list[str]:
        """List all available move names.

        Returns:
            List of move names.

        """
        ...
