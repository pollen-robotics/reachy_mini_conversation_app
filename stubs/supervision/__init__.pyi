"""Type stubs for supervision library."""

from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray


class Detections:
    """Supervision Detections class."""

    xyxy: NDArray[np.float32]
    confidence: NDArray[np.float32] | None
    class_id: NDArray[np.int64] | None
    tracker_id: NDArray[np.int64] | None
    data: dict[str, Any]

    empty: ClassVar["Detections"]

    def __init__(
        self,
        xyxy: NDArray[np.float32],
        confidence: NDArray[np.float32] | None = None,
        class_id: NDArray[np.int64] | None = None,
        tracker_id: NDArray[np.int64] | None = None,
        data: dict[str, Any] | None = None,
    ) -> None: ...

    @classmethod
    def from_ultralytics(cls, ultralytics_results: Any) -> "Detections": ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: Any) -> "Detections": ...
