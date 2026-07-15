from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.schemas import DetectedObject


@dataclass
class DetectionSessionState:
    session_id: str
    video_name: str | None = None
    previous_frame_gray: object | None = None
    previous_frame_id: str | None = None
    previous_frame_index: int | None = None
    previous_shape: tuple[int, int] | None = None
    warmup_count: int = 0
    last_processed_frame_id: str | None = None
    last_result: list[DetectedObject] | None = None
    freeze_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access_time: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_access_time = time.monotonic()

    def replace_baseline(
        self,
        *,
        video_name: str,
        frame_gray: object,
        frame_id: str,
        frame_index: int | None,
        shape: tuple[int, int],
        result: list[DetectedObject],
        reset_warmup: bool,
        frozen: bool,
    ) -> None:
        if reset_warmup:
            self.warmup_count = 0
        self.warmup_count += 1
        self.video_name = video_name
        self.previous_frame_gray = frame_gray
        self.previous_frame_id = frame_id
        self.previous_frame_index = frame_index
        self.previous_shape = shape
        self.last_processed_frame_id = frame_id
        self.last_result = [item.model_copy(deep=True) for item in result]
        self.freeze_count = self.freeze_count + 1 if frozen else 0
        self.touch()

    def cached_result(self) -> list[DetectedObject] | None:
        if self.last_result is None:
            return None
        return [item.model_copy(deep=True) for item in self.last_result]
