from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.schemas import ImageModality
from app.services.common import FrameContext
from app.services.localization.state import VisualOdometryState


class ContinuityAction(StrEnum):
    FIRST = "first"
    CONTINUE = "continue"
    DUPLICATE = "duplicate"
    REPEATED_IMAGE = "repeated_image"
    RESET = "reset"


@dataclass(frozen=True)
class ContinuityDecision:
    action: ContinuityAction
    event: str | None = None


def evaluate_continuity(
    state: VisualOdometryState,
    frame: FrameContext,
    *,
    shape: tuple[int, int],
    modality: ImageModality | None,
    fingerprint: bytes,
    max_frame_gap: int,
) -> ContinuityDecision:
    if state.previous_gray is None:
        return ContinuityDecision(ContinuityAction.FIRST, "localization_first_frame")
    if frame.frame_id == state.previous_frame_id:
        return ContinuityDecision(ContinuityAction.DUPLICATE, "localization_duplicate_frame")
    if frame.video_name != state.video_name:
        return ContinuityDecision(ContinuityAction.RESET, "localization_video_changed")
    if shape != state.image_shape:
        return ContinuityDecision(ContinuityAction.RESET, "localization_shape_changed")
    if modality != state.modality:
        return ContinuityDecision(ContinuityAction.RESET, "localization_modality_changed")
    if state.previous_frame_index is None or frame.frame_index is None:
        return ContinuityDecision(ContinuityAction.RESET, "localization_frame_gap")
    gap = frame.frame_index - state.previous_frame_index
    if gap <= 0:
        return ContinuityDecision(ContinuityAction.RESET, "localization_out_of_order")
    if gap > max_frame_gap:
        return ContinuityDecision(ContinuityAction.RESET, "localization_frame_gap")
    if fingerprint == state.previous_fingerprint:
        return ContinuityDecision(ContinuityAction.REPEATED_IMAGE, "localization_freeze_detected")
    return ContinuityDecision(ContinuityAction.CONTINUE)

