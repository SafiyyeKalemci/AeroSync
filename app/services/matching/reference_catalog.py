from __future__ import annotations

from dataclasses import dataclass

from app.services.matching.reference_state import ReferenceState


@dataclass(frozen=True, slots=True)
class ReferenceCatalog:
    references: tuple[ReferenceState, ...]

    def active_for_frame(self, frame_index: int | None) -> tuple[ReferenceState, ...]:
        return tuple(
            reference
            for reference in self.references
            if reference.is_active(frame_index)
        )
