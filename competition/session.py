from dataclasses import dataclass, field
from time import monotonic


@dataclass
class MissionSession:
    duration_seconds: float
    started_at: float = field(default_factory=monotonic)
    reference_sessions: set[str] = field(default_factory=set)

    @property
    def expired(self) -> bool:
        return monotonic() - self.started_at >= self.duration_seconds
