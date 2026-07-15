from __future__ import annotations

import threading
import time

from app.services.detection.session_state import DetectionSessionState


class DetectionSessionStore:
    """Bounded, TTL-cleaned store with one processing lock per session."""

    def __init__(self, *, ttl_seconds: float, max_sessions: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds pozitif olmalıdır")
        if max_sessions < 1:
            raise ValueError("max_sessions en az 1 olmalıdır")
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._states: dict[str, DetectionSessionState] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: str) -> DetectionSessionState:
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            state = self._states.get(session_id)
            if state is None:
                self._enforce_capacity_locked()
                state = DetectionSessionState(session_id=session_id)
                self._states[session_id] = state
            state.last_access_time = now
            return state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

    def reset_all(self) -> None:
        with self._lock:
            self._states.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, state in self._states.items()
            if not state.lock.locked()
            and now - state.last_access_time > self._ttl_seconds
        ]
        for session_id in expired:
            self._states.pop(session_id, None)

    def _enforce_capacity_locked(self) -> None:
        if len(self._states) < self._max_sessions:
            return
        candidates = [state for state in self._states.values() if not state.lock.locked()]
        if not candidates:
            return
        oldest = min(candidates, key=lambda state: state.last_access_time)
        self._states.pop(oldest.session_id, None)
