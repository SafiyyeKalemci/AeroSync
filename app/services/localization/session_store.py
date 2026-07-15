from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from app.services.localization.interface import LocalizationSessionState


class LocalizationSessionStore:
    """Keeps state and a per-session lock so stateful VO receives ordered updates."""

    def __init__(self, *, ttl_seconds: float = 1800.0, max_sessions: int = 32) -> None:
        if ttl_seconds <= 0 or max_sessions < 1:
            raise ValueError("Localization session limits are invalid")
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._states: dict[str, LocalizationSessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _entry(self, session_id: str) -> tuple[LocalizationSessionState, asyncio.Lock]:
        async with self._registry_lock:
            self._cleanup()
            if session_id not in self._states and len(self._states) >= self._max_sessions:
                candidates = [key for key, lock in self._locks.items() if not lock.locked()]
                if candidates:
                    oldest = min(candidates, key=lambda key: self._states[key].updated_at)
                    self._states.pop(oldest, None)
                    self._locks.pop(oldest, None)
            state = self._states.setdefault(session_id, LocalizationSessionState(session_id=session_id))
            lock = self._locks.setdefault(session_id, asyncio.Lock())
            return state, lock

    def _cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            key for key, state in self._states.items()
            if (now - state.updated_at).total_seconds() > self._ttl_seconds
            and not self._locks[key].locked()
        ]
        for key in expired:
            self._states.pop(key, None)
            self._locks.pop(key, None)

    @asynccontextmanager
    async def locked(self, session_id: str):
        state, lock = await self._entry(session_id)
        async with lock:
            yield state

    async def reset(self, session_id: str) -> None:
        async with self._registry_lock:
            self._states.pop(session_id, None)
            self._locks.pop(session_id, None)
