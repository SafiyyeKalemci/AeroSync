from __future__ import annotations

import httpx


class BackendClient:
    def __init__(self, base_url: str, timeout: float, headers: dict[str, str]) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._client = httpx.AsyncClient(timeout=timeout)

    async def set_references(self, session_id: str, references: list[dict]) -> int:
        response = await self._client.post(
            f"{self._base_url}/sessions/{session_id}/references",
            json={"references": references},
            headers=self._headers,
        )
        response.raise_for_status()
        return int(response.json()["loaded"])

    async def process_frame(self, frame: dict) -> dict:
        response = await self._client.post(
            f"{self._base_url}/process_frame",
            json=frame,
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
