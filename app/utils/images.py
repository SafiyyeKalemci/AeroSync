from __future__ import annotations

from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx

SUPPORTED_IMAGE_FORMATS = frozenset({"jpg", "jpeg", "png", "webp"})


def detect_image_format(content: bytes) -> str:
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    raise ValueError("Desteklenmeyen veya bozuk görüntü içeriği; jpg, jpeg, png, webp bekleniyor.")


async def read_image_bytes(source: str, timeout: float) -> bytes:
    parsed = urlparse(source)
    is_windows_path = bool(PureWindowsPath(source).drive)
    if parsed.scheme in {"http", "https"}:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(source)
            response.raise_for_status()
            content = response.content
            detect_image_format(content)
            return content
    if parsed.scheme not in {"", "file"} and not is_windows_path:
        raise ValueError(f"Desteklenmeyen görüntü şeması: {parsed.scheme}")
    if is_windows_path:
        path_value = source
    elif parsed.scheme == "file":
        path_value = url2pathname(parsed.path)
    else:
        path_value = source
    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower().lstrip(".") not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError("Desteklenmeyen görüntü uzantısı; jpg, jpeg, png, webp bekleniyor.")
    content = path.read_bytes()
    detect_image_format(content)
    return content
