from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import sys
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Protocol, TypeVar
from urllib.parse import urlparse, urlunparse

from app.core.config import Settings, get_settings
from app.utils.images import detect_image_format
from competition.official_interface_adapter import (
    OfficialAuthenticationError,
    OfficialInterfaceAdapter,
)
from competition.runner import validate_official_progress

logger = logging.getLogger(__name__)
T = TypeVar("T")

EXIT_OK = 0
EXIT_CONFIG = 10
EXIT_AUTH = 20
EXIT_PROGRESS = 30
EXIT_REFERENCE = 40
EXIT_FRAME = 50
EXIT_IMAGE = 60
EXIT_TRANSLATION = 70

SUPPORTED_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp"})
SENSITIVE_FIELD_PARTS = ("password", "token", "authorization", "credential", "secret")


class ReadOnlyOfficialAdapter(Protocol):
    """Dry-run capability list; deliberately contains no result submission API."""

    def authenticate(self) -> None: ...

    def get_progress(self) -> dict | None: ...

    def get_reference_objects(self) -> list[dict]: ...

    def get_current_frame(self) -> dict | None: ...

    def get_current_translation(self) -> dict | None: ...

    def download_media(self, image_url: str, session_name: str, category: str) -> Path: ...


class ConnectionCheckTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class CheckOptions:
    fetch_frame: bool = False
    fetch_references: bool = False
    timeout: float = 30.0
    verbose: bool = False
    allow_missing_translation: bool = False


@dataclass(frozen=True)
class DecodedImageInfo:
    width: int
    height: int
    channels: int
    dtype: str
    image_format: str
    url_extension: str
    shape: tuple[int, ...]


@dataclass
class CheckSummary:
    authentication: str = "FAIL"
    progress: str = "FAIL"
    session_match: str = "FAIL"
    references: int = 0
    frame_fetch: str = "skipped"
    translation_fetch: str = "skipped"
    image_decode: str = "skipped"
    resolution: str = "skipped"
    frames_requested: int = 0
    predictions_submitted: int = 0

    def render(self) -> list[str]:
        return [
            "Dry-run summary:",
            f"  Authentication: {self.authentication}",
            f"  Progress: {self.progress}",
            f"  Session match: {self.session_match}",
            f"  References: {self.references}",
            f"  Frame fetch: {self.frame_fetch}",
            f"  Translation fetch: {self.translation_fetch}",
            f"  Image decode: {self.image_decode}",
            f"  Resolution: {self.resolution}",
            "  Prediction submission: DISABLED",
            f"  Frames requested: {self.frames_requested}",
            f"  Predictions submitted: {self.predictions_submitted}",
        ]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("timeout pozitif olmalıdır")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TEKNOFEST salt-okunur bağlantı dry-run kontrolü"
    )
    parser.add_argument(
        "--fetch-frame",
        action="store_true",
        help="Yalnızca sıradaki tek frame ve translation bilgisini al.",
    )
    parser.add_argument(
        "--fetch-references",
        action="store_true",
        help="Referans metadata yanında görüntüleri geçici indirip doğrula.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        help="Her dry-run adımı için üst süre, saniye (varsayılan: 30).",
    )
    parser.add_argument("--verbose", action="store_true", help="Ayrıntılı güvenli log.")
    parser.add_argument(
        "--allow-missing-translation",
        action="store_true",
        help="Translation yoksa exit 70 yerine WARNING ile başarılı çık.",
    )
    return parser


def _call_with_timeout(operation: Callable[[], T], timeout: float) -> T:
    results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            results.put((True, operation()))
        except BaseException as exc:
            results.put((False, exc))

    threading.Thread(
        target=invoke, daemon=True, name="connection-check-read"
    ).start()
    try:
        succeeded, value = results.get(timeout=timeout)
    except queue.Empty as exc:
        raise ConnectionCheckTimeout(
            f"Dry-run adımı {timeout:g} saniyede tamamlanmadı."
        ) from exc
    if succeeded:
        return value  # type: ignore[return-value]
    raise value  # type: ignore[misc]


def _masked_team_name(value: str) -> str:
    if len(value) <= 2:
        return "*" * len(value)
    return value[:2] + "*" * max(3, len(value) - 3) + value[-1]


def _safe_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def _is_sensitive_field(name: object) -> bool:
    lowered = str(name).lower()
    return any(part in lowered for part in SENSITIVE_FIELD_PARTS)


def _safe_value(value: object) -> object:
    if isinstance(value, str):
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            return _safe_url(value)
        if parsed.query or parsed.fragment:
            return urlunparse(("", "", parsed.path, "", "", ""))
        return value
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not _is_sensitive_field(key)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return value


def _display(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_safe_value(value), ensure_ascii=False, sort_keys=True)
    return str(_safe_value(value))


def _provided(payload: dict, key: str) -> str:
    return _display(payload[key]) if key in payload else "not provided"


def _format_from_url(value: object) -> str:
    if not isinstance(value, str):
        return "not provided"
    suffix = Path(urlparse(value).path).suffix.lower().lstrip(".")
    return suffix or "not provided"


def resolve_frame_number(frame: dict, progress: dict) -> tuple[object, str]:
    for key in ("frame_index", "frame_number", "number", "order"):
        if key in frame and frame[key] is not None:
            return frame[key], f"official frame field: {key}"
    if progress.get("frame_index") is not None:
        return progress["frame_index"], "official progress field: frame_index"
    image_url = frame.get("image_url")
    if isinstance(image_url, str):
        match = re.search(r"(\d+)(?!.*\d)", Path(urlparse(image_url).path).stem)
        if match:
            return int(match.group(1)), "derived from image_url filename"
    return "not provided", "not provided"


def decode_image_bytes(
    content: bytes,
    image_url: str,
    *,
    numpy_module=None,
    cv2_module=None,
) -> DecodedImageInfo:
    if not content:
        raise ValueError("Görüntü bytes boş.")
    image_format = detect_image_format(content)
    extension = _format_from_url(image_url)
    if numpy_module is None:
        import numpy as numpy_module
    if cv2_module is None:
        import cv2 as cv2_module

    memory_stream = BytesIO(content)
    encoded = numpy_module.frombuffer(memory_stream.getbuffer(), dtype=numpy_module.uint8)
    decoded = cv2_module.imdecode(encoded, cv2_module.IMREAD_UNCHANGED)
    if decoded is None or getattr(decoded, "size", 0) == 0:
        raise ValueError("OpenCV görüntüyü RAM içinde decode edemedi.")
    shape = tuple(int(value) for value in decoded.shape)
    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3 and shape[2] > 0:
        height, width, channels = shape
    else:
        raise ValueError(f"Beklenmeyen decoded image shape: {shape}")
    return DecodedImageInfo(
        width=width,
        height=height,
        channels=channels,
        dtype=str(decoded.dtype),
        image_format=image_format,
        url_extension=extension,
        shape=shape,
    )


def _download_bytes_and_delete(
    adapter: ReadOnlyOfficialAdapter,
    image_url: str,
    session_name: str,
    category: str,
    timeout: float,
) -> bytes:
    path = Path(
        _call_with_timeout(
            lambda: adapter.download_media(image_url, session_name, category), timeout
        )
    )
    try:
        return path.read_bytes()
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError("Resmî downloader geçici dosyası silinemedi.") from exc


def _print_config(settings: Settings, emit: Callable[[str], None]) -> None:
    emit("Safe configuration summary:")
    emit(f"  Team: {_masked_team_name(settings.team_name)}")
    emit("  Password: configured")
    emit(f"  Server: {_safe_url(settings.evaluation_server_url)}")
    emit(f"  Session: {settings.official_session_name}")
    emit(f"  Official interface: {settings.official_interface_path}")


def _print_summary(summary: CheckSummary, emit: Callable[[str], None]) -> None:
    for line in summary.render():
        emit(line)


def _report_frame_metadata(
    frame: dict, progress: dict, emit: Callable[[str], None]
) -> None:
    emit("Frame metadata:")
    emit(f"  frame URL/ID: {_provided(frame, 'url')}")
    emit(f"  image_url: {_provided(frame, 'image_url')}")
    emit(f"  video_name: {_provided(frame, 'video_name')}")
    emit(f"  session: {_provided(frame, 'session')}")
    number, source = resolve_frame_number(frame, progress)
    emit(f"  frame number: {_display(number)} ({source})")
    for key in sorted(frame):
        if _is_sensitive_field(key):
            continue
        emit(f"  official field {key}: {_display(frame[key])}")


def _report_translation_metadata(
    translation: dict, emit: Callable[[str], None]
) -> None:
    health_key = "gps_health_status" if "gps_health_status" in translation else "health_status"
    emit(f"  gps_health_status / health_status: {_provided(translation, health_key)}")
    for key in ("translation_x", "translation_y", "translation_z"):
        emit(f"  {key}: {_provided(translation, key)}")
    for key in sorted(translation):
        if _is_sensitive_field(key):
            continue
        emit(f"  official field {key}: {_display(translation[key])}")


def run_connection_check(
    settings: Settings,
    options: CheckOptions,
    *,
    adapter: ReadOnlyOfficialAdapter | None = None,
    adapter_factory: Callable[[Settings], ReadOnlyOfficialAdapter] = OfficialInterfaceAdapter,
    emit: Callable[[str], None] = print,
    image_decoder: Callable[[bytes, str], DecodedImageInfo] = decode_image_bytes,
) -> int:
    summary = CheckSummary()
    try:
        settings.validate_official_integration()
        _print_config(settings, emit)
    except RuntimeError as exc:
        emit(f"Configuration: FAIL ({exc})")
        _print_summary(summary, emit)
        return EXIT_CONFIG
    try:
        if adapter is not None:
            resolved_adapter = adapter
        else:
            # Import the external official interface without writing __pycache__
            # into the read-only source tree supplied by TEKNOFEST.
            previous_bytecode_setting = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                resolved_adapter = adapter_factory(settings)
            finally:
                sys.dont_write_bytecode = previous_bytecode_setting
    except Exception:
        emit("Configuration: FAIL (official interface could not be loaded)")
        _print_summary(summary, emit)
        return EXIT_CONFIG

    emit("Authentication:")
    try:
        _call_with_timeout(resolved_adapter.authenticate, options.timeout)
        summary.authentication = "OK"
        emit("  authentication successful")
    except OfficialAuthenticationError:
        emit("  authentication failed")
        _print_summary(summary, emit)
        return EXIT_AUTH
    except Exception:
        logger.error("connection_check_auth_failed", exc_info=options.verbose)
        emit("  authentication failed")
        _print_summary(summary, emit)
        return EXIT_AUTH

    emit("Progress:")
    try:
        raw_progress = _call_with_timeout(resolved_adapter.get_progress, options.timeout)
        if raw_progress is None:
            raise ValueError("progress unavailable")
        progress = validate_official_progress(raw_progress)
        summary.progress = "OK"
        active_session = progress["session_name"]
        emit(f"  Active session: {active_session or 'none'}")
        emit(f"  Completed: {progress['completed']}")
        emit(f"  Frame index: {progress['frame_index']}/{progress['total_frames']}")
        emit("Session match:")
        if not active_session:
            summary.session_match = "FAIL"
            emit("  FAIL (no active session)")
            _print_summary(summary, emit)
            return EXIT_PROGRESS
        if active_session == settings.official_session_name:
            summary.session_match = "OK"
            emit("  OK")
        else:
            summary.session_match = "WARNING"
            emit("  WARNING (SESSION_NAME differs from active session)")
    except Exception:
        logger.error("connection_check_progress_failed", exc_info=options.verbose)
        emit("  progress check failed")
        _print_summary(summary, emit)
        return EXIT_PROGRESS

    emit("References:")
    try:
        references = _call_with_timeout(
            resolved_adapter.get_reference_objects, options.timeout
        )
        summary.references = len(references)
        emit(f"  Count: {len(references)}")
        for index, reference in enumerate(references, start=1):
            if not isinstance(reference, dict):
                raise ValueError("reference item must be an object")
            emit(
                f"  Reference {index}: order={_provided(reference, 'order')}, "
                f"frame_start={_provided(reference, 'frame_start')}, "
                f"frame_end={_provided(reference, 'frame_end')}, "
                f"format={_format_from_url(reference.get('image_url'))}"
            )
            if options.fetch_references:
                content = _download_bytes_and_delete(
                    resolved_adapter,
                    str(reference["image_url"]),
                    str(progress["session_name"]),
                    "connection-check-references",
                    options.timeout,
                )
                emit(f"    Downloaded: format={detect_image_format(content)}; temporary file removed")
    except Exception:
        logger.error("connection_check_reference_failed", exc_info=options.verbose)
        emit("  reference check failed")
        _print_summary(summary, emit)
        return EXIT_REFERENCE

    if options.fetch_frame:
        try:
            summary.frames_requested = 1
            frame = _call_with_timeout(resolved_adapter.get_current_frame, options.timeout)
            if not isinstance(frame, dict):
                raise ValueError("frame unavailable")
            summary.frame_fetch = "OK"
            _report_frame_metadata(frame, progress, emit)
            image_url = frame.get("image_url")
            if not image_url:
                raise ValueError("frame image_url missing")
        except Exception:
            summary.frame_fetch = "FAIL"
            logger.error("connection_check_frame_failed", exc_info=options.verbose)
            emit("Frame metadata:\n  frame fetch failed")
            _print_summary(summary, emit)
            return EXIT_FRAME

        emit("Image validation:")
        try:
            content = _download_bytes_and_delete(
                resolved_adapter,
                str(image_url),
                str(progress["session_name"]),
                "connection-check-frame",
                options.timeout,
            )
            info = image_decoder(content, str(image_url))
            summary.image_decode = "OK"
            summary.resolution = (
                "OK" if (info.width, info.height) == (1920, 1080) else "WARNING"
            )
            emit(f"  width: {info.width}")
            emit(f"  height: {info.height}")
            emit(f"  channel count: {info.channels}")
            emit(f"  NumPy dtype: {info.dtype}")
            emit(f"  image format (magic bytes): {info.image_format}")
            emit(f"  image URL extension: {info.url_extension}")
            emit(f"  decoded image shape: {info.shape}")
            emit("  OpenCV decode: successful")
            emit(
                "  supported format: "
                + ("yes" if info.image_format in {"jpeg", "png", "webp"} else "no")
            )
            emit("  image empty/corrupt: no")
            emit(f"  resolution 1920x1080: {summary.resolution}")
            emit("  official downloader temporary file: removed")
        except Exception:
            summary.image_decode = "FAIL"
            summary.resolution = "FAIL"
            logger.error("connection_check_image_failed", exc_info=options.verbose)
            emit("  download or OpenCV RAM decode failed")
            _print_summary(summary, emit)
            return EXIT_IMAGE

        emit("Translation metadata:")
        try:
            translation = _call_with_timeout(
                resolved_adapter.get_current_translation, options.timeout
            )
            if translation is None:
                if options.allow_missing_translation:
                    summary.translation_fetch = "WARNING"
                    emit("  not provided (allowed)")
                else:
                    summary.translation_fetch = "FAIL"
                    emit("  not provided")
                    _print_summary(summary, emit)
                    return EXIT_TRANSLATION
            elif not isinstance(translation, dict):
                raise ValueError("translation response must be an object")
            else:
                summary.translation_fetch = "OK"
                _report_translation_metadata(translation, emit)
        except Exception:
            summary.translation_fetch = "FAIL"
            logger.error("connection_check_translation_failed", exc_info=options.verbose)
            emit("  translation fetch failed")
            _print_summary(summary, emit)
            return EXIT_TRANSLATION

    _print_summary(summary, emit)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if namespace.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return run_connection_check(
        get_settings(),
        CheckOptions(
            fetch_frame=namespace.fetch_frame,
            fetch_references=namespace.fetch_references,
            timeout=namespace.timeout,
            verbose=namespace.verbose,
            allow_missing_translation=namespace.allow_missing_translation,
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
