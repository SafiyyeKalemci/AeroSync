from __future__ import annotations

import asyncio
import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas import CompetitionResponse
from app.services.common import FrameContext
from app.services.frame_processor import FrameProcessor
from app.services.registry import ServiceRegistry, build_services
from competition.frame_mapper import map_official_frame
from competition.official_interface_adapter import (
    OfficialAuthenticationError,
    OfficialInterfaceAdapter,
    OfficialInterfaceError,
)
from competition.reference_mapper import ReferenceCatalog, map_official_references
from competition.result_mapper import map_aerosync_result

logger = logging.getLogger(__name__)


class CompetitionRunError(RuntimeError):
    exit_code = 1


class OfficialServerUnavailable(CompetitionRunError):
    exit_code = 4


class OfficialSessionUnavailable(CompetitionRunError):
    exit_code = 3


class FrameUnavailable(CompetitionRunError):
    exit_code = 5


class TaskProcessingTimeout(CompetitionRunError):
    exit_code = 6


class PredictionRejected(CompetitionRunError):
    exit_code = 7


class DuplicateFrameError(CompetitionRunError):
    exit_code = 8


@dataclass
class PendingPrediction:
    frame_id: str
    prediction: object


def validate_official_progress(progress: object) -> dict:
    if not isinstance(progress, dict):
        raise OfficialServerUnavailable("Resmî progress yanıtı JSON nesnesi olmalıdır.")
    required = {"frame_index", "total_frames", "completed", "session_name"}
    missing = sorted(required - set(progress))
    if missing:
        raise OfficialServerUnavailable(
            "Resmî progress JSON alanları eksik: " + ", ".join(missing)
        )
    unknown = sorted(set(progress) - required)
    if unknown:
        logger.warning(
            "official_payload_unknown_fields",
            extra={
                "event": "official_payload_unknown_fields",
                "source": "progress",
                "fields": unknown,
            },
        )
    try:
        frame_index = int(progress["frame_index"])
        total_frames = int(progress["total_frames"])
    except (TypeError, ValueError) as exc:
        raise OfficialServerUnavailable(
            "Resmî progress frame_index/total_frames tamsayı olmalıdır."
        ) from exc
    if frame_index < 0 or total_frames < 0 or frame_index > total_frames:
        raise OfficialServerUnavailable("Resmî progress frame aralığı geçersiz.")
    if not isinstance(progress["completed"], bool):
        raise OfficialServerUnavailable("Resmî progress completed boolean olmalıdır.")
    session_name = progress["session_name"]
    if session_name is not None and not isinstance(session_name, str):
        raise OfficialServerUnavailable("Resmî progress session_name string/null olmalıdır.")
    return {
        **progress,
        "frame_index": frame_index,
        "total_frames": total_frames,
    }


class OfficialCompetitionRunner:
    def __init__(
        self,
        settings: Settings,
        adapter: OfficialInterfaceAdapter,
        services: ServiceRegistry,
    ) -> None:
        self.settings = settings
        self.adapter = adapter
        self.services = services
        self.processor = FrameProcessor(services)
        self.catalog = ReferenceCatalog(
            references=[], official_urls={}, image_url_windows={}
        )
        self.session_id: str | None = None
        self.submitted_frames: set[str] = set()
        self.pending: PendingPrediction | None = None
        self._loaded_reference_ids: frozenset[int] = frozenset()
        self._response_ids = itertools.count(1)

    async def initialize(self) -> dict:
        await asyncio.to_thread(self.adapter.authenticate)
        progress = await asyncio.to_thread(self.adapter.get_progress)
        if progress is None:
            raise OfficialServerUnavailable(
                "Resmî progress bilgisi alınamadı."
            )
        progress = validate_official_progress(progress)
        active_session = progress["session_name"]
        if not active_session:
            raise OfficialSessionUnavailable("Sunucuda aktif oturum bulunmuyor.")
        if active_session != self.settings.official_session_name:
            raise OfficialSessionUnavailable(
                "Aktif resmî oturum SESSION_NAME ile eşleşmiyor."
            )
        self.session_id = str(active_session)
        await self._reset_session_state()
        if not progress["completed"]:
            await self._load_references()
        return progress

    async def _reset_session_state(self) -> None:
        assert self.session_id is not None
        await self.services.detection.reset_session(self.session_id)
        await self.services.matching.clear_session(self.session_id)
        await self.services.localization.reset_session(self.session_id)
        await self.services.localization_sessions.reset(self.session_id)
        self.catalog = ReferenceCatalog(
            references=[], official_urls={}, image_url_windows={}
        )
        self._loaded_reference_ids = frozenset()
        self.submitted_frames.clear()
        self.pending = None
        logger.info(
            "official_session_state_reset",
            extra={"event": "official_session_state_reset", "session_id": self.session_id},
        )

    async def _load_references(self) -> None:
        assert self.session_id is not None
        payloads = await asyncio.to_thread(self.adapter.get_reference_objects)
        content_by_url: dict[str, bytes] = {}
        for payload in payloads:
            if "url" not in payload or "image_url" not in payload:
                raise OfficialServerUnavailable(
                    "Resmî reference JSON içinde url/image_url eksik."
                )
            path = await asyncio.to_thread(
                self.adapter.download_media,
                str(payload["image_url"]),
                self.session_id,
                "references",
            )
            content_by_url[str(payload["url"])] = Path(path).read_bytes()
        self.catalog = map_official_references(payloads, content_by_url)
        initial_references = self.catalog.initially_loadable_references()
        loaded = await self.services.matching.set_references(
            self.session_id, initial_references
        )
        self._loaded_reference_ids = frozenset(
            item.object_id for item in initial_references
        )
        logger.info(
            "official_references_loaded",
            extra={
                "event": "official_references_loaded",
                "session_id": self.session_id,
                "received": len(payloads),
                "loaded": loaded,
            },
        )

    async def process_next_frame(self, frame_index: int) -> bool:
        if self.pending is not None:
            await self._submit_pending()
            return True
        assert self.session_id is not None
        official_frame = await asyncio.to_thread(self.adapter.get_current_frame)
        if official_frame is None:
            progress = await asyncio.to_thread(self.adapter.get_progress)
            if progress is None:
                # Official GET helpers do not expose the HTTP status. A single
                # credential refresh distinguishes a possibly expired token
                # without inventing another HTTP client or endpoint.
                await asyncio.to_thread(self.adapter.authenticate)
                progress = await asyncio.to_thread(self.adapter.get_progress)
                if progress is None:
                    raise OfficialServerUnavailable(
                        "Token yenileme sonrasında progress bilgisi alınamadı."
                    )
            progress = validate_official_progress(progress)
            if progress["completed"] or not progress["session_name"]:
                return False
            await asyncio.to_thread(self.adapter.authenticate)
            official_frame = await asyncio.to_thread(self.adapter.get_current_frame)
            if official_frame is None:
                raise FrameUnavailable(
                    "Aktif oturum sürerken token yenileme sonrasında frame alınamadı."
                )
        frame_id = str(official_frame.get("url", ""))
        if not frame_id:
            raise FrameUnavailable("Resmî frame URL/kimliği eksik.")
        if frame_id in self.submitted_frames:
            logger.error(
                "official_duplicate_frame_after_success",
                extra={"event": "official_duplicate_frame_after_success", "frame_id": frame_id},
            )
            raise DuplicateFrameError(
                "Başarıyla gönderilmiş frame yeniden geldi; çift gönderim engellendi."
            )

        official_translation = await asyncio.to_thread(
            self.adapter.get_current_translation
        )
        local_image = await asyncio.to_thread(
            self.adapter.download_media,
            str(official_frame.get("image_url", "")),
            self.session_id,
            "frames",
        )
        request = map_official_frame(
            official_frame,
            official_translation,
            session_id=self.session_id,
            frame_index=frame_index,
            local_image_path=local_image,
        )
        if self.catalog.requires_dynamic_activation:
            selected = self.catalog.references_for_frame(
                frame_index, str(official_frame["image_url"])
            )
            selected_ids = frozenset(item.object_id for item in selected)
            if selected_ids != self._loaded_reference_ids:
                await self.services.matching.set_references(self.session_id, selected)
                self._loaded_reference_ids = selected_ids
        context = FrameContext.from_request(request)
        try:
            async with self.services.localization_sessions.locked(self.session_id) as state:
                task_results = await asyncio.wait_for(
                    self.processor.process(context, state),
                    timeout=self.settings.competition_task_timeout_seconds,
                )
        except TimeoutError as exc:
            logger.error(
                "official_aerosync_task_timeout",
                extra={"event": "official_aerosync_task_timeout", "frame_id": frame_id},
            )
            raise TaskProcessingTimeout(
                "AeroSync frame işleme zaman aşımına uğradı."
            ) from exc

        response = CompetitionResponse.from_task_results(
            response_id=next(self._response_ids),
            user=self.settings.team_user_url,
            frame=request.url,
            detected_objects=task_results.detected_objects,
            detected_translation=task_results.detected_translation,
            matched_reference_objects=task_results.matched_reference_objects,
        )
        prediction = map_aerosync_result(
            response,
            official_frame=official_frame,
            official_translation=official_translation,
            catalog=self.catalog,
            bindings=self.adapter.bindings,
        )
        self.pending = PendingPrediction(frame_id=frame_id, prediction=prediction)
        await self._submit_pending()
        return True

    async def _submit_pending(self) -> None:
        assert self.pending is not None
        wait_time = self.settings.competition_retry_initial_seconds
        reauthenticated = False
        for attempt in range(self.settings.competition_max_retries):
            response = await asyncio.to_thread(
                self.adapter.send_prediction, self.pending.prediction
            )
            status_code = getattr(response, "status_code", None)
            if status_code in {201, 406}:
                self.submitted_frames.add(self.pending.frame_id)
                self.pending = None
                return
            if status_code == 401 and not reauthenticated:
                await asyncio.to_thread(self.adapter.authenticate)
                reauthenticated = True
                continue
            logger.warning(
                "official_prediction_retry",
                extra={
                    "event": "official_prediction_retry",
                    "frame_id": self.pending.frame_id,
                    "attempt": attempt + 1,
                    "status_code": status_code,
                },
            )
            if attempt + 1 < self.settings.competition_max_retries:
                await asyncio.sleep(wait_time)
                wait_time *= 2
        raise PredictionRejected(
            "Sonuç resmî sunucu tarafından kabul edilmedi; frame beklemede tutuldu."
        )

    async def run(self) -> None:
        progress = await self.initialize()
        if progress["completed"]:
            return
        frame_index = int(progress["frame_index"])
        while await self.process_next_frame(frame_index):
            frame_index += 1
            if self.settings.competition_frame_interval_seconds:
                await asyncio.sleep(self.settings.competition_frame_interval_seconds)


async def async_main() -> int:
    settings = get_settings()
    try:
        settings.validate_official_integration()
        adapter = OfficialInterfaceAdapter(settings)
        runner = OfficialCompetitionRunner(settings, adapter, build_services(settings))
        await runner.run()
        return 0
    except OfficialAuthenticationError:
        logger.error("official_runner_authentication_failed")
        return 2
    except CompetitionRunError as exc:
        logger.error(
            "official_runner_failed",
            extra={"event": "official_runner_failed", "exit_code": exc.exit_code},
        )
        return exc.exit_code
    except OfficialInterfaceError:
        logger.error("official_interface_failed", exc_info=True)
        return 9
    except RuntimeError as exc:
        logger.error("official_config_invalid: %s", exc)
        return 10
    except Exception:
        logger.error("official_runner_unexpected_failure", exc_info=True)
        return 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
