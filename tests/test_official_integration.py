from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import PROJECT_ROOT, _project_path, get_settings
from app.schemas import (
    CompetitionResponse,
    DetectedObject,
    DetectedTranslation,
    LandingStatus,
    MatchedReferenceObject,
    MotionStatus,
    ObjectClass,
)
from app.services.registry import build_services
from competition.frame_mapper import OfficialFrameMappingError, map_official_frame
from competition.official_interface_adapter import (
    OfficialAuthenticationError,
    OfficialBindings,
    OfficialInterfaceAdapter,
)
from competition.reference_mapper import map_official_references
from competition.result_mapper import map_aerosync_result
from competition.runner import (
    DuplicateFrameError,
    OfficialCompetitionRunner,
    OfficialSessionUnavailable,
    PredictionRejected,
    OfficialServerUnavailable,
    validate_official_progress,
)


class FakeDetectedObject:
    def __init__(self, cls, landing, moving, x1, y1, x2, y2):
        self.values = (cls, landing, moving, x1, y1, x2, y2)

    def create_payload(self, base_url):
        cls, landing, moving, x1, y1, x2, y2 = self.values
        return {
            "cls": f"{base_url}classes/{int(cls[0]) + 1}/",
            "landing_status": str(landing),
            "moving_status": str(moving),
            "top_left_x": str(x1),
            "top_left_y": str(y1),
            "bottom_right_x": str(x2),
            "bottom_right_y": str(y2),
        }


class FakeTranslation:
    def __init__(self, x, y, z):
        self.values = (x, y, z)

    def create_payload(self):
        x, y, z = self.values
        return {
            "translation_x": str(x),
            "translation_y": str(y),
            "translation_z": str(z),
        }


class FakeReferencePrediction:
    def __init__(self, reference, frame, x1, y1, x2, y2):
        self.values = (reference, frame, x1, y1, x2, y2)

    def create_payload(self):
        reference, frame, x1, y1, x2, y2 = self.values
        return {
            "reference": reference,
            "frame": frame,
            "top_left_x": str(x1),
            "top_left_y": str(y1),
            "bottom_right_x": str(x2),
            "bottom_right_y": str(y2),
        }


class FakeFramePredictions:
    def __init__(self, frame_url, image_url, video_name, x, y, z):
        self.frame_url = frame_url
        self.detected_objects = []
        self.translations = []
        self.reference_predictions = []

    def add_detected_object(self, item):
        self.detected_objects.append(item)

    def add_translation_object(self, item):
        self.translations.append(item)

    def add_reference_prediction(self, item):
        self.reference_predictions.append(item)

    def create_payload(self, base_url):
        return {
            "frame": self.frame_url,
            "detected_objects": [x.create_payload(base_url) for x in self.detected_objects],
            "detected_translations": [x.create_payload() for x in self.translations],
            "reference_predictions": [x.create_payload() for x in self.reference_predictions],
        }


def bindings():
    return OfficialBindings(
        connection_handler=object,
        frame_predictions=FakeFramePredictions,
        detected_object=FakeDetectedObject,
        detected_translation=FakeTranslation,
        reference_prediction=FakeReferencePrediction,
        image_downloader=lambda *args, **kwargs: None,
    )


def test_relative_official_interface_path_resolves_from_project_root(monkeypatch):
    monkeypatch.setenv("OFFICIAL_INTERFACE_PATH", "official_interface")
    assert _project_path("OFFICIAL_INTERFACE_PATH") == (PROJECT_ROOT / "official_interface").resolve()


def settings(tmp_path, **changes):
    overrides = {
        "team_name": "dummy-team",
        "password": "dummy-password",
        "evaluation_server_url": "http://example.invalid/",
        "official_session_name": "test-session",
        "official_interface_path": tmp_path,
        "official_media_dir": tmp_path / "media",
        "competition_max_retries": 2,
        "competition_retry_initial_seconds": 0.0,
        "competition_frame_interval_seconds": 0.0,
        "competition_task_timeout_seconds": 1.0,
        "matching_enabled": False,
    }
    overrides.update(changes)
    return replace(get_settings(), **overrides)


class FakeOfficialClient:
    def __init__(self, *, login_ok=True):
        self.auth_token = None
        self.login_ok = login_ok
        self.frame_kwargs = None

    def login(self, username, password):
        logging.info("official login payload username=%s password=%s", username, password)
        if self.login_ok:
            self.auth_token = "dummy-token"

    def get_current_frame(self, **kwargs):
        self.frame_kwargs = kwargs
        return None


def test_official_env_values_and_url_normalization(monkeypatch, tmp_path):
    monkeypatch.setenv("TEAM_NAME", "dummy-team")
    monkeypatch.setenv("PASSWORD", "dummy-password")
    monkeypatch.setenv("EVALUATION_SERVER_URL", "http://example.invalid///")
    monkeypatch.setenv("SESSION_NAME", "test-session")
    monkeypatch.setenv("OFFICIAL_INTERFACE_PATH", str(tmp_path))
    get_settings.cache_clear()
    try:
        loaded = get_settings()
        assert loaded.team_name == "dummy-team"
        assert loaded.password == "dummy-password"
        assert loaded.evaluation_server_url == "http://example.invalid/"
        assert loaded.official_session_name == "test-session"
    finally:
        get_settings.cache_clear()


def test_missing_password_produces_clear_startup_error(tmp_path):
    configured = settings(tmp_path, password="")
    with pytest.raises(RuntimeError, match="PASSWORD"):
        configured.validate_official_integration()


def test_login_failure_is_explicit_and_password_is_not_logged(tmp_path, caplog):
    client = FakeOfficialClient(login_ok=False)
    adapter = OfficialInterfaceAdapter(
        settings(tmp_path), bindings=bindings(), client=client
    )
    with caplog.at_level(logging.INFO), pytest.raises(OfficialAuthenticationError):
        adapter.authenticate()
    assert "dummy-password" not in caplog.text
    assert "dummy-token" not in caplog.text


def test_official_frame_retry_configuration_is_forwarded(tmp_path):
    client = FakeOfficialClient()
    configured = settings(
        tmp_path, competition_max_retries=4, competition_retry_initial_seconds=0.75
    )
    adapter = OfficialInterfaceAdapter(configured, bindings=bindings(), client=client)
    adapter.get_current_frame()
    assert client.frame_kwargs == {"retries": 4, "initial_wait_time": 0.75}


def test_official_frame_maps_to_aerosync_and_preserves_ids(tmp_path):
    request = map_official_frame(
        {"url": "/frames/17/", "image_url": "/media/f17.webp", "video_name": "v1"},
        {
            "health_status": "1",
            "translation_x": 1.5,
            "translation_y": 2.5,
            "translation_z": 3.5,
        },
        session_id="test-session",
        frame_index=16,
        local_image_path=tmp_path / "f17.webp",
    )
    assert request.url == "/frames/17/"
    assert request.session == "test-session"
    assert request.frame_index == 16
    assert request.gps_health_status == 1
    assert request.translation_x == 1.5


def test_null_translation_maps_without_fabricated_coordinates(tmp_path):
    request = map_official_frame(
        {"url": "/frames/1/", "image_url": "/media/f1.webp", "video_name": "v1"},
        None,
        session_id="test-session",
        frame_index=0,
        local_image_path=tmp_path / "f1.webp",
    )
    assert request.gps_health_status is None
    assert request.translation_x is None
    assert request.translation_y is None
    assert request.translation_z is None


def test_unexpected_health_status_is_rejected(tmp_path):
    with pytest.raises(OfficialFrameMappingError):
        map_official_frame(
            {"url": "/frames/1/", "image_url": "/f.webp", "video_name": "v"},
            {"health_status": "unknown"},
            session_id="test-session",
            frame_index=0,
            local_image_path=tmp_path / "f.webp",
        )


def test_unexpected_progress_json_is_rejected():
    with pytest.raises(OfficialServerUnavailable, match="eksik"):
        validate_official_progress({"unexpected": True})


def test_reference_mapping_preserves_range_and_official_url():
    catalog = map_official_references(
        [
            {
                "url": "/reference/abc/",
                "session": "/session/1/",
                "image_url": "/media/ref.png",
                "frame_start": 10,
                "frame_end": 20,
                "order": 0,
                "video_name": "video-1",
            }
        ],
        {"/reference/abc/": b"\x89PNG\r\n\x1a\ncontent"},
    )
    item = catalog.references[0]
    assert item.object_id == 1
    assert item.active_from_frame == 10
    assert item.active_until_frame == 20
    assert catalog.official_url_for(1) == "/reference/abc/"


def test_reference_mapping_supports_official_image_url_window():
    catalog = map_official_references(
        [
            {
                "url": "/reference/abc/",
                "image_url": "/media/ref.png",
                "frame_start": "/frames/10/",
                "frame_end": "/frames/20/",
                "frame_start_image_url": "/media/frame-010.webp",
                "frame_end_image_url": "/media/frame-020.webp",
                "order": 0,
                "video_name": "video-1",
            }
        ],
        {"/reference/abc/": b"content"},
    )
    assert catalog.requires_dynamic_activation is False
    assert catalog.references_for_frame(14, "/media/frame-015.webp") == catalog.references
    assert catalog.references_for_frame(30, "/media/frame-030.webp") == []


def test_aerosync_result_maps_multiple_detections_and_empty_translation():
    response = CompetitionResponse(
        id=1,
        user="",
        frame="/frames/1/",
        detected_objects=[
            DetectedObject(
                cls=ObjectClass.TASIT,
                landing_status=LandingStatus.NOT_APPLICABLE,
                motion_status=MotionStatus.MOVING,
                top_left_x=1,
                top_left_y=2,
                bottom_right_x=10,
                bottom_right_y=20,
            ),
            DetectedObject(
                cls=ObjectClass.UAP,
                landing_status=LandingStatus.SUITABLE,
                motion_status=MotionStatus.UNKNOWN,
                top_left_x=30,
                top_left_y=40,
                bottom_right_x=60,
                bottom_right_y=80,
            ),
        ],
        detected_translations=[],
        detected_undefined_objects=[],
    )
    prediction = map_aerosync_result(
        response,
        official_frame={"image_url": "/f.webp", "video_name": "v"},
        official_translation=None,
        catalog=map_official_references([], {}),
        bindings=bindings(),
    )
    payload = prediction.create_payload("http://example.invalid/")
    assert len(payload["detected_objects"]) == 2
    assert payload["detected_objects"][0]["cls"].endswith("classes/1/")
    assert payload["detected_objects"][1]["landing_status"] == "1"
    assert payload["detected_translations"] == []


def test_translation_and_reference_result_mapping():
    catalog = map_official_references(
        [{
            "url": "/reference/1/",
            "image_url": "/r.png",
            "frame_start": 0,
            "frame_end": 2,
            "order": 0,
            "video_name": "video-1",
        }],
        {"/reference/1/": b"content"},
    )
    response = CompetitionResponse(
        id=1,
        user="",
        frame="/frames/1/",
        detected_objects=[],
        detected_translations=[DetectedTranslation(translation_x=1, translation_y=2, translation_z=3)],
        detected_undefined_objects=[
            MatchedReferenceObject(
                object_id=1,
                top_left_x=4,
                top_left_y=5,
                bottom_right_x=14,
                bottom_right_y=15,
            )
        ],
    )
    prediction = map_aerosync_result(
        response,
        official_frame={"image_url": "/f.webp", "video_name": "v"},
        official_translation=None,
        catalog=catalog,
        bindings=bindings(),
    )
    payload = prediction.create_payload("http://example.invalid/")
    assert payload["detected_translations"][0]["translation_x"] == "1.0"
    assert payload["reference_predictions"][0]["reference"] == "/reference/1/"


class FakeAdapter:
    def __init__(self, tmp_path, configured, *, frames=None, responses=None, progress=None):
        self.settings = configured
        self.bindings = bindings()
        self.tmp_path = tmp_path
        self.frames = list(frames or [])
        self.responses = list(responses or [201])
        self.progress_values = list(
            progress
            or [
                {"frame_index": 0, "total_frames": len(self.frames), "completed": False, "session_name": "test-session"},
                {"frame_index": len(self.frames), "total_frames": len(self.frames), "completed": True, "session_name": "test-session"},
            ]
        )
        self.auth_calls = 0
        self.frame_calls = 0
        self.send_calls = 0

    def authenticate(self):
        self.auth_calls += 1

    def get_progress(self):
        if len(self.progress_values) > 1:
            return self.progress_values.pop(0)
        return self.progress_values[0]

    def get_reference_objects(self):
        return []

    def get_current_frame(self):
        self.frame_calls += 1
        return self.frames.pop(0) if self.frames else None

    def get_current_translation(self):
        return None

    def download_media(self, image_url, session_name, category):
        return self.tmp_path / Path(image_url).name

    def send_prediction(self, prediction):
        self.send_calls += 1
        status = self.responses.pop(0) if self.responses else 500
        return SimpleNamespace(status_code=status)


def frame(number):
    return {
        "url": f"/frames/{number}/",
        "image_url": f"/media/frame-{number}.webp",
        "video_name": "video-1",
    }


@pytest.mark.asyncio
async def test_result_retry_does_not_fetch_next_frame(tmp_path):
    configured = settings(tmp_path, competition_max_retries=2)
    adapter = FakeAdapter(tmp_path, configured, frames=[frame(1)], responses=[500, 201])
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    await runner.initialize()
    assert await runner.process_next_frame(0) is True
    assert adapter.send_calls == 2
    assert adapter.frame_calls == 1


@pytest.mark.asyncio
async def test_rejected_result_keeps_pending_frame(tmp_path):
    configured = settings(tmp_path, competition_max_retries=2)
    adapter = FakeAdapter(tmp_path, configured, frames=[frame(1)], responses=[500, 500])
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    await runner.initialize()
    with pytest.raises(PredictionRejected):
        await runner.process_next_frame(0)
    assert runner.pending is not None
    assert adapter.frame_calls == 1


@pytest.mark.asyncio
async def test_successful_submission_allows_next_frame(tmp_path):
    configured = settings(tmp_path)
    adapter = FakeAdapter(tmp_path, configured, frames=[frame(1), frame(2)], responses=[201, 201])
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    await runner.initialize()
    await runner.process_next_frame(0)
    await runner.process_next_frame(1)
    assert adapter.frame_calls == 2
    assert adapter.send_calls == 2


@pytest.mark.asyncio
async def test_duplicate_frame_is_not_submitted_twice(tmp_path):
    configured = settings(tmp_path)
    adapter = FakeAdapter(tmp_path, configured, frames=[frame(1), frame(1)], responses=[201])
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    await runner.initialize()
    await runner.process_next_frame(0)
    with pytest.raises(DuplicateFrameError):
        await runner.process_next_frame(1)
    assert adapter.send_calls == 1


@pytest.mark.asyncio
async def test_token_expiry_reauthenticates_and_retries_same_result(tmp_path):
    configured = settings(tmp_path)
    adapter = FakeAdapter(tmp_path, configured, frames=[frame(1)], responses=[401, 201])
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    await runner.initialize()
    await runner.process_next_frame(0)
    assert adapter.auth_calls == 2
    assert adapter.frame_calls == 1


@pytest.mark.asyncio
async def test_session_state_is_reset_on_initialize(tmp_path):
    configured = settings(tmp_path)
    services = build_services(configured)
    async with services.localization_sessions.locked("test-session") as state:
        state.frame_count = 9
    adapter = FakeAdapter(tmp_path, configured, frames=[])
    runner = OfficialCompetitionRunner(configured, adapter, services)
    await runner.initialize()
    assert "test-session" not in services.localization_sessions._states
    assert await services.matching.list_references("test-session") == (None, [])


@pytest.mark.asyncio
async def test_session_name_mismatch_stops_without_fetching_frame(tmp_path):
    configured = settings(tmp_path)
    adapter = FakeAdapter(
        tmp_path,
        configured,
        progress=[{"frame_index": 0, "total_frames": 1, "completed": False, "session_name": "another-session"}],
    )
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    with pytest.raises(OfficialSessionUnavailable):
        await runner.initialize()
    assert adapter.frame_calls == 0


@pytest.mark.asyncio
async def test_completed_session_ends_without_frame_request(tmp_path):
    configured = settings(tmp_path)
    adapter = FakeAdapter(
        tmp_path,
        configured,
        progress=[{"frame_index": 4, "total_frames": 4, "completed": True, "session_name": "test-session"}],
    )
    runner = OfficialCompetitionRunner(configured, adapter, build_services(configured))
    await runner.run()
    assert adapter.frame_calls == 0
