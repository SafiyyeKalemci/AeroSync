from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import LandingStatus, MotionStatus
from app.services.detection.service import YoloDetectionService
from scripts.validate_task1_detection import (
    EXIT_MODEL,
    EXIT_OK,
    LoadedImage,
    ValidationOptions,
    run_validation,
)


class FakeBox:
    def __init__(self, class_id, confidence, bbox):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [bbox]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, names=None):
        self.names = names or {0: "tasit", 1: "insan", 2: "uap", 3: "uai"}


class FakeRuntime:
    def __init__(self, results, *, names=None, loaded=True):
        self.model_path = Path("best.pt")
        self.confidence = 0.25
        self.iou = 0.45
        self._model = FakeModel(names) if loaded else None
        self._load_attempted = True
        self._results = results
        self.calls = 0

    def predict(self, image):
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index] if self._model is not None else []


class FakeMotionAnalyzer:
    def to_grayscale(self, image):
        return np.asarray(image)[:, :, 0]

    def is_frozen(self, previous, current):
        return False

    def compute_flow(self, previous, current, exclusions):
        return object()

    def classify_vehicle(self, field, bbox):
        return MotionStatus.MOVING


class FakeLandingAnalyzer:
    def analyze(self, **kwargs):
        return LandingStatus.SUITABLE


def settings(tmp_path, **changes):
    model = tmp_path / "best.pt"
    model.write_bytes(b"test-double")
    values = {
        "detection_enabled": True,
        "detection_model_path": model,
        "detection_motion_enabled": True,
        "detection_motion_warmup_frames": 0,
        "detection_landing_enabled": True,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def loaded(path: Path, color=10):
    image = np.full((80, 100, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    content = encoded.tobytes()
    return LoadedImage(
        path=path.resolve(),
        content=content,
        image=image,
        metadata={
            "path": str(path.resolve()), "format": "png", "width": 100,
            "height": 80, "channels": 3, "sha256_short": "abc123",
        },
    )


def image_loader(path):
    return loaded(path, 10 if "first" in path.stem or "single" in path.stem else 20)


def production_service(settings_value, runtime, reader):
    return YoloDetectionService(
        settings_value,
        runtime=runtime,
        image_reader=reader,
        motion_analyzer=FakeMotionAnalyzer(),
        landing_analyzer=FakeLandingAnalyzer(),
    )


def runtime_with_multiple_detections():
    boxes = [
        FakeBox(0, 0.91, [10, 10, 40, 40]),
        FakeBox(1, 0.82, [45, 10, 60, 50]),
        FakeBox(2, 0.73, [65, 10, 90, 40]),
        FakeBox(3, 0.64, [10, 45, 40, 75]),
    ]
    return FakeRuntime([[FakeResult(boxes)], [FakeResult(boxes)]])


@pytest.mark.asyncio
async def test_single_frame_uses_production_service_and_reports_all_detections(tmp_path):
    runtime = runtime_with_multiple_detections()
    code, report = await run_validation(
        settings(tmp_path),
        ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: runtime,
        service_factory=production_service,
        image_loader=image_loader,
        emit=lambda _: None,
    )
    assert code == EXIT_OK
    assert runtime.calls == 1
    detections = report["frames"][0]["detections"]
    assert len(detections) == 4
    assert [item["class"] for item in detections] == ["tasit", "insan", "uap", "uai"]
    assert report["frames"][0]["detected_class_ids"] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_first_frame_unknown_second_frame_vehicle_motion(tmp_path):
    runtime = runtime_with_multiple_detections()
    code, report = await run_validation(
        settings(tmp_path),
        ValidationOptions(image=Path("first.png"), next_image=Path("second.png")),
        runtime_factory=lambda _: runtime,
        service_factory=production_service,
        image_loader=image_loader,
        emit=lambda _: None,
    )
    assert code == EXIT_OK
    first = report["frames"][0]["detections"]
    second = report["frames"][1]["detections"]
    assert first[0]["motion_status"] == "unknown"
    assert second[0]["motion_status"] == "moving"
    assert all(item["motion_status"] == "unknown" for item in second[1:])
    assert report["frames"][0]["session_id"] == report["frames"][1]["session_id"]
    assert report["frames"][0]["video_name"] == report["frames"][1]["video_name"]
    assert [frame["frame_index"] for frame in report["frames"]] == [0, 1]


@pytest.mark.asyncio
async def test_landing_results_preserve_production_class_policy(tmp_path):
    code, report = await run_validation(
        settings(tmp_path),
        ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: runtime_with_multiple_detections(),
        service_factory=production_service,
        image_loader=image_loader,
        emit=lambda _: None,
    )
    by_class = {item["class"]: item for item in report["frames"][0]["detections"]}
    assert code == EXIT_OK
    assert by_class["tasit"]["landing_status"] == "not_applicable"
    assert by_class["insan"]["landing_status"] == "not_applicable"
    assert by_class["uap"]["landing_status"] == "suitable"
    assert by_class["uai"]["landing_status"] == "suitable"


@pytest.mark.asyncio
async def test_model_class_mapping_report_and_warning(tmp_path):
    messages = []
    runtime = runtime_with_multiple_detections()
    _, report = await run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: runtime, service_factory=production_service,
        image_loader=image_loader, emit=messages.append,
    )
    assert report["model_info"]["matches_expected_mapping"] is True
    assert report["model_info"]["production_class_mapping"] == {
        "0": "tasit", "1": "insan", "2": "uap", "3": "uai"
    }
    mismatched = FakeRuntime(
        runtime._results,
        names={0: "person", 1: "vehicle", 2: "uap", 3: "uai"},
    )
    warning_messages = []
    _, warning_report = await run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: mismatched, service_factory=production_service,
        image_loader=image_loader, emit=warning_messages.append,
    )
    assert warning_report["model_info"]["matches_expected_mapping"] is False
    assert "Class mapping compatibility: WARNING" in warning_messages


@pytest.mark.asyncio
async def test_json_output_contains_model_frames_bbox_motion_and_landing(tmp_path):
    target = tmp_path / "task1.json"
    code, _ = await run_validation(
        settings(tmp_path),
        ValidationOptions(image=Path("single.png"), json_output=target),
        runtime_factory=lambda _: runtime_with_multiple_detections(),
        service_factory=production_service,
        image_loader=image_loader,
        emit=lambda _: None,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    detection = payload["frames"][0]["detections"][0]
    assert code == EXIT_OK
    assert payload["model_info"]["load_status"] == "OK"
    assert {"bbox", "confidence", "motion_status", "landing_status"} <= detection.keys()
    assert payload["prediction_submission"] == "DISABLED"


@pytest.mark.asyncio
async def test_visualization_single_and_two_frame_output_paths(tmp_path):
    single = tmp_path / "single-result.png"
    await run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png"), save_visualization=single),
        runtime_factory=lambda _: runtime_with_multiple_detections(),
        service_factory=production_service, image_loader=image_loader, emit=lambda _: None,
    )
    assert single.is_file() and single.stat().st_size > 0
    base = tmp_path / "pair.png"
    _, report = await run_validation(
        settings(tmp_path),
        ValidationOptions(
            image=Path("first.png"), next_image=Path("second.png"),
            save_visualization=base,
        ),
        runtime_factory=lambda _: runtime_with_multiple_detections(),
        service_factory=production_service, image_loader=image_loader, emit=lambda _: None,
    )
    assert [Path(path).name for path in report["visualizations"]] == [
        "pair_frame1.png", "pair_frame2.png"
    ]
    assert all(Path(path).is_file() for path in report["visualizations"])


@pytest.mark.asyncio
async def test_model_load_failure_is_explicit_and_never_fabricates_detection(tmp_path):
    runtime = FakeRuntime([[]], loaded=False)
    code, report = await run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: runtime, service_factory=production_service,
        image_loader=image_loader, emit=lambda _: None,
    )
    assert code == EXIT_MODEL
    assert report["model_info"]["load_status"] == "FAIL"
    assert report["frames"][0]["detections"] == []


def test_validation_source_has_no_prediction_runner_or_network_post():
    source = Path("scripts/validate_task1_detection.py").read_text(encoding="utf-8")
    forbidden = (
        "send_prediction", "prediction/", "competition.runner", "requests.post", "httpx.post"
    )
    assert not any(token in source for token in forbidden)
