from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.schemas import LandingStatus, MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection import DisabledDetectionService, YoloDetectionService
from app.services.detection.class_mapping import YoloClassMapper
from app.services.detection.yolo_runtime import YoloRuntime
from app.services.registry import build_services


def configured(tmp_path: Path, **changes):
    values = {
        "detection_enabled": True,
        "detection_model_path": tmp_path / "best.pt",
        "detection_confidence": 0.25,
        "detection_iou": 0.45,
        "detection_landing_enabled": False,
        "matching_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def frame() -> FrameContext:
    return FrameContext(
        frame_id="frame-1",
        image_url="frame.webp",
        video_name="video",
        session_id="session-1",
        gps_health_status=None,
        gps_x=None,
        gps_y=None,
        gps_z=None,
        frame_index=1,
    )


def box(class_id, confidence, coordinates):
    return SimpleNamespace(cls=[class_id], conf=[confidence], xyxy=[coordinates])


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class FakeImage:
    shape = (100, 200, 3)


async def image_reader(_source: str, _timeout: float) -> bytes:
    return b"encoded-image"


def test_class_ids_are_mapped_to_central_enum():
    mapper = YoloClassMapper()
    assert mapper.resolve(0) is ObjectClass.TASIT
    assert mapper.resolve(1.0) is ObjectClass.INSAN
    assert mapper.resolve(2) is ObjectClass.UAP
    assert mapper.resolve(3) is ObjectClass.UAI
    assert mapper.resolve(99) is None
    assert mapper.resolve(1.5) is None


def test_runtime_loads_model_only_on_first_inference_and_reuses_it(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights-test-double")
    model = FakeModel([])
    factory_calls = []

    def factory(path):
        factory_calls.append(path)
        return model

    runtime = YoloRuntime(model_path, 0.31, 0.52, model_factory=factory)
    assert factory_calls == []
    runtime.predict(FakeImage())
    runtime.predict(FakeImage())

    assert factory_calls == [str(model_path)]
    assert len(model.calls) == 2
    assert model.calls[0]["conf"] == 0.31
    assert model.calls[0]["iou"] == 0.52
    assert model.calls[0]["verbose"] is False


def test_missing_model_is_logged_once_and_returns_empty(tmp_path, caplog):
    factory_calls = []
    runtime = YoloRuntime(
        tmp_path / "missing.pt",
        0.25,
        0.45,
        model_factory=lambda path: factory_calls.append(path),
    )
    assert runtime.predict(FakeImage()) == []
    assert runtime.predict(FakeImage()) == []
    assert factory_calls == []
    assert sum(record.message == "detection_model_file_missing" for record in caplog.records) == 1


@pytest.mark.parametrize("confidence,iou", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_runtime_validates_thresholds(tmp_path, confidence, iou):
    with pytest.raises(ValueError):
        YoloRuntime(tmp_path / "best.pt", confidence, iou)


@pytest.mark.asyncio
async def test_service_returns_multiple_typed_detections_and_clips_boxes(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights-test-double")
    model = FakeModel(
        [
            SimpleNamespace(
                boxes=[
                    box(0, 0.90, [-10, -5, 50, 60]),
                    box(1, 0.80, [100, 20, 250, 120]),
                    box(2, 0.70, [10, 10, 30, 30]),
                    box(3, 0.60, [40, 40, 80, 90]),
                ]
            )
        ]
    )
    runtime = YoloRuntime(model_path, 0.25, 0.45, model_factory=lambda _path: model)
    service = YoloDetectionService(
        configured(tmp_path),
        runtime=runtime,
        image_reader=image_reader,
        image_decoder=lambda _content: FakeImage(),
    )

    detections = await service.process_frame(frame())

    assert [item.cls for item in detections] == [
        ObjectClass.TASIT,
        ObjectClass.INSAN,
        ObjectClass.UAP,
        ObjectClass.UAI,
    ]
    assert detections[0].top_left_x == 0
    assert detections[0].top_left_y == 0
    assert detections[1].bottom_right_x == 200
    assert detections[1].bottom_right_y == 100
    assert all(item.motion_status is MotionStatus.UNKNOWN for item in detections)
    assert detections[0].landing_status is LandingStatus.NOT_APPLICABLE
    assert detections[1].landing_status is LandingStatus.NOT_APPLICABLE
    assert detections[2].landing_status is LandingStatus.NOT_APPLICABLE
    assert detections[3].landing_status is LandingStatus.NOT_APPLICABLE
    assert [item.confidence for item in detections] == [0.9, 0.8, 0.7, 0.6]


@pytest.mark.asyncio
async def test_service_rejects_invalid_bbox_confidence_threshold_and_class(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"weights-test-double")
    model = FakeModel(
        [
            SimpleNamespace(
                boxes=[
                    box(0, 0.9, [20, 20, 10, 30]),
                    box(0, float("nan"), [1, 1, 20, 20]),
                    box(0, 1.1, [1, 1, 20, 20]),
                    box(0, 0.1, [1, 1, 20, 20]),
                    box(99, 0.9, [1, 1, 20, 20]),
                    box(1, 0.9, [1, 1, 20, 20]),
                ]
            )
        ]
    )
    service = YoloDetectionService(
        configured(tmp_path),
        runtime=YoloRuntime(model_path, 0.25, 0.45, model_factory=lambda _path: model),
        image_reader=image_reader,
        image_decoder=lambda _content: FakeImage(),
    )
    detections = await service.process_frame(frame())
    assert len(detections) == 1
    assert detections[0].cls is ObjectClass.INSAN


@pytest.mark.asyncio
async def test_service_returns_empty_when_image_decode_fails(tmp_path):
    service = YoloDetectionService(
        configured(tmp_path),
        image_reader=image_reader,
        image_decoder=lambda _content: (_ for _ in ()).throw(ValueError("bad image")),
    )
    assert await service.process_frame(frame()) == []


def test_registry_selects_real_or_disabled_detection_service(tmp_path):
    enabled = build_services(configured(tmp_path))
    disabled = build_services(configured(tmp_path, detection_enabled=False))
    assert isinstance(enabled.detection, YoloDetectionService)
    assert isinstance(disabled.detection, DisabledDetectionService)


def test_phase_one_landing_status_keeps_existing_api_wire_value():
    assert LandingStatus.NOT_APPLICABLE.value == "not_applicable"
