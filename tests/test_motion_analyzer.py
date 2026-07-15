from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import get_settings
from app.schemas import MotionStatus, ObjectClass
from app.services.common import FrameContext
from app.services.detection.motion_analyzer import MotionAnalyzer
from app.services.detection.service import YoloDetectionService
from app.services.detection.session_store import DetectionSessionStore


def settings(**changes):
    values = {
        "detection_enabled": True,
        "detection_motion_enabled": True,
        "detection_motion_threshold_px": 2.0,
        "detection_motion_min_valid_pixels": 9,
        "detection_motion_inner_crop_ratio": 0.1,
        "detection_motion_max_frame_gap": 1,
        "detection_motion_warmup_frames": 1,
        "detection_motion_flow_downscale": 1.0,
        "detection_motion_freeze_threshold": 0.0,
        "detection_motion_session_ttl_seconds": 60.0,
        "detection_motion_max_sessions": 8,
        "detection_landing_enabled": False,
        "matching_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def frame(frame_id="f1", index=1, session="s1", video="v1", source="image-1"):
    return FrameContext(
        frame_id=frame_id,
        image_url=source,
        video_name=video,
        session_id=session,
        gps_health_status=None,
        gps_x=None,
        gps_y=None,
        gps_z=None,
        frame_index=index,
    )


def yolo_box(class_id, coordinates, confidence=0.9):
    return SimpleNamespace(cls=[class_id], conf=[confidence], xyxy=[coordinates])


class Runtime:
    confidence = 0.25

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def predict(self, _image):
        self.calls += 1
        return [SimpleNamespace(boxes=self.boxes)]


class FlowFactory:
    def __init__(self, moving_boxes=()):
        self.moving_boxes = list(moving_boxes)
        self.calls = 0

    def __call__(self, previous, current):
        self.calls += 1
        flow = np.zeros((*current.shape, 2), dtype=np.float32)
        for x1, y1, x2, y2 in self.moving_boxes:
            flow[y1:y2, x1:x2, 0] = 5.0
        return flow


def analyzer(flow_factory, **changes):
    options = {
        "threshold_px": 2.0,
        "min_valid_pixels": 9,
        "inner_crop_ratio": 0.1,
        "flow_downscale": 1.0,
        "freeze_threshold": 0.0,
    }
    options.update(changes)
    return MotionAnalyzer(flow_calculator=flow_factory, **options)


def service(images, boxes, flow_factory, **setting_changes):
    async def reader(source, _timeout):
        value = images[source]
        if isinstance(value, Exception):
            raise value
        return source.encode()

    return YoloDetectionService(
        settings(**setting_changes),
        runtime=Runtime(boxes),
        image_reader=reader,
        image_decoder=lambda content: images[content.decode()].copy(),
        motion_analyzer=analyzer(flow_factory),
    )


@pytest.mark.asyncio
async def test_first_frame_vehicle_is_unknown():
    flow = FlowFactory()
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8)}, [yolo_box(0, [20, 20, 40, 40])], flow)
    result = await detection.process_frame(frame())
    assert result[0].motion_status is MotionStatus.UNKNOWN
    assert flow.calls == 0


@pytest.mark.asyncio
async def test_second_consecutive_frame_stationary():
    flow = FlowFactory()
    detection = service(
        {"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((100, 100, 3), np.uint8)},
        [yolo_box(0, [20, 20, 40, 40])], flow,
    )
    await detection.process_frame(frame())
    result = await detection.process_frame(frame("f2", 2, source="image-2"))
    assert result[0].motion_status is MotionStatus.STATIONARY
    assert flow.calls == 1


@pytest.mark.asyncio
async def test_second_consecutive_frame_moving():
    flow = FlowFactory([(20, 20, 40, 40)])
    detection = service(
        {"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((100, 100, 3), np.uint8)},
        [yolo_box(0, [20, 20, 40, 40])], flow,
    )
    await detection.process_frame(frame())
    result = await detection.process_frame(frame("f2", 2, source="image-2"))
    assert result[0].motion_status is MotionStatus.MOVING


@pytest.mark.asyncio
async def test_non_vehicle_classes_remain_unknown():
    flow = FlowFactory()
    boxes = [yolo_box(1, [5, 5, 15, 15]), yolo_box(2, [20, 20, 30, 30]), yolo_box(3, [40, 40, 50, 50])]
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((100, 100, 3), np.uint8)}, boxes, flow)
    await detection.process_frame(frame())
    result = await detection.process_frame(frame("f2", 2, source="image-2"))
    assert [item.cls for item in result] == [ObjectClass.INSAN, ObjectClass.UAP, ObjectClass.UAI]
    assert all(item.motion_status is MotionStatus.UNKNOWN for item in result)
    assert flow.calls == 0


@pytest.mark.asyncio
async def test_multiple_vehicles_share_one_flow_calculation():
    flow = FlowFactory([(10, 10, 30, 30), (50, 50, 75, 75)])
    boxes = [yolo_box(0, [10, 10, 30, 30]), yolo_box(0, [50, 50, 75, 75])]
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((100, 100, 3), np.uint8)}, boxes, flow)
    await detection.process_frame(frame())
    result = await detection.process_frame(frame("f2", 2, source="image-2"))
    assert [item.motion_status for item in result] == [MotionStatus.MOVING, MotionStatus.MOVING]
    assert flow.calls == 1


@pytest.mark.asyncio
async def test_sessions_are_isolated_and_resettable():
    flow = FlowFactory()
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((100, 100, 3), np.uint8)}, [yolo_box(0, [20, 20, 40, 40])], flow)
    first_a = await detection.process_frame(frame(session="a"))
    first_b = await detection.process_frame(frame(session="b"))
    await detection.reset_session("a")
    after_reset = await detection.process_frame(frame("f2", 2, session="a", source="image-2"))
    assert first_a[0].motion_status is MotionStatus.UNKNOWN
    assert first_b[0].motion_status is MotionStatus.UNKNOWN
    assert after_reset[0].motion_status is MotionStatus.UNKNOWN
    assert flow.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second",
    [
        frame("f2", 2, video="v2", source="image-2"),
        frame("f2", 1, source="image-2"),
        frame("f2", 5, source="image-2"),
    ],
)
async def test_video_change_out_of_order_and_large_gap_reset_baseline(second):
    flow = FlowFactory()
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((100, 100, 3), np.uint8)}, [yolo_box(0, [20, 20, 40, 40])], flow)
    await detection.process_frame(frame())
    result = await detection.process_frame(second)
    assert result[0].motion_status is MotionStatus.UNKNOWN
    assert flow.calls == 0


@pytest.mark.asyncio
async def test_shape_change_resets_baseline():
    flow = FlowFactory()
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8), "image-2": np.ones((80, 120, 3), np.uint8)}, [yolo_box(0, [20, 20, 40, 40])], flow)
    await detection.process_frame(frame())
    result = await detection.process_frame(frame("f2", 2, source="image-2"))
    assert result[0].motion_status is MotionStatus.UNKNOWN
    assert flow.calls == 0


@pytest.mark.asyncio
async def test_duplicate_frame_returns_cache_without_advancing_or_inference():
    flow = FlowFactory()
    detection = service({"image-1": np.zeros((100, 100, 3), np.uint8)}, [yolo_box(0, [20, 20, 40, 40])], flow)
    first = await detection.process_frame(frame())
    second = await detection.process_frame(frame())
    assert second == first
    assert detection._runtime.calls == 1
    assert flow.calls == 0


@pytest.mark.asyncio
async def test_corrupt_frame_does_not_replace_previous_state():
    flow = FlowFactory()
    store = DetectionSessionStore(ttl_seconds=60, max_sessions=4)
    images = {"image-1": np.zeros((100, 100, 3), np.uint8), "bad": ValueError("corrupt")}

    async def reader(source, _timeout):
        if isinstance(images[source], Exception):
            raise images[source]
        return source.encode()

    detection = YoloDetectionService(
        settings(), runtime=Runtime([yolo_box(0, [20, 20, 40, 40])]),
        image_reader=reader, image_decoder=lambda content: images[content.decode()].copy(),
        motion_analyzer=analyzer(flow), session_store=store,
    )
    await detection.process_frame(frame())
    assert await detection.process_frame(frame("bad", 2, source="bad")) == []
    state = store.get_or_create("s1")
    assert state.previous_frame_id == "f1"
    assert state.previous_frame_index == 1


@pytest.mark.asyncio
async def test_freeze_detection_returns_unknown_without_flow():
    flow = FlowFactory([(20, 20, 40, 40)])
    same = np.zeros((100, 100, 3), np.uint8)
    detection = service({"image-1": same, "image-2": same}, [yolo_box(0, [20, 20, 40, 40])], flow)
    await detection.process_frame(frame())
    result = await detection.process_frame(frame("f2", 2, source="image-2"))
    assert result[0].motion_status is MotionStatus.UNKNOWN
    assert flow.calls == 0


@pytest.mark.asyncio
async def test_warmup_frames_keep_motion_unknown_until_ready():
    flow = FlowFactory([(20, 20, 40, 40)])
    images = {
        "image-1": np.zeros((100, 100, 3), np.uint8),
        "image-2": np.ones((100, 100, 3), np.uint8),
        "image-3": np.full((100, 100, 3), 2, np.uint8),
    }
    detection = service(images, [yolo_box(0, [20, 20, 40, 40])], flow, detection_motion_warmup_frames=2)
    first = await detection.process_frame(frame())
    second = await detection.process_frame(frame("f2", 2, source="image-2"))
    third = await detection.process_frame(frame("f3", 3, source="image-3"))
    assert first[0].motion_status is MotionStatus.UNKNOWN
    assert second[0].motion_status is MotionStatus.UNKNOWN
    assert third[0].motion_status is MotionStatus.MOVING
    assert flow.calls == 2


def test_analyzer_rejects_small_bbox_and_insufficient_global_flow():
    flow = FlowFactory()
    motion = analyzer(flow, min_valid_pixels=20)
    previous = np.zeros((10, 10), np.uint8)
    current = np.ones((10, 10), np.uint8)
    field = motion.compute_flow(previous, current, [(0, 0, 10, 9)])
    assert field is None
    valid_field = analyzer(flow, min_valid_pixels=4).compute_flow(previous, current, [])
    assert analyzer(flow, min_valid_pixels=4).classify_vehicle(valid_field, (1, 1, 2, 2)) is MotionStatus.UNKNOWN


def test_downscale_scales_bbox_and_threshold_back_to_original_pixels():
    calls = []

    def downscaled_flow(previous, current):
        calls.append(current.shape)
        flow = np.zeros((*current.shape, 2), np.float32)
        flow[10:20, 10:20, 0] = 2.0  # 4 original pixels at 0.5 scale
        return flow

    motion = MotionAnalyzer(
        threshold_px=3.0, min_valid_pixels=4, inner_crop_ratio=0,
        flow_downscale=0.5, freeze_threshold=0, flow_calculator=downscaled_flow,
    )
    field = motion.compute_flow(np.zeros((100, 100), np.uint8), np.ones((100, 100), np.uint8), [(20, 20, 40, 40)])
    assert calls == [(50, 50)]
    assert motion.classify_vehicle(field, (20, 20, 40, 40)) is MotionStatus.MOVING


@pytest.mark.parametrize(
    "change",
    [
        {"detection_motion_threshold_px": -1},
        {"detection_motion_min_valid_pixels": 0},
        {"detection_motion_inner_crop_ratio": 0.5},
        {"detection_motion_max_frame_gap": 0},
        {"detection_motion_warmup_frames": -1},
        {"detection_motion_flow_downscale": 0},
        {"detection_motion_freeze_threshold": -1},
        {"detection_motion_session_ttl_seconds": 0},
        {"detection_motion_max_sessions": 0},
    ],
)
def test_motion_config_validation(change):
    with pytest.raises(ValueError):
        settings(**change).validate_detection_motion()


def test_session_store_capacity_and_ttl_cleanup():
    store = DetectionSessionStore(ttl_seconds=0.001, max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("c")
    assert len(store) == 2
    time.sleep(0.03)
    store.get_or_create("d")
    assert len(store) == 1
    store.reset_all()
    assert len(store) == 0
