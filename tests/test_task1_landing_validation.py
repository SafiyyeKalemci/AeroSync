from __future__ import annotations

import ast
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app.core.config import get_settings
from app.schemas import LandingStatus, ObjectClass
from app.services.detection.landing_analyzer import LandingAnalyzer
from scripts.validate_task1_landing import (
    LandingCall,
    ValidationOptions,
    landing_diagnostics,
    policy_from_settings,
    run_real_validation,
    run_synthetic_validation,
)


def _settings():
    return replace(get_settings(), detection_motion_enabled=False, detection_landing_enabled=True)


def _analyze(raw, clipped=None, obstacles=None, obstacle_classes=()):
    settings = _settings()
    policy = policy_from_settings(settings)
    clipped = raw if clipped is None else clipped
    obstacles = [] if obstacles is None else obstacles
    result = LandingAnalyzer(policy).analyze(
        raw_bbox=raw,
        clipped_bbox=clipped,
        frame_width=100,
        frame_height=100,
        obstacles=obstacles,
    )
    call = LandingCall(tuple(raw), tuple(clipped), 100, 100, tuple(obstacles), result)
    return result, landing_diagnostics(call, policy, obstacle_classes)


def test_clear_uap_is_suitable():
    result, diagnostic = _analyze((10, 10, 60, 60))
    assert result is LandingStatus.SUITABLE
    assert diagnostic["decision_reason"] == "clear_landing_area"


def test_vehicle_obstacle_is_unsuitable():
    result, diagnostic = _analyze(
        (10, 10, 60, 60),
        obstacles=[(20, 20, 35, 35)],
        obstacle_classes=(ObjectClass.TASIT,),
    )
    assert result is LandingStatus.UNSUITABLE
    assert diagnostic["decision_reason"] == "occupied_by_vehicle"
    assert diagnostic["obstacles"][0]["occupied_by_production_policy"] is True


def test_person_obstacle_is_unsuitable():
    result, diagnostic = _analyze(
        (10, 10, 60, 60),
        obstacles=[(50, 50, 65, 65)],
        obstacle_classes=(ObjectClass.INSAN,),
    )
    assert result is LandingStatus.UNSUITABLE
    assert diagnostic["decision_reason"] == "occupied_by_person"
    assert diagnostic["obstacles"][0]["obstacle_center_inside"] is True


def test_frame_edge_beyond_tolerance_is_unsuitable():
    result, diagnostic = _analyze((-3, 10, 40, 50), (0, 10, 40, 50))
    assert result is LandingStatus.UNSUITABLE
    assert diagnostic["frame_edge"] is True
    assert diagnostic["decision_reason"] == "outside_frame_tolerance"


def test_too_small_is_not_applicable():
    result, diagnostic = _analyze((10, 10, 14, 14))
    assert result is LandingStatus.NOT_APPLICABLE
    assert diagnostic["min_area_check"]["passes"] is False
    assert diagnostic["decision_reason"] == "bbox_too_small"


def test_synthetic_covers_non_applicable_classes_and_uap_uai_ignore_policy():
    code, report = run_synthetic_validation(
        _settings(), ValidationOptions(synthetic=True), emit=lambda _: None
    )
    assert code == 0
    by_name = {item["scenario"]: item for item in report["scenarios"]}
    assert by_name["vehicle_not_applicable"]["actual"] == "not_applicable"
    assert by_name["person_not_applicable"]["actual"] == "not_applicable"
    assert by_name["uap_ignores_uai"]["actual"] == "suitable"
    assert by_name["uai_ignores_uap"]["actual"] == "suitable"
    assert report["summary"]["passed"] == report["summary"]["total"]


def test_synthetic_visualization_and_json_are_created(tmp_path):
    visualization = tmp_path / "synthetic.jpg"
    output = tmp_path / "synthetic.json"
    code, _ = run_synthetic_validation(
        _settings(),
        ValidationOptions(synthetic=True, save_visualization=visualization, json_output=output),
        emit=lambda _: None,
    )
    assert code == 0
    assert visualization.is_file() and cv2.imread(str(visualization)) is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["prediction_submission"] == "DISABLED"
    assert payload["scenarios"]


class FakeRuntime:
    def __init__(self, _settings):
        self.model_path = Path("fake-best.pt")
        self.confidence = 0.25
        self.iou = 0.45
        self._load_attempted = True
        self._model = SimpleNamespace(names={0: "tasit", 1: "insan", 2: "uap", 3: "uai"})

    def predict(self, _image):
        boxes = [
            _box(0, (5, 5, 15, 15)),
            _box(1, (75, 75, 85, 85)),
            _box(2, (20, 20, 60, 60)),
            _box(3, (0, 0, 20, 20)),
        ]
        return [SimpleNamespace(boxes=boxes)]


class UapUaiOnlyRuntime(FakeRuntime):
    def predict(self, _image):
        return [
            SimpleNamespace(
                boxes=[
                    _box(2, (20, 20, 70, 70)),
                    _box(3, (30, 30, 80, 80)),
                ]
            )
        ]


def _box(class_id, bbox):
    return SimpleNamespace(
        cls=np.asarray([class_id], dtype=np.float32),
        conf=np.asarray([0.9], dtype=np.float32),
        xyxy=np.asarray([bbox], dtype=np.float32),
    )


def test_real_mode_uses_production_service_and_writes_diagnostics(tmp_path):
    image = tmp_path / "frame.jpg"
    visualization = tmp_path / "result.jpg"
    output = tmp_path / "result.json"
    assert cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    code, report = asyncio.run(
        run_real_validation(
            _settings(),
            ValidationOptions(image=image, save_visualization=visualization, json_output=output),
            runtime_factory=FakeRuntime,
            emit=lambda _: None,
        )
    )
    assert code == 0
    by_class = {item["class"]: item for item in report["detections"]}
    assert by_class["tasit"]["landing_status"] == "not_applicable"
    assert by_class["insan"]["landing_status"] == "not_applicable"
    assert by_class["uap"]["landing_status"] == "suitable"
    assert by_class["uap"]["landing_diagnostics"]["decision_reason"] == "clear_landing_area"
    assert by_class["uai"]["landing_status"] == "unsuitable"
    assert by_class["uai"]["landing_diagnostics"]["decision_reason"] == "occupied_by_vehicle"
    assert visualization.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model_info"]["load_status"] == "OK"
    assert payload["frame_metadata"]["width"] == 100
    assert payload["prediction_submission"] == "DISABLED"


def test_production_service_does_not_count_uap_uai_as_each_others_obstacle(tmp_path):
    image = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    code, report = asyncio.run(
        run_real_validation(
            _settings(),
            ValidationOptions(image=image),
            runtime_factory=UapUaiOnlyRuntime,
            emit=lambda _: None,
        )
    )
    assert code == 0
    assert [item["landing_status"] for item in report["detections"]] == ["suitable", "suitable"]
    assert all(item["landing_diagnostics"]["obstacle_count"] == 0 for item in report["detections"])


def test_validator_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "validate_task1_landing.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"requests", "httpx", "competition"} & imported_roots)
    for token in (
        "send_prediction",
        "prediction/",
        "competition.runner",
        "requests.post",
        "httpx.post",
        ".post(",
    ):
        assert token not in source.casefold()
