from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.services.matching.aliked_runtime import AlikedRuntime
from app.services.matching.lightglue_runtime import LightGlueRuntime
from scripts import export_task3_local_artifacts as exporter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIKED_ARTIFACT = PROJECT_ROOT / "models" / "matching" / "aliked.pt"
LIGHTGLUE_ARTIFACT = PROJECT_ROOT / "models" / "matching" / "lightglue_aliked.pt"
POSITIVE_REFERENCE = PROJECT_ROOT / "work" / "task3_smoke_inputs" / "references" / "object_001_tight.jpg"
POSITIVE_FRAME = PROJECT_ROOT / "work" / "task3_smoke_inputs" / "frames" / "frame_001.jpg"
NEGATIVE_FRAME = PROJECT_ROOT / "external" / "LightGlue" / "assets" / "DSC_0410.JPG"


def _read_image(path: Path):
    content = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image, hashlib.sha256(content).hexdigest()


@pytest.fixture(scope="module")
def production_smoke():
    settings = replace(
        get_settings(),
        matching_aliked_model_path=ALIKED_ARTIFACT,
        matching_lightglue_model_path=LIGHTGLUE_ARTIFACT,
        matching_aliked_device="cpu",
        matching_lightglue_device="cpu",
    )
    aliked = AlikedRuntime(settings)
    lightglue = LightGlueRuntime(settings)
    extracted = []
    for path in (POSITIVE_REFERENCE, POSITIVE_FRAME, NEGATIVE_FRAME):
        image, source_hash = _read_image(path)
        features, _ = aliked.extract(image, source_hash)
        extracted.append(features)
    positive, _ = lightglue.match(extracted[0], extracted[1])
    negative, _ = lightglue.match(extracted[0], extracted[2])
    return aliked, lightglue, extracted, positive, negative


def test_export_defaults_match_production_preprocessing_contract():
    args = exporter.build_parser().parse_args([])
    assert args.canvas_size == get_settings().matching_dinov2_max_long_edge == 1120
    assert args.max_keypoints == 1024


def test_torchscript_artifacts_load_and_run_through_production_runtimes(production_smoke):
    aliked, lightglue, feature_sets, positive, _ = production_smoke
    assert ALIKED_ARTIFACT.is_file() and LIGHTGLUE_ARTIFACT.is_file()
    assert aliked.is_loaded and lightglue.is_loaded
    for features in feature_sets:
        assert features.keypoint_count >= 64
        assert np.isfinite(features.keypoints).all()
        assert np.isfinite(features.descriptors).all()
        assert np.isfinite(features.scores).all()
        assert (features.keypoints[:, 0] >= 0).all()
        assert (features.keypoints[:, 0] <= features.image_width).all()
        assert (features.keypoints[:, 1] >= 0).all()
        assert (features.keypoints[:, 1] <= features.image_height).all()
    assert positive.match_count >= 20
    assert len(np.unique(positive.reference_points_px, axis=0)) == positive.match_count
    assert len(np.unique(positive.frame_points_px, axis=0)) == positive.match_count


def test_positive_and_negative_smoke_are_separated(production_smoke):
    _, _, _, positive, negative = production_smoke
    assert positive.match_count >= 20
    assert positive.mean_score > 0.5
    assert negative.match_count < 20
    assert negative.mean_score < positive.mean_score


def test_runtime_and_export_tool_have_no_network_or_prediction_submission_calls():
    sources = "\n".join(
        inspect.getsource(module)
        for module in (
            __import__("app.services.matching.aliked_runtime", fromlist=["*"]),
            __import__("app.services.matching.lightglue_runtime", fromlist=["*"]),
            exporter,
        )
    )
    forbidden = (
        "send_prediction(",
        "requests.post(",
        "httpx.post(",
        "prediction/",
    )
    assert all(fragment not in sources for fragment in forbidden)

