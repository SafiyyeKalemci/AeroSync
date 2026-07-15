from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from app.core.config import get_settings
from app.services.matching.descriptor_types import DenseDescriptorSet, DescriptorMetrics
from scripts.validate_dinov2_artifacts import (
    EXIT_ARTIFACT,
    EXIT_IMAGE,
    EXIT_MODEL,
    EXIT_OK,
    ValidationOptions,
    inspect_artifacts,
    run_validation,
)


def settings(tmp_path, **changes):
    repo = tmp_path / "dinov2"
    repo.mkdir(exist_ok=True)
    (repo / "hubconf.py").write_text("def dinov2_vitb14():\n    return None\n", encoding="utf-8")
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"trusted-test-double")
    values = {
        "matching_dinov2_repo_path": repo,
        "matching_dinov2_weights_path": weights,
        "matching_dinov2_device": "cpu",
        "matching_coarse_min_correspondences": 4,
        "matching_homography_min_inliers": 4,
        "matching_homography_min_inlier_ratio": 0.5,
        "matching_geometry_max_frame_area_ratio": 1.0,
        "matching_bbox_max_frame_area_ratio": 1.0,
        "matching_geometry_min_projected_area_px": 16.0,
        "matching_bbox_min_area_px": 16.0,
        "matching_min_confidence": 0.2,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def dense(value=None):
    tensor = torch.eye(16, dtype=torch.float32) if value is None else value
    return DenseDescriptorSet(
        descriptors=tensor,
        grid_width=4,
        grid_height=4,
        descriptor_dim=int(tensor.shape[1]),
        image_width=56,
        image_height=56,
        resized_width=56,
        resized_height=56,
        patch_size=14,
        scale_x=1.0,
        scale_y=1.0,
        device="cpu",
        dtype="float32",
        source_hash="source",
    )


class FakeRuntime:
    device = "cpu"
    model_hash = "0123456789abcdef"

    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.calls = 0

    def extract(self, image, source_hash):
        self.calls += 1
        descriptor = self.outputs.get(source_hash, dense())
        metrics = DescriptorMetrics(0.001, 0.002, descriptor.shape[0], descriptor.descriptor_dim, descriptor.nbytes)
        return descriptor, metrics


class BrokenRuntime:
    @property
    def model_hash(self):
        raise RuntimeError("load failed")


def loader(path: Path):
    key = path.stem
    return np.zeros((56, 56, 3), dtype=np.uint8), {
        "path": str(path), "format": "png", "width": 56, "height": 56,
        "channels": 3, "decode": "OK", "sha256_short": key,
        "source_hash": key,
    }


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"matching_dinov2_repo_path": Path("missing-repo")}, "dinov2_repository_missing"),
        ({"remove_hubconf": True}, "hubconf_missing"),
        ({"matching_dinov2_weights_path": Path("missing.pth")}, "dinov2_weights_missing"),
        ({"invalid_extension": True}, "weight_extension_invalid"),
    ],
)
def test_invalid_artifacts_stop_before_runtime(tmp_path, changes, reason):
    configured = settings(tmp_path)
    if changes.pop("remove_hubconf", False):
        (configured.matching_dinov2_repo_path / "hubconf.py").unlink()
    elif changes.pop("invalid_extension", False):
        invalid = tmp_path / "model.bin"
        invalid.write_bytes(b"x")
        configured = replace(configured, matching_dinov2_weights_path=invalid)
    else:
        configured = replace(configured, **changes)
    calls = []
    code, report = run_validation(
        configured, ValidationOptions(), runtime_factory=lambda _: calls.append(1), emit=lambda _: None
    )
    assert code == EXIT_ARTIFACT
    assert report["failure_reason"] == reason
    assert calls == []


def test_artifact_metadata_contains_size_hash_model_and_policy(tmp_path):
    report = inspect_artifacts(settings(tmp_path))
    assert report["valid"] is True
    assert report["weights"]["size_bytes"] > 0
    assert len(report["weights"]["sha256_short"]) == 12
    assert report["model_name"] == "dinov2_vitb14"
    assert report["cpu_fallback_allowed"] is True


def test_model_load_failure_is_not_reported_as_success(tmp_path):
    code, report = run_validation(
        settings(tmp_path), ValidationOptions(), runtime_factory=lambda _: BrokenRuntime(),
        emit=lambda _: None,
    )
    assert code == EXIT_MODEL
    assert report["model"]["load"] == "FAIL"
    assert report["final_result"] == "FAIL"


def test_single_image_descriptor_report(tmp_path):
    runtime = FakeRuntime()
    code, report = run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: runtime, image_loader=loader, emit=lambda _: None,
    )
    metadata = report["descriptor_metadata"]["image"]
    assert code == EXIT_OK
    assert metadata["descriptor_count"] == 16
    assert metadata["descriptor_dimension"] == 16
    assert metadata["nan_or_inf"] is False
    assert metadata["l2_norm"]["mean"] == pytest.approx(1.0)


def test_nan_descriptor_is_rejected(tmp_path):
    invalid = torch.eye(16)
    invalid[0, 0] = float("nan")
    runtime = FakeRuntime({"single": dense(invalid)})
    code, report = run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png")),
        runtime_factory=lambda _: runtime, image_loader=loader, emit=lambda _: None,
    )
    assert code == EXIT_IMAGE
    assert report["failure_reason"] == "descriptor_non_finite"


def test_positive_synthetic_match_and_reference_cache_hit(tmp_path):
    runtime = FakeRuntime()
    code, report = run_validation(
        settings(tmp_path),
        ValidationOptions(reference=Path("reference.png"), frame=Path("frame.png")),
        runtime_factory=lambda _: runtime, image_loader=loader, emit=lambda _: None,
    )
    assert code == EXIT_OK
    assert report["match_metrics"]["accepted"] is True
    assert report["match_metrics"]["confidence"] != 0.5
    assert report["reference_cache"]["first_request"] == "MISS"
    assert report["reference_cache"]["second_request"] == "HIT"
    assert report["reference_cache"]["reference_forward_repeated"] is False
    assert runtime.calls == 2  # one reference, one frame


def test_negative_synthetic_match_reports_rejection(tmp_path):
    repeated = torch.zeros((16, 16), dtype=torch.float32)
    repeated[:, 0] = 1.0
    runtime = FakeRuntime({"frame": dense(repeated)})
    code, report = run_validation(
        settings(tmp_path),
        ValidationOptions(reference=Path("reference.png"), frame=Path("frame.png")),
        runtime_factory=lambda _: runtime, image_loader=loader, emit=lambda _: None,
    )
    assert code == EXIT_OK
    assert report["final_result"] == "REJECTED"
    assert report["match_metrics"]["accepted"] is False
    assert report["failure_reason"] is not None


def test_benchmark_reports_statistics_and_does_not_cache_frames(tmp_path):
    runtime = FakeRuntime()
    code, report = run_validation(
        settings(tmp_path),
        ValidationOptions(
            reference=Path("reference.png"), frame=Path("frame.png"), benchmark_runs=3
        ),
        runtime_factory=lambda _: runtime, image_loader=loader, emit=lambda _: None,
    )
    benchmark = report["benchmark_metrics"]
    assert code == EXIT_OK
    assert benchmark["runs"] == 3
    assert benchmark["warmup_runs"] == 1
    for values in benchmark["timings"].values():
        assert set(values) == {"minimum", "maximum", "mean", "p50", "p95"}
    assert runtime.calls == 6  # ref + initial frame + warmup frame + three measured frames


def test_json_output_contains_safe_sections_and_no_credentials(tmp_path):
    target = tmp_path / "report.json"
    code, report = run_validation(
        settings(tmp_path), ValidationOptions(image=Path("single.png"), json_output=target),
        runtime_factory=lambda _: FakeRuntime(), image_loader=loader, emit=lambda _: None,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert code == EXIT_OK
    assert payload["prediction_submission"] == "DISABLED"
    assert {"timestamp", "artifact_metadata", "environment", "image_metadata",
            "descriptor_metadata", "match_metrics", "benchmark_metrics", "final_result",
            "failure_reason"} <= payload.keys()
    serialized = json.dumps(payload).lower()
    assert "authorization" not in serialized
    assert get_settings().password not in serialized if get_settings().password else True


def test_visualization_is_created_only_when_requested(tmp_path):
    runtime = FakeRuntime()
    absent = tmp_path / "absent.png"
    run_validation(
        settings(tmp_path),
        ValidationOptions(reference=Path("reference.png"), frame=Path("frame.png")),
        runtime_factory=lambda _: runtime, image_loader=loader, emit=lambda _: None,
    )
    assert not absent.exists()
    target = tmp_path / "visualization.png"
    run_validation(
        settings(tmp_path),
        ValidationOptions(
            reference=Path("reference.png"), frame=Path("frame.png"),
            save_visualization=target,
        ),
        runtime_factory=lambda _: FakeRuntime(), image_loader=loader, emit=lambda _: None,
    )
    assert target.is_file() and target.stat().st_size > 0


def test_validation_source_has_no_prediction_or_network_client_call():
    source = Path("scripts/validate_dinov2_artifacts.py").read_text(encoding="utf-8")
    forbidden = ("send_prediction(", "prediction/", "httpx.", "requests.", "torch.hub.load")
    assert not any(token in source for token in forbidden)
