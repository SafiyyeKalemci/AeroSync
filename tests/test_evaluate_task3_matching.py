from __future__ import annotations

import ast
import asyncio
import csv
import hashlib
import json
import threading
import time
import unicodedata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from app.core.config import get_settings
from app.schemas import MatchedReferenceObject
from app.services.matching.descriptor_types import DenseDescriptorSet, DescriptorMetrics
from app.services.matching.local_features import LocalRefinementDiagnostics
from app.services.matching.local_matcher import LocalPipelineResult
from scripts.evaluate_task3_matching import (
    EvaluationOptions,
    confusion_metrics,
    evaluate_match,
    ground_truth_resolution,
    load_ground_truth,
    load_references,
    run_evaluation,
)


def _settings(**changes):
    values = {
        "matching_enabled": True,
        "matching_dinov2_enabled": True,
        "matching_geometry_method": "dinov2",
        "matching_aliked_model_path": None,
        "matching_lightglue_model_path": None,
        "matching_coarse_min_similarity": 0.4,
        "matching_coarse_min_correspondences": 4,
        "matching_coarse_max_correspondences": 128,
        "matching_coarse_topk_per_reference": 1,
        "matching_coarse_chunk_size": 128,
        "matching_coarse_spatial_dedup_radius_px": 0.0,
        "matching_homography_min_inliers": 4,
        "matching_homography_min_inlier_ratio": 0.5,
        "matching_homography_max_rms_reprojection_error": 2.0,
        "matching_geometry_min_projected_area_px": 16.0,
        "matching_geometry_max_frame_area_ratio": 0.9,
        "matching_geometry_min_visible_ratio": 0.25,
        "matching_geometry_min_edge_length_px": 2.0,
        "matching_geometry_max_aspect_ratio": 10.0,
        "matching_geometry_max_perspective_distortion": 8.0,
        "matching_bbox_min_width_px": 2.0,
        "matching_bbox_min_height_px": 2.0,
        "matching_bbox_min_area_px": 16.0,
        "matching_bbox_max_frame_area_ratio": 0.9,
        "matching_min_confidence": 0.2,
        "matching_coarse_timeout_seconds": 2.0,
        "matching_reference_timeout_seconds": 2.0,
        "matching_dinov2_timeout_seconds": 2.0,
        "matching_preload_models": False,
        "matching_warmup_enabled": False,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def _descriptor(tensor, grid_width, grid_height, image_width, image_height, source_hash):
    return DenseDescriptorSet(
        descriptors=tensor,
        grid_width=grid_width,
        grid_height=grid_height,
        descriptor_dim=int(tensor.shape[1]),
        image_width=image_width,
        image_height=image_height,
        resized_width=grid_width * 14,
        resized_height=grid_height * 14,
        patch_size=14,
        scale_x=(grid_width * 14) / image_width,
        scale_y=(grid_height * 14) / image_height,
        device="cpu",
        dtype="float32",
        source_hash=source_hash,
    )


def _positive_descriptors(reference_hash="ref", frame_hash="positive"):
    frame_tensor = torch.eye(64, dtype=torch.float32)
    central = [row * 8 + column for row in range(2, 6) for column in range(2, 6)]
    reference_tensor = frame_tensor[central].clone()
    reference = _descriptor(reference_tensor, 4, 4, 56, 56, reference_hash)
    frame = _descriptor(frame_tensor, 8, 8, 112, 112, frame_hash)
    return reference, frame


class FakeRuntime:
    def __init__(self, descriptors):
        self.descriptors = descriptors
        self.model_hash = "fake-model-hash"
        self.device = "cpu"
        self.is_loaded = True
        self.inference_lock = threading.RLock()
        self.calls = []

    def extract(self, _image, source_hash):
        self.calls.append(source_hash)
        descriptor = self.descriptors[source_hash]
        metrics = DescriptorMetrics(0.001, 0.002, descriptor.shape[0], descriptor.descriptor_dim, descriptor.nbytes)
        return descriptor, metrics


def _write_image(path, shape, value):
    image = np.full(shape, value, np.uint8)
    cv2.circle(image, (shape[1] // 2, shape[0] // 2), min(shape[:2]) // 4, (255 - value,) * 3, -1)
    assert cv2.imwrite(str(path), image)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def evaluation_dataset(tmp_path):
    references = tmp_path / "references"
    frames = tmp_path / "frames"
    references.mkdir()
    frames.mkdir()
    ref_path = references / "object_001_tight.png"
    pos_path = frames / "frame_001.png"
    neg_path = frames / "frame_002.png"
    ignore_path = frames / "frame_003.png"
    ref_hash = _write_image(ref_path, (56, 56, 3), 30)
    pos_hash = _write_image(pos_path, (112, 112, 3), 60)
    neg_hash = _write_image(neg_path, (112, 112, 3), 90)
    ignore_hash = _write_image(ignore_path, (112, 112, 3), 120)
    reference_descriptor, positive_descriptor = _positive_descriptors(ref_hash, pos_hash)
    negative_tensor = -torch.eye(64, dtype=torch.float32)
    descriptors = {
        ref_hash: reference_descriptor,
        pos_hash: positive_descriptor,
        neg_hash: _descriptor(negative_tensor, 8, 8, 112, 112, neg_hash),
        ignore_hash: _descriptor(negative_tensor.clone(), 8, 8, 112, 112, ignore_hash),
    }
    ground_truth = tmp_path / "ground_truth.csv"
    ground_truth.write_text(
        "reference_id,frame_name,expected_match\n"
        "object_001_tight,frame_001.png,1\n"
        "object_001_tight,frame_002.png,0\n"
        "object_001_tight,frame_003.png,IGNORE\n",
        encoding="utf-8",
    )
    return references, frames, ground_truth, FakeRuntime(descriptors)


def test_reference_and_frame_loading_extracts_id_metadata(evaluation_dataset):
    references_dir, frames_dir, _, _ = evaluation_dataset
    loaded = load_references([references_dir / "object_001_tight.png"])
    assert loaded[0].reference_id == "object_001_tight"
    assert loaded[0].object_id == 1
    assert loaded[0].crop_type == "tight"
    assert loaded[0].asset.width == 56 and loaded[0].asset.height == 56
    assert len(list(frames_dir.iterdir())) == 3


def test_positive_negative_cache_metrics_csv_and_visualization(evaluation_dataset, tmp_path):
    references, frames, ground_truth, runtime = evaluation_dataset
    output = tmp_path / "output"
    report = asyncio.run(
        run_evaluation(
            _settings(),
            EvaluationOptions(output, references_dir=references, frames_dir=frames, ground_truth_csv=ground_truth, save_visualizations=True),
            runtime_factory=lambda _settings: runtime,
            emit=lambda _message: None,
        )
    )
    rows = report["detailed"]
    assert rows[0]["accepted"] is True
    assert rows[0]["matched_reference_object_produced"] is True
    assert rows[0]["homography_valid"] is True
    assert rows[0]["bbox_valid"] is True
    assert rows[1]["accepted"] is False
    assert rows[1]["matched_reference_object_produced"] is False
    assert rows[1]["rejection_reason"] in {"no_mutual_match", "below_minimum_correspondences"}
    assert report["cache"]["first_request"] == "MISS"
    assert report["cache"]["second_request"] == "HIT"
    assert report["cache"]["reference_forward_repeated"] is False
    assert report["cache"]["frame_descriptor_persistently_cached"] is False
    assert report["cache"]["frame_forward_repeated"] is True
    assert len(report["threshold_sensitivity"]) == 15
    assert report["reference_summary"][0]["crop_type"] == "tight"
    assert report["ground_truth_metrics"] == {
        "true_positive": 1, "false_positive": 0, "true_negative": 1, "false_negative": 0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0,
        "evaluated_count": 2, "ignored_count": 1,
    }
    expected_files = {
        "task3_matching_detailed.csv", "task3_matching_summary.csv",
        "task3_matching_performance.csv", "task3_matching_threshold_sensitivity.csv",
        "task3_matching_report.json", "task3_matching_confusion.csv",
    }
    assert expected_files <= {path.name for path in output.iterdir()}
    assert len(list((output / "visualizations").glob("*.jpg"))) == 3
    assert (output / "task3_service_consistency_detailed.csv").is_file()
    assert (output / "task3_service_consistency_summary.txt").is_file()
    assert (output / "task3_service_config_snapshot.json").is_file()
    payload = json.loads((output / "task3_matching_report.json").read_text(encoding="utf-8"))
    assert payload["prediction_submission"] == "DISABLED"
    with (output / "task3_matching_detailed.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_single_reference_frame_quick_mode(evaluation_dataset, tmp_path):
    references, frames, _, runtime = evaluation_dataset
    report = asyncio.run(
        run_evaluation(
            _settings(),
            EvaluationOptions(
                tmp_path / "single-output",
                reference=references / "object_001_tight.png",
                frame=frames / "frame_001.png",
            ),
            runtime_factory=lambda _settings: runtime,
            emit=lambda _message: None,
        )
    )
    assert report["mode"] == "single"
    assert len(report["detailed"]) == 1
    assert report["detailed"][0]["matched_reference_object_produced"] is True


def test_homography_bbox_and_confidence_rejection_paths_use_production_components():
    reference, frame = _positive_descriptors()
    homography, _, _ = evaluate_match(_settings(matching_homography_min_inliers=20), reference, frame)
    assert homography["homography_valid"] is False
    assert homography["failure_reason"] == "low_inliers"
    bbox, _, _ = evaluate_match(_settings(matching_bbox_min_width_px=60.0), reference, frame)
    assert bbox["homography_valid"] is True
    assert bbox["failure_reason"] == "bbox_invalid"
    confidence, _, _ = evaluate_match(_settings(matching_min_confidence=1.0), reference, frame)
    assert confidence["homography_valid"] is True
    assert confidence["failure_reason"] == "confidence_below_threshold"


def test_ground_truth_ignore_handling_and_metrics(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "reference_id,frame_name,expected_match\nr,f1.jpg,1\nr,f2.jpg,0\nr,f3.jpg,IGNORE\n",
        encoding="utf-8",
    )
    assert load_ground_truth(path)[("r", "f3.jpg")] == "IGNORE"
    rows = [
        {"expected_match": "1", "matched_reference_object_produced": True},
        {"expected_match": "0", "matched_reference_object_produced": False},
        {"expected_match": "IGNORE", "matched_reference_object_produced": True},
    ]
    metrics = confusion_metrics(rows)
    assert metrics["accuracy"] == 1.0
    assert metrics["ignored_count"] == 1


class AcceptedLocalPipeline:
    def __init__(self, delay=0.0):
        self.delay = delay

    def match_reference(self, *, method, object_id, **_kwargs):
        if self.delay:
            time.sleep(self.delay)
        return LocalPipelineResult(
            MatchedReferenceObject(
                object_id=object_id,
                top_left_x=10,
                top_left_y=11,
                bottom_right_x=40,
                bottom_right_y=41,
                confidence=0.9,
            ),
            LocalRefinementDiagnostics(
                method=method,
                accepted=True,
                reason="accepted",
                coarse_correspondence_count=16,
                coarse_similarity=0.9,
                coarse_coverage=0.8,
                aliked_reference_keypoints=128,
                aliked_frame_keypoints=256,
                lightglue_match_count=32,
                lightglue_mean_score=0.9,
                local_homography_inliers=30,
                local_homography_inlier_ratio=0.9375,
                local_homography_rms=0.5,
                total_seconds=self.delay,
            ),
        )


def test_hybrid_diagnostic_and_production_service_use_same_result_without_stale_registry_config(
    evaluation_dataset, tmp_path
):
    references, frames, _, runtime = evaluation_dataset
    report = asyncio.run(
        run_evaluation(
            _settings(
                matching_geometry_method="hybrid",
                matching_local_refinement_timeout_sec=5.0,
            ),
            EvaluationOptions(
                tmp_path / "hybrid-consistency",
                reference=references / "object_001_tight.png",
                frame=frames / "frame_001.png",
            ),
            runtime_factory=lambda _settings: runtime,
            local_pipeline_factory=lambda _settings: AcceptedLocalPipeline(0.05),
            emit=lambda _message: None,
        )
    )
    row = report["detailed"][0]
    assert row["hybrid_accepted"] is True
    assert row["matched_reference_object_produced"] is True
    assert row["bbox"] == pytest.approx([10, 11, 40, 41])
    assert row["root_cause"] == "consistent"
    assert row["service_match_outcome"] == "matched"
    assert row["service_timeout_stage"] == "local_refinement"
    assert row["service_timeout_limit_sec"] == 5.0
    assert report["service_consistency"]["evaluator_service_config_equal"] is True
    assert report["service_consistency"]["registry_stale_config"] is False
    assert report["config_snapshot"]["evaluator"]["MATCHING_GEOMETRY_METHOD"] == "hybrid"
    assert report["config_snapshot"]["production_service"]["MATCHING_GEOMETRY_METHOD"] == "hybrid"
    assert report["config_snapshot"]["production_service"]["MATCHING_LOCAL_REFINEMENT_TIMEOUT_SEC"] == 5.0


def test_local_timeout_does_not_fallback_and_is_reported_as_root_cause(
    evaluation_dataset, tmp_path
):
    references, frames, _, runtime = evaluation_dataset
    report = asyncio.run(
        run_evaluation(
            _settings(
                matching_geometry_method="hybrid",
                matching_local_refinement_timeout_sec=0.01,
                matching_coarse_timeout_seconds=10.0,
                matching_reference_timeout_seconds=20.0,
            ),
            EvaluationOptions(
                tmp_path / "timeout-consistency",
                reference=references / "object_001_tight.png",
                frame=frames / "frame_001.png",
            ),
            runtime_factory=lambda _settings: runtime,
            local_pipeline_factory=lambda _settings: AcceptedLocalPipeline(0.05),
            emit=lambda _message: None,
        )
    )
    row = report["detailed"][0]
    assert row["hybrid_accepted"] is True
    assert row["matched_reference_object_produced"] is False
    assert row["local_refinement_timeout_triggered"] is True
    assert row["service_match_outcome"] == "timeout"
    assert row["service_timeout_stage"] == "local_refinement"
    assert row["service_timeout_limit_sec"] == pytest.approx(0.01)
    assert row["service_match_elapsed_sec"] >= 0.01
    assert row["fallback_to_dinov2_after_timeout"] is False
    assert row["root_cause"] == "local_refinement_timeout"


def test_ground_truth_unicode_filename_resolution_uses_nfc(tmp_path):
    composed_reference = "Ekran görüntüsü 220100"
    decomposed_reference = unicodedata.normalize("NFD", composed_reference)
    composed_frame = "ş.jpg"
    decomposed_frame = unicodedata.normalize("NFD", composed_frame)
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "reference_id,frame_name,expected_match\n"
        f"{decomposed_reference},{decomposed_frame},1\n",
        encoding="utf-8",
    )
    resolved = ground_truth_resolution(
        path,
        [SimpleNamespace(reference_id=composed_reference)],
        [SimpleNamespace(path=Path(composed_frame))],
    )
    assert resolved[0]["gt_reference_resolved"] is True
    assert resolved[0]["gt_frame_resolved"] is True
    assert resolved[0]["reference_normalization_changed"] is True
    assert resolved[0]["frame_normalization_changed"] is True
    assert load_ground_truth(path)[(composed_reference, composed_frame)] == "1"


class SelectiveLocalPipeline(AcceptedLocalPipeline):
    def __init__(self, accepted_hash):
        super().__init__()
        self.accepted_hash = accepted_hash

    def match_reference(self, *, method, object_id, frame_hash, **kwargs):
        if frame_hash == self.accepted_hash:
            return super().match_reference(method=method, object_id=object_id, **kwargs)
        return LocalPipelineResult(
            None,
            LocalRefinementDiagnostics(
                method=method,
                accepted=False,
                reason="low_local_matches",
                aliked_reference_keypoints=128,
                aliked_frame_keypoints=128,
                lightglue_match_count=3,
                total_seconds=0.001,
            ),
        )


def test_hybrid_production_rejects_negative_pairs_without_false_positive(
    evaluation_dataset, tmp_path
):
    references, frames, ground_truth, runtime = evaluation_dataset
    positive_hash = hashlib.sha256((frames / "frame_001.png").read_bytes()).hexdigest()
    report = asyncio.run(
        run_evaluation(
            _settings(
                matching_geometry_method="hybrid",
                matching_local_refinement_timeout_sec=5.0,
            ),
            EvaluationOptions(
                tmp_path / "hybrid-positive-negative",
                references_dir=references,
                frames_dir=frames,
                ground_truth_csv=ground_truth,
            ),
            runtime_factory=lambda _settings: runtime,
            local_pipeline_factory=lambda _settings: SelectiveLocalPipeline(positive_hash),
            emit=lambda _message: None,
        )
    )
    evaluated = [row for row in report["detailed"] if row["expected_match"] != "IGNORE"]
    assert [row["matched_reference_object_produced"] for row in evaluated] == [True, False]
    assert report["ground_truth_metrics"]["false_positive"] == 0


def test_evaluator_has_no_prediction_server_or_post_calls():
    source_path = Path(__file__).parents[1] / "scripts" / "evaluate_task3_matching.py"
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
        "send_prediction", "prediction/", "competition.runner", "requests.post",
        "httpx.post", ".post(",
    ):
        assert token not in source.casefold()
