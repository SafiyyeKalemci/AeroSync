from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.config import get_settings
from competition import preflight_check as module
from competition.preflight_check import (
    EXIT_CONFIG,
    EXIT_DEPENDENCY,
    EXIT_OFFICIAL,
    EXIT_STRICT_WARNING,
    Check,
    Options,
    Report,
    Status,
    run_preflight,
    write_json_report,
)


def settings(tmp_path: Path, **changes):
    interface = tmp_path / "official"
    for relative in module.OFFICIAL_FILES:
        path = interface / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# official test double\n", encoding="utf-8")
    values = {
        "team_name": "safe-team",
        "password": "safe-password",
        "evaluation_server_url": "http://example.invalid/",
        "official_session_name": "test-session",
        "official_interface_path": interface,
        "official_media_dir": tmp_path / "work" / "official_media",
        "detection_enabled": False,
        "localization_enabled": False,
        "matching_enabled": False,
        "matching_allow_cpu_fallback": True,
    }
    values.update(changes)
    return replace(get_settings(), **values)


def make_root(tmp_path: Path, *, env: bool = True) -> Path:
    for directory in ("app", "competition", "models/detection", "models/localization", "models/matching", "scripts"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    for name in ("requirements.txt", "pyproject.toml", ".env.example"):
        (tmp_path / name).write_text("\n", encoding="utf-8")
    if env:
        (tmp_path / ".env").write_text("PASSWORD=not-printed\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(
        ".env\n.venv/\nwork/\nlogs/\n*.pt\n*.pth\n*.ckpt\n__pycache__/\n",
        encoding="utf-8",
    )
    return tmp_path


def ok_command(command, cwd, env):
    return subprocess.CompletedProcess(command, 0, "collection OK\n", "")


def test_local_successful_orchestration_never_calls_online(monkeypatch, tmp_path):
    root = make_root(tmp_path)
    called = []

    def add_ok(report, *_args, **_kwargs):
        report.add("Configuration", "stub", Status.OK, "ok")

    for name in (
        "_check_configuration", "_check_python", "_check_dependencies", "_check_official",
        "_check_tasks", "_check_storage", "_check_gpu", "_check_application",
        "_check_security", "_check_tests",
    ):
        monkeypatch.setattr(module, name, add_ok)
    report = run_preflight(
        root, settings(tmp_path), Options(), command_runner=ok_command,
        online_runner=lambda *_args: called.append(True) or 0,
    )
    assert report.exit_code(False) == 0
    assert report.readiness() == "READY"
    assert called == []
    assert any(c.section == "Online Connectivity" and c.status is Status.SKIPPED for c in report.checks)


def test_missing_env_is_config_failure(tmp_path):
    report = Report(make_root(tmp_path, env=False))
    module._check_configuration(report, settings(tmp_path))
    check = next(c for c in report.checks if c.name == ".env")
    assert check.status is Status.FAIL
    assert report.exit_code(False) == EXIT_CONFIG


def test_configuration_does_not_expose_sensitive_values(tmp_path):
    configured = settings(tmp_path)
    report = Report(make_root(tmp_path))
    module._check_configuration(report, configured)
    rendered = "\n".join(check.message for check in report.checks)
    assert configured.password not in rendered
    assert configured.evaluation_server_url not in rendered
    assert configured.official_session_name not in rendered
    assert str(configured.official_interface_path) not in rendered


def test_missing_required_dependency(tmp_path):
    report = Report(tmp_path)
    module._check_dependencies(report, lambda name: name != "numpy")
    assert next(c for c in report.checks if c.name == "numpy").message == "required missing"
    assert report.exit_code(False) == EXIT_DEPENDENCY


def test_missing_optional_dependency_is_warning(tmp_path):
    report = Report(tmp_path)
    module._check_dependencies(report, lambda name: name not in module.OPTIONAL_PACKAGES)
    assert next(c for c in report.checks if c.name == "torch").status is Status.WARNING
    assert report.exit_code(False) == 0


def test_wrong_official_interface_path(tmp_path):
    report = Report(tmp_path)
    module._check_official(report, settings(tmp_path, official_interface_path=tmp_path / "missing"))
    assert report.exit_code(False) == EXIT_OFFICIAL


def test_missing_best_pt_is_warning_when_detection_disabled(tmp_path):
    report = Report(tmp_path)
    module._artifact_check(report, "Task 1 Detection", "best.pt", tmp_path / "best.pt", False)
    assert report.checks[0].status is Status.WARNING


def test_configured_camera_intrinsics_are_ready_without_external_file(tmp_path):
    root = make_root(tmp_path)
    report = Report(root)
    module._check_tasks(report, settings(tmp_path), skip_models=True)
    check = next(c for c in report.checks if c.name == "camera calibration")
    assert check.status is Status.OK
    assert "configured camera intrinsics" in check.message


def test_task1_service_and_builtin_mapping_report_current_implementation(tmp_path):
    root = make_root(tmp_path)
    report = Report(root)
    module._check_tasks(report, settings(tmp_path, detection_enabled=False), skip_models=True)
    service = next(
        c for c in report.checks if c.section == "Task 1 Detection" and c.name == "service"
    )
    mapping = next(c for c in report.checks if c.name == "class mapping")
    assert service.status is Status.OK
    assert service.message == "implemented; disabled by configuration"
    assert mapping.status is Status.OK
    assert "built-in central ObjectClass mapping" in mapping.message


def test_task1_enabled_with_model_reports_implemented_and_enabled(tmp_path):
    root = make_root(tmp_path)
    model = tmp_path / "models" / "detection" / "best.pt"
    model.write_bytes(b"test-model")
    report = Report(root)
    module._check_tasks(
        report,
        settings(tmp_path, detection_enabled=True, detection_model_path=model),
        skip_models=False,
    )
    service = next(
        c for c in report.checks if c.section == "Task 1 Detection" and c.name == "service"
    )
    assert service.status is Status.OK
    assert service.message == "implemented and enabled"


def test_task2_reports_affinevo_and_current_expected_frame_setting(tmp_path):
    report = Report(make_root(tmp_path))
    module._check_tasks(report, settings(tmp_path, localization_enabled=False), skip_models=True)
    service = next(
        c for c in report.checks if c.section == "Task 2 Localization" and c.name == "service"
    )
    expected = next(c for c in report.checks if c.name == "calibration expected max frame")
    gps_calibration = next(c for c in report.checks if c.name == "GPS scale calibration")
    assert service.message == "implemented; disabled by configuration"
    assert "LOCALIZATION_CALIBRATION_EXPECTED_MAX_FRAME=450" in expected.message
    assert "inactive while localization service is disabled" in gps_calibration.message
    assert all("LOCALIZATION_CALIBRATION_FRAMES" not in c.message for c in report.checks)


def test_task3_reports_current_coarse_matching_pipeline(tmp_path):
    report = Report(make_root(tmp_path))
    module._check_tasks(
        report,
        settings(
            tmp_path,
            matching_enabled=True,
            matching_dinov2_enabled=True,
            matching_geometry_method="dinov2",
        ),
        skip_models=True,
    )
    service = next(
        c for c in report.checks if c.section == "Task 3 Matching" and c.name == "service"
    )
    pipeline = next(c for c in report.checks if c.name == "RGB-RGB production pipeline")
    assert service.status is Status.OK
    assert "USAC_MAGSAC homography" in service.message
    assert "MatchedReferenceObject" in service.message
    assert "stage 2" not in pipeline.message


def test_task3_hybrid_requires_both_local_artifacts(tmp_path):
    report = Report(make_root(tmp_path))
    module._check_tasks(
        report,
        settings(
            tmp_path,
            matching_enabled=True,
            matching_dinov2_enabled=True,
            matching_geometry_method="hybrid",
            matching_local_refinement_enabled=True,
            matching_aliked_model_path=tmp_path / "missing-aliked.ts",
            matching_lightglue_model_path=tmp_path / "missing-lightglue.ts",
        ),
        skip_models=False,
    )
    aliked = next(c for c in report.checks if c.name == "ALIKED")
    lightglue = next(c for c in report.checks if c.name == "LightGlue")
    service = next(c for c in report.checks if c.section == "Task 3 Matching" and c.name == "service")
    assert aliked.status is Status.FAIL
    assert lightglue.status is Status.FAIL
    assert "DINOv2 coarse gate -> ALIKED -> LightGlue" in service.message


def test_gpu_missing_reports_cpu_fallback(tmp_path):
    report = Report(tmp_path)
    module._check_gpu(report, settings(tmp_path), False, lambda _name: False)
    assert any(c.name == "CPU fallback" and c.status is Status.OK for c in report.checks)


def test_security_secret_detection_reports_only_file_and_line(tmp_path):
    root = make_root(tmp_path)
    source = root / "app" / "bad.py"
    source.write_text('PASSWORD = "do-not-repeat-this"\n', encoding="utf-8")
    report = Report(root)
    module._check_security(report, settings(tmp_path, team_name="", password=""))
    finding = next(c for c in report.checks if c.name == "hardcoded credential")
    assert finding.status is Status.FAIL
    assert "bad.py:1" in finding.message
    assert "do-not-repeat-this" not in finding.message


def test_json_output_contains_schema_but_no_credentials(tmp_path):
    report = Report(tmp_path)
    report.add("Configuration", "PASSWORD", Status.OK, "configured")
    target = tmp_path / "report.json"
    write_json_report(report, target, False)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert {"timestamp", "project_root", "python_version", "checks", "summary", "readiness", "exit_code"} <= payload.keys()
    assert "safe-password" not in target.read_text(encoding="utf-8")


def test_strict_mode_turns_warning_into_exit_one(tmp_path):
    report = Report(tmp_path, [Check("GPU", "CUDA", Status.WARNING, "unavailable")])
    assert report.exit_code(False) == 0
    assert report.exit_code(True) == EXIT_STRICT_WARNING


def test_fetch_frame_requires_online(capsys):
    assert module.main(["--fetch-frame"]) == EXIT_CONFIG
    assert "yalnızca --online" in capsys.readouterr().out


def test_online_mode_delegates_once_and_prediction_remains_zero(tmp_path):
    report = Report(tmp_path)
    calls = []

    def online_runner(_settings, fetch_frame, _verbose):
        calls.append(fetch_frame)
        return 0

    module._check_online(report, Options(online=True, fetch_frame=True), settings(tmp_path), online_runner)
    assert calls == [True]
    prediction = next(c for c in report.checks if c.name == "prediction submission")
    assert prediction.status is Status.SKIPPED
    assert "submitted=0" in prediction.message


def test_test_runner_marks_real_server_false(tmp_path):
    report = Report(tmp_path)
    module._check_tests(report, Options(run_tests=True), ok_command)
    assert report.checks[0].details["real_server"] is False
