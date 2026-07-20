from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlparse, urlunparse

from app.core.config import Settings, get_settings


EXIT_OK = 0
EXIT_STRICT_WARNING = 1
EXIT_CONFIG = 10
EXIT_DEPENDENCY = 20
EXIT_OFFICIAL = 30
EXIT_ARTIFACT = 40
EXIT_APPLICATION = 50
EXIT_SECURITY = 60
EXIT_ONLINE = 70
EXIT_TEST = 80

SECTIONS = (
    "Configuration",
    "Python Environment",
    "Dependencies",
    "Official Interface",
    "Task 1 Detection",
    "Task 2 Localization",
    "Task 3 Matching",
    "Storage",
    "GPU",
    "Application",
    "Security",
    "Tests",
    "Online Connectivity",
)
REQUIRED_PACKAGES = ("numpy", "cv2", "requests", "pydantic", "fastapi", "dotenv", "pytest")
OPTIONAL_PACKAGES = ("torch", "torchvision", "ultralytics", "kornia", "einops", "lightglue")
OFFICIAL_FILES = ("main.py", "src/connection_handler.py", "src/frame_predictions.py")
IGNORE_RULES = (".env", ".venv", "work", "logs", "*.pt", "*.pth", "*.ckpt", "__pycache__")


class Status(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class Check:
    section: str
    name: str
    status: Status
    message: str
    exit_code: int | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Options:
    online: bool = False
    fetch_frame: bool = False
    run_tests: bool = False
    verbose: bool = False
    json_output: Path | None = None
    strict: bool = False
    skip_gpu: bool = False
    skip_models: bool = False


@dataclass
class Report:
    project_root: Path
    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        section: str,
        name: str,
        status: Status,
        message: str,
        *,
        exit_code: int | None = None,
        **details: object,
    ) -> None:
        self.checks.append(Check(section, name, status, message, exit_code, details))

    def counts(self) -> dict[str, int]:
        return {status.value: sum(c.status is status for c in self.checks) for status in Status}

    def exit_code(self, strict: bool) -> int:
        failures = [c.exit_code for c in self.checks if c.status is Status.FAIL and c.exit_code]
        if failures:
            priority = (EXIT_CONFIG, EXIT_DEPENDENCY, EXIT_OFFICIAL, EXIT_ARTIFACT,
                        EXIT_APPLICATION, EXIT_SECURITY, EXIT_ONLINE, EXIT_TEST)
            return next(code for code in priority if code in failures)
        if strict and any(c.status is Status.WARNING for c in self.checks):
            return EXIT_STRICT_WARNING
        return EXIT_OK

    def readiness(self) -> str:
        if any(c.status is Status.FAIL for c in self.checks):
            return "NOT READY"
        if any(c.status is Status.WARNING for c in self.checks):
            return "READY WITH WARNINGS"
        return "READY"

    def payload(self, strict: bool) -> dict[str, object]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "python_version": platform.python_version(),
            "checks": [
                {**asdict(check), "status": check.status.value}
                for check in self.checks
            ],
            "summary": {"total": len(self.checks), **self.counts()},
            "readiness": self.readiness(),
            "exit_code": self.exit_code(strict),
        }


ImportProbe = Callable[[str], bool]
CommandRunner = Callable[[list[str], Path, Mapping[str, str]], subprocess.CompletedProcess[str]]
OnlineRunner = Callable[[Settings, bool, bool], int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AeroSync yarışma öncesi güvenli kontrol")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--fetch-frame", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _run_command(command: list[str], cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=dict(env), text=True, capture_output=True, timeout=180)


def _safe_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return "invalid"
    return urlunparse((parsed.scheme.lower(), parsed.netloc, parsed.path.rstrip("/") + "/", "", "", ""))


def _mask_team(value: str) -> str:
    if not value:
        return "missing"
    return value[0] + "***" + (value[-1] if len(value) > 1 else "")


def _file_info(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "extension": path.suffix.lower(),
        "readable": os.access(path, os.R_OK),
        "sha256_short": digest.hexdigest()[:12],
    }


def _check_configuration(report: Report, settings: Settings) -> None:
    root = report.project_root
    required = ("app", "competition", "requirements.txt", "pyproject.toml", ".env", ".env.example")
    for item in required:
        path = root / item
        report.add("Configuration", item, Status.OK if path.exists() else Status.FAIL,
                   "present" if path.exists() else "missing", exit_code=None if path.exists() else EXIT_CONFIG)
    values = {
        "TEAM_NAME": settings.team_name,
        "PASSWORD": settings.password,
        "EVALUATION_SERVER_URL": settings.evaluation_server_url,
        "SESSION_NAME": settings.official_session_name,
        "OFFICIAL_INTERFACE_PATH": settings.official_interface_path,
    }
    for name, value in values.items():
        configured = bool(value)
        display = (
            _mask_team(str(value))
            if name == "TEAM_NAME" and value
            else ("configured" if value else "missing")
        )
        report.add("Configuration", name, Status.OK if configured else Status.FAIL, display,
                   exit_code=None if configured else EXIT_CONFIG)


def _check_python(report: Report, command_runner: CommandRunner) -> None:
    executable = Path(sys.executable)
    report.add("Python Environment", "interpreter", Status.OK, str(executable))
    supported = sys.version_info >= (3, 11) and sys.version_info < (3, 13)
    report.add("Python Environment", "version", Status.OK if supported else Status.WARNING,
               platform.python_version())
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or ".venv" in executable.parts
    report.add("Python Environment", "virtual environment", Status.OK if in_venv else Status.WARNING,
               ".venv active" if in_venv else ".venv not active")
    try:
        result = command_runner([sys.executable, "-m", "pip", "--version"], report.project_root, os.environ)
        ok = result.returncode == 0
    except Exception:
        ok = False
    report.add("Python Environment", "pip", Status.OK if ok else Status.FAIL,
               "available" if ok else "unavailable", exit_code=None if ok else EXIT_DEPENDENCY)


def _check_dependencies(report: Report, import_probe: ImportProbe) -> None:
    for package in REQUIRED_PACKAGES:
        ok = import_probe(package)
        report.add("Dependencies", package, Status.OK if ok else Status.FAIL,
                   "importable" if ok else "required missing", exit_code=None if ok else EXIT_DEPENDENCY)
    for package in OPTIONAL_PACKAGES:
        ok = import_probe(package)
        report.add("Dependencies", package, Status.OK if ok else Status.WARNING,
                   "importable" if ok else "optional missing")


def _check_official(report: Report, settings: Settings) -> None:
    path = settings.official_interface_path
    if path is None or not path.is_dir():
        report.add("Official Interface", "interface path", Status.FAIL, "missing or invalid", exit_code=EXIT_OFFICIAL)
        return
    report.add("Official Interface", "interface path", Status.OK, "directory exists", path=str(path))
    for relative in OFFICIAL_FILES:
        present = (path / relative).is_file()
        report.add("Official Interface", relative, Status.OK if present else Status.FAIL,
                   "present" if present else "missing", exit_code=None if present else EXIT_OFFICIAL)
    for module in ("official_interface_adapter", "frame_mapper", "result_mapper", "reference_mapper", "runner"):
        try:
            __import__(f"competition.{module}")
            ok = True
        except Exception:
            ok = False
        report.add("Official Interface", f"import competition.{module}", Status.OK if ok else Status.FAIL,
                   "importable" if ok else "import failed", exit_code=None if ok else EXIT_OFFICIAL)


def _artifact_check(report: Report, section: str, name: str, path: Path | None, enabled: bool) -> None:
    if path and path.is_file():
        report.add(section, name, Status.OK, "artifact ready", **_file_info(path))
    else:
        report.add(section, name, Status.FAIL if enabled else Status.WARNING,
                   "artifact missing", exit_code=EXIT_ARTIFACT if enabled else None,
                   path=str(path) if path else "not configured")


def _probe_dinov2_hubconf(repo: Path) -> tuple[bool, str]:
    hubconf = repo / "hubconf.py"
    if not hubconf.is_file():
        return False, "hubconf.py missing"
    module_name = f"_aerosync_dinov2_preflight_{hash(repo)}"
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo))
        spec = importlib.util.spec_from_file_location(module_name, hubconf)
        if spec is None or spec.loader is None:
            return False, "hubconf import spec unavailable"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "local hubconf importable without model inference"
    except Exception as exc:
        return False, f"hubconf import failed: {type(exc).__name__}"
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = original_path


def _check_dinov2(report: Report, settings: Settings, skip_models: bool) -> None:
    section = "Task 3 Matching"
    required = settings.matching_enabled
    repo = settings.matching_dinov2_repo_path
    weights = settings.matching_dinov2_weights_path
    if skip_models:
        report.add(section, "DINOv2 local repository", Status.SKIPPED, "--skip-models")
        report.add(section, "DINOv2 weights", Status.SKIPPED, "--skip-models")
    else:
        repo_ok = bool(repo and repo.is_dir())
        report.add(
            section, "DINOv2 local repository",
            Status.OK if repo_ok else (Status.FAIL if required else Status.WARNING),
            "local directory ready" if repo_ok else "local repository missing",
            exit_code=EXIT_ARTIFACT if required and not repo_ok else None,
            path=str(repo) if repo else "not configured",
        )
        if repo_ok:
            hub_ok, message = _probe_dinov2_hubconf(repo)
            report.add(
                section, "DINOv2 hubconf",
                Status.OK if hub_ok else (Status.FAIL if required else Status.WARNING),
                message,
                exit_code=EXIT_ARTIFACT if required and not hub_ok else None,
            )
        valid_suffix = bool(weights and weights.suffix.lower() in {".pt", ".pth", ".ckpt"})
        weights_ok = bool(weights and weights.is_file() and valid_suffix)
        details = _file_info(weights) if weights_ok and weights is not None else {
            "path": str(weights) if weights else "not configured"
        }
        report.add(
            section, "DINOv2 weights",
            Status.OK if weights_ok else (Status.FAIL if required else Status.WARNING),
            "artifact ready" if weights_ok else "required local weight missing or extension invalid",
            exit_code=EXIT_ARTIFACT if required and not weights_ok else None,
            **details,
        )
    for dependency in ("torch", "cv2", "numpy"):
        available = _importable(dependency)
        report.add(
            section, f"DINOv2 dependency {dependency}",
            Status.OK if available else (Status.FAIL if required else Status.WARNING),
            "importable" if available else "not importable",
            exit_code=EXIT_DEPENDENCY if required and not available else None,
        )
    device = settings.matching_dinov2_device.lower()
    device_ok = device in {"auto", "cpu", "cuda"}
    fallback_ok = settings.matching_dinov2_allow_cpu_fallback or device == "cpu"
    report.add(
        section, "DINOv2 device policy",
        Status.OK if device_ok and fallback_ok else Status.WARNING,
        f"device={device}; cpu_fallback={'enabled' if settings.matching_dinov2_allow_cpu_fallback else 'disabled'}",
    )


def _find_calibration(root: Path) -> Path | None:
    base = root / "models" / "localization"
    candidates = list(base.glob("*calib*.json")) + list(base.glob("*camera*.json")) + list(base.glob("*.yaml")) + list(base.glob("*.yml"))
    return candidates[0] if candidates else None


def _calibration_fields(path: Path) -> tuple[bool, bool, bool]:
    try:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        has_k = bool(re.search(r'(^|["\s])k["\s]*:', lowered, re.MULTILINE)) or "camera_matrix" in lowered
        has_dist = "distortion" in lowered or "dist_coeff" in lowered
        has_resolution = "1920" in lowered and "1080" in lowered
        return has_k, has_dist, has_resolution
    except OSError:
        return False, False, False


def _check_tasks(report: Report, settings: Settings, skip_models: bool) -> None:
    detection_model_ready = bool(
        settings.detection_model_path and settings.detection_model_path.is_file()
    )
    if not settings.detection_enabled:
        detection_status = Status.OK
        detection_message = "implemented; disabled by configuration"
    elif detection_model_ready:
        detection_status = Status.OK
        detection_message = "implemented and enabled"
    else:
        detection_status = Status.WARNING
        detection_message = "implemented; enabled but model artifact is unavailable"
    report.add("Task 1 Detection", "service", detection_status, detection_message)
    if skip_models:
        report.add("Task 1 Detection", "model", Status.SKIPPED, "--skip-models")
    else:
        _artifact_check(report, "Task 1 Detection", "best.pt / configured model", settings.detection_model_path, settings.detection_enabled)
    report.add("Task 1 Detection", "Ultralytics", Status.OK if _importable("ultralytics") else Status.WARNING,
               "importable" if _importable("ultralytics") else "not importable")
    try:
        from app.schemas import ObjectClass
        from app.services.detection.class_mapping import DEFAULT_YOLO_CLASS_MAPPING

        mapping_valid = (
            set(DEFAULT_YOLO_CLASS_MAPPING) == {0, 1, 2, 3}
            and set(DEFAULT_YOLO_CLASS_MAPPING.values()) == set(ObjectClass)
        )
    except Exception:
        mapping_valid = False
    report.add(
        "Task 1 Detection",
        "class mapping",
        Status.OK if mapping_valid else Status.FAIL,
        "built-in central ObjectClass mapping ready" if mapping_valid
        else "built-in class mapping invalid",
        exit_code=None if mapping_valid else EXIT_ARTIFACT,
    )
    thresholds_ok = 0 <= settings.detection_confidence <= 1 and 0 <= settings.detection_iou <= 1
    report.add("Task 1 Detection", "confidence/IoU thresholds", Status.OK if thresholds_ok else Status.FAIL,
               f"confidence={settings.detection_confidence}; iou={settings.detection_iou}", exit_code=None if thresholds_ok else EXIT_ARTIFACT)

    try:
        settings.validate_localization_vo()
        localization_config_valid = True
    except ValueError:
        localization_config_valid = False
    if not settings.localization_enabled:
        localization_status = Status.OK
        localization_message = "implemented; disabled by configuration"
    elif not settings.localization_vo_enabled:
        localization_status = Status.WARNING
        localization_message = "implemented; localization enabled but AffineVO disabled"
    elif localization_config_valid:
        localization_status = Status.OK
        localization_message = "AffineVO + GPS calibration implemented and enabled"
    else:
        localization_status = Status.FAIL
        localization_message = "AffineVO implemented; configuration invalid"
    report.add(
        "Task 2 Localization", "service", localization_status, localization_message,
        exit_code=EXIT_CONFIG if localization_status is Status.FAIL else None,
    )
    try:
        from app.services.localization.camera_model import CameraModelProvider

        camera_ready = CameraModelProvider(settings).for_resolution(
            settings.localization_camera_width, settings.localization_camera_height
        ) is not None
    except Exception:
        camera_ready = False
    camera_source = (
        "local calibration file" if settings.localization_camera_calibration_path is not None
        else "configured camera intrinsics"
    )
    report.add(
        "Task 2 Localization", "camera calibration",
        Status.OK if camera_ready else (Status.FAIL if settings.localization_enabled else Status.WARNING),
        f"camera model ready from {camera_source}" if camera_ready else "camera model invalid",
        exit_code=EXIT_ARTIFACT if settings.localization_enabled and not camera_ready else None,
    )
    expected_frame_valid = settings.localization_calibration_expected_max_frame >= 1
    report.add(
        "Task 2 Localization", "calibration expected max frame",
        Status.OK if expected_frame_valid else Status.FAIL,
        f"LOCALIZATION_CALIBRATION_EXPECTED_MAX_FRAME={settings.localization_calibration_expected_max_frame}",
        exit_code=None if expected_frame_valid else EXIT_CONFIG,
    )
    if not settings.localization_enabled:
        gps_calibration_status = Status.OK
        gps_calibration_message = "implemented; inactive while localization service is disabled"
    elif settings.localization_calibration_enabled:
        gps_calibration_status = Status.OK
        gps_calibration_message = "implemented and enabled"
    else:
        gps_calibration_status = Status.WARNING
        gps_calibration_message = "implemented; disabled by configuration"
    report.add(
        "Task 2 Localization", "GPS scale calibration",
        gps_calibration_status, gps_calibration_message,
    )

    try:
        settings.validate_matching_local()
        matching_config_valid = True
    except ValueError as exc:
        matching_config_valid = False
        report.add(
            "Task 3 Matching", "local refinement config", Status.FAIL, str(exc),
            exit_code=EXIT_CONFIG,
        )
    if not settings.matching_enabled:
        matching_status = Status.OK
        matching_message = "implemented; disabled by configuration"
    elif not settings.matching_dinov2_enabled:
        matching_status = Status.WARNING
        matching_message = "implemented; matching enabled but DINOv2 runtime disabled"
    elif not matching_config_valid:
        matching_status = Status.FAIL
        matching_message = "implemented; matching configuration invalid"
    else:
        matching_status = Status.OK
        if settings.matching_geometry_method == "hybrid":
            matching_message = (
                "implemented and enabled: DINOv2 coarse gate -> ALIKED -> LightGlue -> "
                "USAC_MAGSAC -> polygon -> bbox -> confidence -> MatchedReferenceObject"
            )
        elif settings.matching_geometry_method == "aliked_lightglue":
            matching_message = "implemented and enabled: ALIKED + LightGlue local geometry"
        else:
            matching_message = (
                "implemented and enabled: DINOv2 dense descriptor -> mutual nearest-neighbor "
                "-> USAC_MAGSAC homography -> polygon validation -> bbox -> dynamic confidence "
                "-> MatchedReferenceObject"
            )
    report.add("Task 3 Matching", "service", matching_status, matching_message)
    _check_dinov2(report, settings, skip_models)
    local_required = bool(
        settings.matching_enabled
        and settings.matching_local_refinement_enabled
        and settings.matching_geometry_method in {"hybrid", "aliked_lightglue"}
    )
    xoftr_required = bool(settings.matching_enabled and settings.matching_xoftr_enabled)
    artifacts = (
        ("ALIKED", settings.matching_aliked_model_path, local_required),
        ("LightGlue", settings.matching_lightglue_model_path, local_required),
        ("XoFTR TorchScript (legacy)", settings.matching_xoftr_model_path, False),
        ("XoFTR checkpoint", settings.matching_xoftr_ckpt_path, xoftr_required),
    )
    for name, path, required in artifacts:
        if skip_models:
            report.add("Task 3 Matching", name, Status.SKIPPED, "--skip-models")
        else:
            _artifact_check(report, "Task 3 Matching", name, path, required)
    xoftr_repo_ok = bool(
        settings.matching_xoftr_repo_path and settings.matching_xoftr_repo_path.is_dir()
    )
    report.add(
        "Task 3 Matching", "XoFTR local repository",
        Status.OK if xoftr_repo_ok else (Status.FAIL if xoftr_required else Status.WARNING),
        "local directory ready" if xoftr_repo_ok else "local repository missing",
        exit_code=EXIT_ARTIFACT if xoftr_required and not xoftr_repo_ok else None,
        path=str(settings.matching_xoftr_repo_path) if settings.matching_xoftr_repo_path else "not configured",
    )
    report.add(
        "Task 3 Matching",
        "geometry method",
        Status.OK,
        f"MATCHING_GEOMETRY_METHOD={settings.matching_geometry_method}; "
        f"DINOv2 fallback={'enabled' if settings.matching_local_fallback_to_dinov2 else 'disabled'}",
    )
    preload_ready = settings.matching_preload_models and settings.matching_warmup_enabled
    report.add(
        "Task 3 Matching",
        "preload/warmup",
        Status.OK if preload_ready else Status.WARNING,
        (
            "configured: model preload and dummy inference warmup enabled"
            if preload_ready
            else "disabled or partially configured"
        ),
    )
    report.add(
        "Task 3 Matching",
        "local refinement timeout",
        Status.OK,
        f"MATCHING_LOCAL_REFINEMENT_TIMEOUT_SEC={settings.matching_local_refinement_timeout_sec}",
    )
    report.add(
        "Task 3 Matching", "RGB-RGB production pipeline",
        Status.OK if settings.matching_enabled and settings.matching_dinov2_enabled else Status.WARNING,
        "dense descriptor, mutual NN, homography, polygon, bbox and dynamic confidence ready"
        if settings.matching_enabled and settings.matching_dinov2_enabled
        else "pipeline implemented but disabled by configuration",
    )
    report.add(
        "Task 3 Matching", "thermal pipeline",
        Status.OK if xoftr_required else Status.WARNING,
        "XoFTR cross-modal path enabled (thermal reference -> RGB frame)"
        if xoftr_required
        else "disabled; thermal references will be skipped",
    )
    cache = report.project_root / "work" / "reference_cache"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        writable = os.access(cache, os.W_OK)
    except OSError:
        writable = False
    report.add("Task 3 Matching", "reference cache", Status.OK if writable else Status.FAIL,
               "writable" if writable else "not writable", exit_code=None if writable else EXIT_ARTIFACT)


def _check_storage(report: Report) -> None:
    for name in ("work", "logs"):
        path = report.project_root / name
        try:
            path.mkdir(parents=True, exist_ok=True)
            ok = os.access(path, os.W_OK)
        except OSError:
            ok = False
        report.add("Storage", name, Status.OK if ok else Status.FAIL, "writable" if ok else "not writable", exit_code=None if ok else EXIT_CONFIG)
    models = report.project_root / "models"
    report.add("Storage", "models", Status.OK if models.is_dir() and os.access(models, os.R_OK) else Status.FAIL,
               "readable" if models.is_dir() and os.access(models, os.R_OK) else "not readable", exit_code=EXIT_ARTIFACT)
    free = shutil.disk_usage(report.project_root).free
    enough = free >= 5 * 1024**3
    report.add("Storage", "free disk", Status.OK if enough else Status.WARNING,
               f"{free / 1024**3:.2f} GiB free; 5 GiB recommended", free_bytes=free)
    media = report.project_root / "work" / "official_media"
    try:
        media.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix="preflight-", dir=media)
        os.close(fd)
        Path(raw).unlink()
        ok = True
    except OSError:
        ok = False
    report.add("Storage", "official_media write/delete", Status.OK if ok else Status.FAIL,
               "temporary file created and removed" if ok else "write/delete failed", exit_code=None if ok else EXIT_CONFIG)


def _check_gpu(report: Report, settings: Settings, skip_gpu: bool, import_probe: ImportProbe) -> None:
    if skip_gpu:
        report.add("GPU", "GPU inspection", Status.SKIPPED, "--skip-gpu")
        return
    if not import_probe("torch"):
        report.add("GPU", "PyTorch CUDA", Status.WARNING, "torch missing; GPU check skipped")
        report.add("GPU", "CPU fallback", Status.OK if settings.matching_dinov2_allow_cpu_fallback else Status.WARNING,
                   "enabled" if settings.matching_dinov2_allow_cpu_fallback else "disabled")
        return
    try:
        import torch
        available = bool(torch.cuda.is_available())
        if not available:
            report.add("GPU", "CUDA", Status.WARNING, "GPU unavailable")
        else:
            props = torch.cuda.get_device_properties(0)
            report.add("GPU", "CUDA", Status.OK, "available", gpu_name=props.name,
                       cuda_version=torch.version.cuda, memory_bytes=props.total_memory,
                       compute_capability=f"{props.major}.{props.minor}", cudnn_enabled=bool(torch.backends.cudnn.enabled))
        report.add("GPU", "CPU fallback", Status.OK if settings.matching_dinov2_allow_cpu_fallback else Status.WARNING,
                   "enabled" if settings.matching_dinov2_allow_cpu_fallback else "disabled")
    except Exception:
        report.add("GPU", "GPU inspection", Status.WARNING, "inspection failed")


def _check_application(report: Report, settings: Settings) -> None:
    try:
        from httpx import ASGITransport, AsyncClient
        from app.main import create_app

        safe_settings = replace(settings, api_key="preflight-local-key", matching_enabled=False,
                                detection_enabled=False, localization_enabled=False)
        app = create_app(settings=safe_settings)
        schema = app.openapi()
        if not schema.get("paths"):
            raise RuntimeError("OpenAPI paths missing")

        async def probe() -> tuple[int, int, dict]:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://preflight.invalid") as client:
                health = await client.get("/health")
                response = await client.post("/process_frame", headers={"X-API-Key": "preflight-local-key"}, json={
                    "url": "preflight-frame", "image_url": "not-read.jpg", "video_name": "preflight",
                    "session": "preflight", "gps_health_status": 0,
                })
                return health.status_code, response.status_code, response.json()

        health_code, process_code, body = asyncio.run(probe())
        safe_empty = process_code == 200 and body.get("detected_objects") == [] and body.get("detected_translations") == [] and body.get("detected_undefined_objects") == []
        report.add("Application", "app.main import/create", Status.OK, "FastAPI app created")
        report.add("Application", "OpenAPI", Status.OK, "schema generated")
        report.add("Application", "/health", Status.OK if health_code == 200 else Status.FAIL, f"HTTP {health_code}", exit_code=None if health_code == 200 else EXIT_APPLICATION)
        report.add("Application", "/process_frame disabled services", Status.OK if safe_empty else Status.FAIL,
                   "HTTP 200 with empty typed results" if safe_empty else f"unsafe or invalid response (HTTP {process_code})",
                   exit_code=None if safe_empty else EXIT_APPLICATION)
    except Exception as exc:
        report.add("Application", "startup/schema", Status.FAIL, f"local validation failed: {type(exc).__name__}", exit_code=EXIT_APPLICATION)


def _iter_source_files(root: Path) -> Iterable[Path]:
    for base in (root / "app", root / "competition", root / "scripts"):
        if base.exists():
            yield from base.rglob("*.py")


def _check_security(report: Report, settings: Settings) -> None:
    sensitive_values = [value for value in (settings.team_name, settings.password) if value and len(value) >= 4]
    patterns = (
        re.compile(r"(?i)authorization\s*[:=]\s*['\"]bearer\s+[^'\"]+"),
        re.compile(r"(?i)(?:password|team_name)\s*=\s*['\"][^'\"]{4,}['\"]"),
    )
    findings: list[tuple[str, int]] = []
    for path in _iter_source_files(report.project_root):
        try:
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if any(pattern.search(line) for pattern in patterns) or any(value in line for value in sensitive_values):
                    findings.append((str(path.relative_to(report.project_root)), line_no))
        except (OSError, UnicodeError):
            continue
    if findings:
        for path, line in findings:
            report.add("Security", "hardcoded credential", Status.FAIL, f"sensitive pattern at {path}:{line}", exit_code=EXIT_SECURITY)
    else:
        report.add("Security", "hardcoded credentials", Status.OK, "none detected")
    gitignore = report.project_root / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    normalized = {line.strip().rstrip("/") for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    for rule in IGNORE_RULES:
        key = rule.rstrip("/")
        present = key in normalized or (key == "work" and any(v.startswith("work/") for v in normalized))
        report.add("Security", f".gitignore {rule}", Status.OK if present else Status.WARNING,
                   "covered" if present else "rule missing")
    # Source endpoint and prediction-path audit; values are never emitted.
    forbidden: list[tuple[str, int]] = []
    prediction_calls: list[str] = []
    legacy_frame_token = "/get" + "_frame"
    legacy_result_token = "/send" + "_result"
    prediction_call_token = "." + "send_prediction("
    for path in (report.project_root / "competition").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if legacy_frame_token in line or legacy_result_token in line:
                forbidden.append((path.name, line_no))
            if prediction_call_token in line:
                prediction_calls.append(path.name)
    if forbidden:
        for name, line in forbidden:
            report.add("Security", "legacy endpoint", Status.FAIL, f"found at {name}:{line}", exit_code=EXIT_SECURITY)
    else:
        report.add("Security", "legacy endpoints", Status.OK, "none detected")
    allowed = {"runner.py", "official_interface_adapter.py"}
    paths_ok = set(prediction_calls) <= allowed
    preflight_source = Path(__file__).read_text(encoding="utf-8")
    connection_path = report.project_root / "competition" / "connection_check.py"
    connection_source = connection_path.read_text(encoding="utf-8") if connection_path.is_file() else ""
    readonly_ok = prediction_call_token not in preflight_source and prediction_call_token not in connection_source
    report.add("Security", "prediction call paths", Status.OK if paths_ok and readonly_ok else Status.FAIL,
               "restricted to runner transport path; dry-run tools contain no call" if paths_ok and readonly_ok else "unexpected prediction call path",
               exit_code=None if paths_ok and readonly_ok else EXIT_SECURITY)


def _check_tests(report: Report, options: Options, command_runner: CommandRunner) -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AEROSYNC_PREFLIGHT_TEST_MODE"] = "1"
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    command.extend(["-m", "not online"])
    if not options.run_tests:
        command.extend(["--collect-only", "-q"])
    start = time.monotonic()
    try:
        result = command_runner(command, report.project_root, env)
        elapsed = time.monotonic() - start
        ok = result.returncode == 0
        label = "tests passed" if options.run_tests else "collection successful"
        tail = (result.stdout or result.stderr).strip().splitlines()[-1:] or [label]
        report.add("Tests", "pytest" if options.run_tests else "pytest collection",
                   Status.OK if ok else Status.FAIL, f"{label if ok else 'failed'} in {elapsed:.2f}s; {tail[0][:160]}",
                   exit_code=None if ok else EXIT_TEST, duration_seconds=round(elapsed, 3), real_server=False)
    except Exception as exc:
        report.add("Tests", "pytest", Status.FAIL, f"execution failed: {type(exc).__name__}", exit_code=EXIT_TEST)


def _default_online_runner(settings: Settings, fetch_frame: bool, verbose: bool) -> int:
    from competition.connection_check import CheckOptions, run_connection_check
    return run_connection_check(settings, CheckOptions(fetch_frame=fetch_frame, verbose=verbose))


def _check_online(report: Report, options: Options, settings: Settings, online_runner: OnlineRunner) -> None:
    if not options.online:
        report.add("Online Connectivity", "server connection", Status.SKIPPED, "offline default; --online not supplied")
        report.add("Online Connectivity", "prediction submission", Status.SKIPPED, "DISABLED; submitted=0")
        return
    try:
        code = online_runner(settings, options.fetch_frame, options.verbose)
    except Exception:
        code = -1
    report.add("Online Connectivity", "connection_check", Status.OK if code == 0 else Status.FAIL,
               "authentication/progress/reference metadata completed" if code == 0 else f"safe connection check failed (code {code})",
               exit_code=None if code == 0 else EXIT_ONLINE, frame_requested=1 if options.fetch_frame else 0)
    report.add("Online Connectivity", "prediction submission", Status.SKIPPED, "DISABLED; submitted=0")


def run_preflight(
    project_root: Path,
    settings: Settings,
    options: Options,
    *,
    import_probe: ImportProbe = _importable,
    command_runner: CommandRunner = _run_command,
    online_runner: OnlineRunner = _default_online_runner,
) -> Report:
    report = Report(project_root.resolve())
    _check_configuration(report, settings)
    _check_python(report, command_runner)
    _check_dependencies(report, import_probe)
    _check_official(report, settings)
    _check_tasks(report, settings, options.skip_models)
    _check_storage(report)
    _check_gpu(report, settings, options.skip_gpu, import_probe)
    _check_application(report, settings)
    _check_security(report, settings)
    _check_tests(report, options, command_runner)
    _check_online(report, options, settings, online_runner)
    return report


def _emit_report(report: Report, strict: bool, emit: Callable[[str], None]) -> None:
    emit("AeroSync Preflight Check")
    for section in SECTIONS:
        emit(f"\n{section}")
        for check in (item for item in report.checks if item.section == section):
            emit(f"[{check.status.value}] {check.name}: {check.message}")
    counts = report.counts()
    emit("\nFinal Summary")
    emit(f"Total checks: {len(report.checks)}")
    emit(f"OK count: {counts['OK']}")
    emit(f"Warning count: {counts['WARNING']}")
    emit(f"Fail count: {counts['FAIL']}")
    emit(f"Skipped count: {counts['SKIPPED']}")
    emit(f"Competition readiness: {report.readiness()}")
    critical = [c for c in report.checks if c.status is Status.FAIL]
    if critical:
        emit("Critical gaps:")
        for item in critical:
            emit(f"- {item.section} / {item.name}: {item.message}")
    emit(f"Exit code: {report.exit_code(strict)}")


def write_json_report(report: Report, path: Path, strict: bool) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.payload(strict), ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fetch_frame and not args.online:
        print("[FAIL] --fetch-frame yalnızca --online ile kullanılabilir.")
        return EXIT_CONFIG
    options = Options(
        online=args.online, fetch_frame=args.fetch_frame, run_tests=args.run_tests,
        verbose=args.verbose, json_output=args.json_output, strict=args.strict,
        skip_gpu=args.skip_gpu, skip_models=args.skip_models,
    )
    root = Path(__file__).resolve().parents[1]
    report = run_preflight(root, get_settings(), options)
    _emit_report(report, options.strict, print)
    if options.json_output:
        write_json_report(report, options.json_output, options.strict)
    return report.exit_code(options.strict)


if __name__ == "__main__":
    raise SystemExit(main())
