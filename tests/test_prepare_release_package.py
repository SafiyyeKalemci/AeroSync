from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.prepare_release_package import (
    INCLUDED_DIRECTORIES,
    INCLUDED_EXTERNAL_DIRECTORIES,
    INCLUDED_FILES,
    prepare_release,
)


def source_tree(root: Path) -> Path:
    for name in INCLUDED_DIRECTORIES:
        (root / name).mkdir(parents=True)
        (root / name / "keep.txt").write_text("keep", encoding="utf-8")
    for name in INCLUDED_FILES:
        (root / name).write_text("safe", encoding="utf-8")
    (root / "models/detection").mkdir(parents=True, exist_ok=True)
    (root / "models/matching").mkdir(parents=True, exist_ok=True)
    (root / "models/detection/best.pt").write_bytes(b"yolo")
    (root / "models/matching/dinov2_vitb14_pretrain.pth").write_bytes(b"dino")
    (root / "models/matching/aliked.pt").write_bytes(b"aliked")
    (root / "models/matching/lightglue_aliked.pt").write_bytes(b"lightglue")
    for name in INCLUDED_EXTERNAL_DIRECTORIES:
        (root / "external" / name).mkdir(parents=True)
        (root / "external" / name / "keep.txt").write_text("keep", encoding="utf-8")
    (root / "external/dinov2-main/hubconf.py").write_text("# local", encoding="utf-8")
    (root / "official_interface/src").mkdir(parents=True, exist_ok=True)
    (root / "official_interface/main.py").write_text("# official", encoding="utf-8")
    (root / "official_interface/src/connection_handler.py").write_text("# official", encoding="utf-8")
    (root / "official_interface/src/frame_predictions.py").write_text("# official", encoding="utf-8")
    (root / ".env").write_text("PASSWORD=private\n", encoding="utf-8")
    (root / "work").mkdir()
    (root / "work/output.txt").write_text("temporary", encoding="utf-8")
    (root / "app/__pycache__").mkdir()
    (root / "app/__pycache__/module.pyc").write_bytes(b"cache")
    return root


def test_default_release_excludes_env_outputs_and_caches(tmp_path):
    source = source_tree(tmp_path / "source")
    result = prepare_release(source, tmp_path / "release", create_zip=True)
    assert not (result.release_directory / ".env").exists()
    assert (result.release_directory / ".env.example").is_file()
    assert not (result.release_directory / "work").exists()
    assert not list(result.release_directory.rglob("__pycache__"))
    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
    assert "AeroSync-integrated/.env" not in names
    assert "AeroSync-integrated/.env.example" in names


def test_include_env_is_explicit_and_existing_target_is_not_overwritten(tmp_path):
    source = source_tree(tmp_path / "source")
    result = prepare_release(source, tmp_path / "release", include_env=True)
    assert (result.release_directory / ".env").read_text(encoding="utf-8") == "PASSWORD=private\n"
    with pytest.raises(FileExistsError):
        prepare_release(source, tmp_path / "release")


def test_custom_portable_zip_name_and_official_interface_are_included(tmp_path):
    source = source_tree(tmp_path / "source")
    result = prepare_release(
        source,
        tmp_path / "release",
        create_zip=True,
        zip_name="AeroSync-integrated-portable.zip",
    )
    assert result.zip_path.name == "AeroSync-integrated-portable.zip"
    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
    assert "AeroSync-integrated/official_interface/main.py" in names
    assert "AeroSync-integrated/official_interface/src/connection_handler.py" in names
    assert "AeroSync-integrated/external/LightGlue/keep.txt" in names
    assert "AeroSync-integrated/models/matching/aliked.pt" in names
    assert "AeroSync-integrated/models/matching/lightglue_aliked.pt" in names
    with pytest.raises(ValueError):
        prepare_release(source, tmp_path / "invalid", create_zip=True, zip_name="../bad.zip")
