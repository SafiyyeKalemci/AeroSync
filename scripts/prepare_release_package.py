from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "AeroSync-integrated"

INCLUDED_DIRECTORIES = (
    "app",
    "competition",
    "scripts",
    "tests",
    "docs",
    "models",
    "docker",
    "official_interface",
)
INCLUDED_FILES = (
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
)
INCLUDED_EXTERNAL_DIRECTORIES = (
    "dinov2-main",
    "LightGlue",
)
EXCLUDED_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "work",
    "logs",
    "runs",
    "release",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True, slots=True)
class PackageResult:
    release_directory: Path
    zip_path: Path | None
    file_count: int
    total_bytes: int
    env_included: bool


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in EXCLUDED_NAMES or Path(name).suffix.casefold() in EXCLUDED_SUFFIXES
    }


def _validate_source(source: Path) -> None:
    required = [source / name for name in INCLUDED_DIRECTORIES + INCLUDED_FILES]
    required.extend(
        (
            source / "models" / "detection" / "best.pt",
            source / "models" / "matching" / "dinov2_vitb14_pretrain.pth",
            source / "models" / "matching" / "aliked.pt",
            source / "models" / "matching" / "lightglue_aliked.pt",
            source / "external" / "dinov2-main" / "hubconf.py",
            source / "external" / "LightGlue",
            source / "official_interface" / "main.py",
            source / "official_interface" / "src" / "connection_handler.py",
            source / "official_interface" / "src" / "frame_predictions.py",
        )
    )
    missing = [str(path.relative_to(source)) for path in required if not path.exists()]
    if missing:
        raise ValueError("Release icin kritik dosya/klasor eksik: " + ", ".join(missing))


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=_ignore)


def prepare_release(
    source_root: Path,
    output_dir: Path,
    *,
    include_env: bool = False,
    create_zip: bool = False,
    zip_name: str = f"{PACKAGE_NAME}.zip",
) -> PackageResult:
    source = source_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    target = output / PACKAGE_NAME
    zip_filename = Path(zip_name)
    if zip_filename.name != zip_name or zip_filename.suffix.casefold() != ".zip":
        raise ValueError("ZIP adi yalnizca .zip uzantili bir dosya adi olmalidir.")
    zip_path = output / zip_filename
    _validate_source(source)
    if target.exists():
        raise FileExistsError(f"Release hedefi zaten mevcut: {target}")
    if create_zip and zip_path.exists():
        raise FileExistsError(f"ZIP hedefi zaten mevcut: {zip_path}")
    output.mkdir(parents=True, exist_ok=True)
    target.mkdir()

    for name in INCLUDED_DIRECTORIES:
        _copy_tree(source / name, target / name)
    for name in INCLUDED_EXTERNAL_DIRECTORIES:
        _copy_tree(source / "external" / name, target / "external" / name)
    for name in INCLUDED_FILES:
        shutil.copy2(source / name, target / name)
    if include_env:
        env_file = source / ".env"
        if not env_file.is_file():
            raise ValueError("--include-env istendi ancak kaynak .env bulunamadi.")
        shutil.copy2(env_file, target / ".env")

    forbidden = [
        path
        for path in target.rglob("*")
        if path.name in EXCLUDED_NAMES or path.suffix.casefold() in EXCLUDED_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError("Release dislama politikasi ihlal edildi.")
    if not include_env and (target / ".env").exists():
        raise RuntimeError("Varsayilan release icinde .env bulunamaz.")

    files = [path for path in target.rglob("*") if path.is_file()]
    if create_zip:
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, Path(PACKAGE_NAME) / path.relative_to(target))
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if not names or any(not name.startswith(f"{PACKAGE_NAME}/") for name in names):
                raise RuntimeError("ZIP kok klasor yapisi gecersiz.")
            if not include_env and f"{PACKAGE_NAME}/.env" in names:
                raise RuntimeError("Varsayilan ZIP icinde .env bulunamaz.")
            if f"{PACKAGE_NAME}/.env.example" not in names:
                raise RuntimeError("ZIP icinde .env.example eksik.")

    return PackageResult(
        release_directory=target,
        zip_path=zip_path if create_zip else None,
        file_count=len(files),
        total_bytes=sum(path.stat().st_size for path in files),
        env_included=include_env,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a credential-safe AeroSync release folder")
    parser.add_argument("--output-dir", type=Path, default=Path("release_portable_v2"))
    parser.add_argument("--include-env", action="store_true", help="Include credential-bearing .env")
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    parser.add_argument("--zip-name", default=f"{PACKAGE_NAME}-portable-v2.zip")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_release(
            PROJECT_ROOT,
            args.output_dir,
            include_env=args.include_env,
            create_zip=args.create_zip,
            zip_name=args.zip_name,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Release package FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Release folder: {result.release_directory}")
    print(f"Files: {result.file_count}; bytes: {result.total_bytes}")
    print(f"Environment file included: {'YES' if result.env_included else 'NO'}")
    print(f"ZIP: {result.zip_path if result.zip_path else 'skipped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
