from pathlib import Path

from app.core.config import get_settings


def status(label: str, path: Path | None, expect_directory: bool = False) -> bool:
    present = bool(path and (path.is_dir() if expect_directory else path.is_file()))
    print(f"{'OK' if present else 'MISSING'} | {label} | {path or '-'}")
    return present


def main() -> int:
    settings = get_settings()
    required = [
        status("DINOv2 repo", settings.matching_dinov2_repo_path, expect_directory=True),
        status("DINOv2 weights", settings.matching_dinov2_weights_path),
    ]
    status("ALIKED TorchScript (optional)", settings.matching_aliked_model_path)
    status("LightGlue TorchScript (optional)", settings.matching_lightglue_model_path)
    status("XoFTR TorchScript (optional)", settings.matching_xoftr_model_path)
    return 0 if all(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
