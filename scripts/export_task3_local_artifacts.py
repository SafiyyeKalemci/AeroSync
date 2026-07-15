from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Dict


class AlikedContractWrapper:
    """Namespace for the scripted module class, created after torch is imported."""


def _wrapper_classes(torch):
    class AlikedCoreWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            output = self.model({"image": image})
            return {
                "keypoints": output["keypoints"],
                "descriptors": output["descriptors"],
                "scores": output["keypoint_scores"],
            }

    class AlikedPaddedWrapper(torch.nn.Module):
        def __init__(self, core, canvas_size: int):
            super().__init__()
            self.core = core
            self.canvas_size = canvas_size

        def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
            height = image.size(2)
            width = image.size(3)
            if height > self.canvas_size or width > self.canvas_size:
                raise RuntimeError("ALIKED input exceeds export canvas")
            padded = torch.nn.functional.pad(
                image, (0, self.canvas_size - width, 0, self.canvas_size - height)
            )
            output = self.core(padded)
            keypoints = output["keypoints"]
            keep = (
                (keypoints[0, :, 0] >= 0)
                & (keypoints[0, :, 0] <= float(width))
                & (keypoints[0, :, 1] >= 0)
                & (keypoints[0, :, 1] <= float(height))
            )
            return {
                "keypoints": keypoints[:, keep],
                "descriptors": output["descriptors"][:, keep],
                "scores": output["scores"][:, keep],
            }

    class LightGlueContractWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            features0: Dict[str, torch.Tensor],
            features1: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            output = self.model(
                {
                    "image0": {
                        "keypoints": features0["keypoints"],
                        "descriptors": features0["descriptors"],
                        "image_size": features0["image_size"],
                    },
                    "image1": {
                        "keypoints": features1["keypoints"],
                        "descriptors": features1["descriptors"],
                        "image_size": features1["image_size"],
                    },
                }
            )
            return {
                "matches0": output["matches0"],
                "matching_scores0": output["matching_scores0"],
            }

    return AlikedCoreWrapper, AlikedPaddedWrapper, LightGlueContractWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resmi ALIKED + LightGlue agirliklarini local-only TorchScript artifact'lara aktar."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("external/LightGlue"))
    parser.add_argument("--aliked-weights", type=Path)
    parser.add_argument("--lightglue-weights", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/matching"))
    parser.add_argument("--canvas-size", type=int, default=1120)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    return parser


def export_artifacts(
    source_dir: Path,
    aliked_weights: Path,
    lightglue_weights: Path,
    output_dir: Path,
    *,
    canvas_size: int = 1024,
    max_keypoints: int = 1024,
) -> dict[str, object]:
    import torch

    source_dir = source_dir.expanduser().resolve()
    aliked_weights = aliked_weights.expanduser().resolve()
    lightglue_weights = lightglue_weights.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_inputs(source_dir, aliked_weights, lightglue_weights, canvas_size, max_keypoints)
    output_dir.mkdir(parents=True, exist_ok=True)
    aliked_module, lightglue_module = _load_official_modules(source_dir, torch)
    CoreWrapper, PaddedWrapper, MatcherWrapper = _wrapper_classes(torch)
    original_loader = torch.hub.load_state_dict_from_url
    try:
        torch.hub.load_state_dict_from_url = lambda *_args, **_kwargs: torch.load(
            aliked_weights, map_location="cpu", weights_only=True
        )
        aliked = aliked_module.ALIKED(
            model_name="aliked-n16",
            max_num_keypoints=max_keypoints,
            detection_threshold=-1.0,
        ).eval()
        canvas = torch.rand(1, 3, canvas_size, canvas_size)
        traced_core = torch.jit.trace(
            CoreWrapper(aliked).eval(), canvas, strict=False, check_trace=False
        )
        scripted_aliked = torch.jit.script(
            PaddedWrapper(traced_core, canvas_size).eval()
        )

        sample = scripted_aliked(torch.rand(1, 3, canvas_size // 2, canvas_size))
        feature_sample = {
            "keypoints": sample["keypoints"],
            "descriptors": sample["descriptors"],
            "scores": sample["scores"],
            "image_size": torch.tensor(
                [[float(canvas_size), float(canvas_size // 2)]], dtype=torch.float32
            ),
        }
        torch.hub.load_state_dict_from_url = lambda *_args, **_kwargs: torch.load(
            lightglue_weights, map_location="cpu", weights_only=True
        )
        lightglue = lightglue_module.LightGlue(
            features="aliked",
            depth_confidence=-1,
            width_confidence=-1,
            flash=False,
        ).eval()
        traced_lightglue = torch.jit.trace(
            MatcherWrapper(lightglue).eval(),
            (feature_sample, feature_sample),
            strict=False,
            check_trace=False,
        )
    finally:
        torch.hub.load_state_dict_from_url = original_loader

    aliked_path = output_dir / "aliked.pt"
    lightglue_path = output_dir / "lightglue_aliked.pt"
    _atomic_save(scripted_aliked, aliked_path)
    _atomic_save(traced_lightglue, lightglue_path)
    report = {
        "source_dir": str(source_dir),
        "source_weights": {
            "aliked": _file_report(aliked_weights),
            "lightglue": _file_report(lightglue_weights),
        },
        "artifacts": {
            "aliked": _file_report(aliked_path),
            "lightglue": _file_report(lightglue_path),
        },
        "canvas_size": canvas_size,
        "max_keypoints": max_keypoints,
        "network_downloads": "DISABLED",
        "prediction_submission": "DISABLED",
    }
    (output_dir / "task3_local_artifacts.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _validate_inputs(source_dir, aliked_weights, lightglue_weights, canvas_size, max_keypoints):
    required = (
        source_dir / "lightglue" / "aliked.py",
        source_dir / "lightglue" / "lightglue.py",
        source_dir / "lightglue" / "utils.py",
        aliked_weights,
        lightglue_weights,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Eksik resmi kaynak/artifact: " + ", ".join(missing))
    if canvas_size < 64 or max_keypoints < 1:
        raise ValueError("canvas-size >= 64 ve max-keypoints >= 1 olmali")


def _load_official_modules(source_dir: Path, torch):
    package_dir = source_dir / "lightglue"
    kornia = types.ModuleType("kornia")
    color = types.ModuleType("kornia.color")
    color.grayscale_to_rgb = lambda value: value.repeat_interleave(3, dim=-3)
    kornia.color = color
    kornia.geometry = types.SimpleNamespace(
        transform=types.SimpleNamespace(resize=lambda value, *_a, **_k: value)
    )
    sys.modules.setdefault("kornia", kornia)
    sys.modules.setdefault("kornia.color", color)
    package = types.ModuleType("lightglue")
    package.__path__ = [str(package_dir)]
    sys.modules["lightglue"] = package

    def load(name: str):
        qualified = f"lightglue.{name}"
        spec = importlib.util.spec_from_file_location(qualified, package_dir / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Resmi LightGlue modulu yuklenemedi: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        return module

    load("utils")
    return load("aliked"), load("lightglue")


def _atomic_save(module, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    module.save(str(temporary))
    temporary.replace(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_report(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_dir.expanduser().resolve()
    aliked_weights = (args.aliked_weights or source / "weights" / "aliked-n16.pth")
    lightglue_weights = (
        args.lightglue_weights or source / "weights" / "aliked_lightglue.pth"
    )
    try:
        report = export_artifacts(
            source,
            aliked_weights,
            lightglue_weights,
            args.output_dir,
            canvas_size=args.canvas_size,
            max_keypoints=args.max_keypoints,
        )
    except Exception as exc:
        print(f"Task 3 artifact export: FAIL ({type(exc).__name__}: {exc})")
        print("Prediction submission: DISABLED")
        return 10
    print(json.dumps(report, indent=2))
    print("Prediction submission: DISABLED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
