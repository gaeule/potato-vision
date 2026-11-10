"""Shared paths, artifact ownership, and provenance for the potato baseline."""

import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
BASE: Final = ROOT / "baseline"
SOURCE: Final = ROOT / "KakaoTalk_261020_161759.mp4"
PROVENANCE: Final = "agent_visual_review_not_human_expert"


class PipelineError(RuntimeError):
    """An invalid input or unsafe output operation."""


def sha256(path: Path) -> str:
    """Hash a source without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_directory(path: Path, regenerate: bool) -> Path:
    """Allow reuse only of directories marked as this pipeline's artifacts."""
    marker = path / ".potato-pipeline"
    if path.exists():
        if not regenerate or not marker.is_file():
            raise PipelineError(f"Refusing existing output {path}; use a new path or --regenerate on owned artifacts")
    else:
        path.mkdir(parents=True)
        marker.write_text("potato-baseline-v1\n", encoding="utf-8")
    return path


def write_json[T](path: Path, value: T) -> None:
    """Write a machine-readable pipeline artifact."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def label_line(box: tuple[float, float, float, float], size: tuple[int, int]) -> str:
    """Convert a validated pixel box into one normalized class-zero YOLO row."""
    x1, y1, x2, y2 = box
    width, height = size
    return f"0 {(x1+x2)/2/width:.9f} {(y1+y2)/2/height:.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}\n"
