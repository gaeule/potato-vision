"""Read-only source hashing, rectangle conversion, and isolated output paths."""

import hashlib
import math
import sys
from pathlib import Path
from typing import Final

from PIL import Image

from retraining_data.native import Native, ReviewRow, Shape
from retraining_data.schema import require, safe_filename

ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUTS: Final = ROOT / "retraining_data/outputs"
DATASET: Final = OUTPUTS / "dataset"
LEGACY: Final = ROOT / "stone_experiment/outputs/dataset"
REVIEW: Final = ROOT / "label_review"
PACK: Final = ROOT / "postprocess_review/outputs/review_pack"
VIDEO: Final = ROOT / "KakaoTalk_261020_161759.mp4"
CLASS_IDS: Final = {"potato": 0, "stone": 1}

# Imported legacy helpers also set this; set it before any legacy module import.
sys.dont_write_bytecode = True


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    """Hash large files with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_pixels(path: Path) -> str:
    """Hash raw decoded RGB bytes, exactly as the review pack did."""
    with Image.open(path) as image:
        require(image.format == "PNG" and image.size == (1280, 720), f"Unexpected PNG/size: {path}")
        return digest(image.convert("RGB").tobytes())


def rectangle(shape: Shape, width: int, height: int) -> tuple[float, float, float, float]:
    """Accept two opposite corners or four actual axis-aligned rectangle corners."""
    require(len(shape.points) in (2, 4), "Rectangle requires two or four corners")
    require(all(math.isfinite(n) for p in shape.points for n in p), "Rectangle requires finite corners")
    xs = tuple(p[0] for p in shape.points)
    ys = tuple(p[1] for p in shape.points)
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "Rectangle outside image or empty")
    if len(shape.points) == 4:
        require(set(shape.points) == {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}, "Not rectangle corners")
    return x1, y1, x2, y2


def yolo(native: Native) -> bytes:
    """Convert native potato/stone rectangles; an empty frame stays zero bytes."""
    lines: list[str] = []
    for shape in native.shapes:
        x1, y1, x2, y2 = rectangle(shape, native.imageWidth, native.imageHeight)
        class_id = CLASS_IDS[shape.label]
        lines.append(
            f"{class_id} {(x1+x2)/2/native.imageWidth:.9f} {(y1+y2)/2/native.imageHeight:.9f} "
            + f"{(x2-x1)/native.imageWidth:.9f} {(y2-y1)/native.imageHeight:.9f}\n",
        )
    return "".join(lines).encode("ascii")


def eligibility(row: ReviewRow, native: Native) -> bool:
    """Require manifest selection to agree with every native review/checked flag."""
    safe_filename(row.filename)
    eligible = row.eligible_for_experimental_training
    require(
        eligible == native.review.eligible_for_experimental_training
        and native.flags.needs_confirmation == (not eligible),
        f"Manifest/native eligibility differs: {row.filename}",
    )
    directory = "candidates" if eligible else "needs_confirmation"
    status = "ai_reviewed_candidate" if eligible else "needs_confirmation"
    require(
        row.relative_image == f"{directory}/{row.filename}" and row.review_status == status
        and native.imagePath == row.filename
        and (row.requested_second, row.actual_second)
        == (native.review.requested_second, native.review.actual_second),
        f"Review provenance differs: {row.filename}",
    )
    require(
        (row.after_potato, row.after_stone)
        == (sum(s.label == "potato" for s in native.shapes), sum(s.label == "stone" for s in native.shapes)),
        f"Review shape counts differ: {row.filename}",
    )
    return eligible


def output_path(path: Path) -> Path:
    """Allow new output directories only beneath this package's outputs directory."""
    resolved = path.resolve()
    require(
        OUTPUTS.resolve() == OUTPUTS
        and resolved.is_relative_to(OUTPUTS) and resolved != OUTPUTS,
        f"Output must be below {OUTPUTS}",
    )
    require(not path.is_symlink(), "Output symlink refused")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def yaml_bytes() -> bytes:
    """Relative path keeps dataset bytes reproducible and directory-relocatable."""
    return b"train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: potato\n  1: stone\n"
