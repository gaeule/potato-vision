"""SAM2.1 box snapping; images are uint8 BGR, boxes use exclusive xyxy."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
from contextlib import chdir
from dataclasses import dataclass
from functools import cache
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Results
    from ultralytics.models.sam.model import SAM

type Image = NDArray[np.uint8]
type Mask = NDArray[np.bool_]
type Box = tuple[float, float, float, float]
ROOT: Final = Path(__file__).resolve().parent
WEIGHTS: Final = ROOT / "weights"


@dataclass(frozen=True, slots=True)
class SnapResult:
    """Geometry receipt, not potato identity or body-only visual approval."""

    xyxy: Box
    tight: bool
    reason: str
    mask_area_ratio: float


class SnapError(RuntimeError):
    """A violated runtime or checkpoint contract."""

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def digest(path: Path) -> str:
    """Hash checkpoint bytes without buffering the whole checkpoint."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@cache
def model() -> SAM:
    """Load once per fresh process with all mutable framework paths isolated."""
    if "ultralytics" in sys.modules:
        raise SnapError("fresh_process_required")
    private = WEIGHTS / "runtime" / f"process-{os.getpid()}-{uuid4().hex}"
    private.mkdir(parents=True, exist_ok=False)
    locations = (
        "YOLO_CONFIG_DIR MPLCONFIGDIR TORCH_HOME XDG_CACHE_HOME "
        "XDG_CONFIG_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR"
    )
    for name in locations.split():
        location = private / name.lower()
        location.mkdir()
        os.environ[name] = str(location)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import settings
    from ultralytics.models.sam.model import SAM

    settings.update(
        {
            "sync": False,
            "weights_dir": str(WEIGHTS),
            "runs_dir": str(private / "runs"),
            "datasets_dir": str(private / "datasets"),
            "clearml": False,
            "comet": False,
            "dvc": False,
            "mlflow": False,
            "raytune": False,
            "wandb": False,
            "tensorboard": False,
        }
    )
    path = WEIGHTS / "sam2.1_b.pt"
    manifest = WEIGHTS / "weights.json"
    if path.exists() != manifest.exists():
        raise SnapError("checkpoint_manifest_pair_required")
    if manifest.exists():
        record: dict[str, str] = json.loads(manifest.read_bytes())
        if record["filename"] != path.name or digest(path) != record["sha256"]:
            raise SnapError("checkpoint_sha256_mismatch")
    with chdir(WEIGHTS):
        result = SAM("sam2.1_b.pt")
    if not manifest.exists():
        record = {"filename": path.name, "sha256": digest(path)}
        _ = manifest.write_text(json.dumps(record, indent=2) + "\n")
    return result


def masks_from(predictions: Iterable[Results | Tensor]) -> tuple[Mask, ...]:
    """Narrow third-party result variants once at the SAM boundary."""
    from ultralytics.engine.results import Results

    masks: list[Mask] = []
    for prediction in predictions:
        if not isinstance(prediction, Results):
            raise SnapError("unexpected_tensor_prediction")
        if prediction.masks is not None:
            masks.extend(np.asarray(prediction.masks.cpu().data, dtype=np.bool_))
    return tuple(masks)


def content_box(image: Image) -> Box:
    """Exclude only edge-connected, near-black full rows and columns."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise SnapError("image_must_be_uint8_bgr")
    active = np.max(image, axis=2) > 12
    rows = np.flatnonzero(np.mean(active, axis=1) > 0.01)
    cols = np.flatnonzero(np.mean(active, axis=0) > 0.01)
    if not len(rows) or not len(cols):
        return (0.0, 0.0, 0.0, 0.0)
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def extent(mask: Mask) -> Box:
    """Return occupied pixel edges, with exclusive upper bounds."""
    ys, xs = np.nonzero(mask)
    left, right = xs.min().item(), xs.max().item() + 1
    top, bottom = ys.min().item(), ys.max().item() + 1
    return float(left), float(top), float(right), float(bottom)


def clip(box: Box, bounds: Box) -> Box:
    """Intersect boxes, collapsing disjoint intervals at the nearest boundary."""
    return (
        min(max(box[0], bounds[0]), bounds[2]),
        min(max(box[1], bounds[1]), bounds[3]),
        min(max(box[2], bounds[0]), bounds[2]),
        min(max(box[3], bounds[1]), bounds[3]),
    )


def area(box: Box) -> float:
    """Return a nonnegative exclusive-coordinate area."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def prompt(box: Box, bounds: Box) -> Box:
    """Pad each side by ten percent of the original box dimension."""
    dx, dy = (box[2] - box[0]) * 0.1, (box[3] - box[1]) * 0.1
    return clip((box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy), bounds)


def mask_result(mask: Mask, original: Box, padded: Box) -> SnapResult:
    """Apply connected-component, overlap, area and padded-border guards."""
    x1, y1 = ceil(padded[0]), ceil(padded[1])
    x2, y2 = floor(padded[2]), floor(padded[3])
    if x2 <= x1 or y2 <= y1 or not np.any(mask) or not area(original):
        return SnapResult(original, False, "empty_mask_or_prompt", 0.0)
    count, label_data, stat_data, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    labels = np.asarray(label_data, dtype=np.int32)
    stats = np.asarray(stat_data, dtype=np.int32)
    index = 1 + int(np.argmax(stats[1:count, cv2.CC_STAT_AREA]))
    raw: Mask = np.equal(labels, index)
    component = np.zeros_like(mask)
    component[y1:y2, x1:x2] = raw[y1:y2, x1:x2]
    if not np.any(component):
        return SnapResult(original, False, "empty_mask_or_prompt", 0.0)
    box = extent(component)
    ratio = float(np.count_nonzero(component)) / area(original)
    intersection = area(clip(box, original))
    iou = intersection / (area(box) + area(original) - intersection)
    left, top = floor(padded[0]), floor(padded[1])
    right, bottom = ceil(padded[2]), ceil(padded[3])
    edges = (
        raw[top:bottom, left],
        raw[top, left:right],
        raw[top:bottom, right - 1],
        raw[bottom - 1, left:right],
    )
    sides = sum(bool(np.any(edge)) for edge in edges)
    reason = "accepted"
    if iou < 0.5:
        reason = "iou_below_0.5"
    elif ratio < 0.3:
        reason = "mask_area_below_0.3"
    elif sides > 2:
        reason = "padded_border_more_than_two_sides"
    return SnapResult(
        box if reason == "accepted" else original, reason == "accepted", reason, ratio
    )


def snap_batch(image: Image, boxes: Sequence[Box]) -> tuple[SnapResult, ...]:
    """Batch all valid box prompts for one image on GPU; preserve input order."""
    bounds = content_box(image)
    clipped = [clip(box, bounds) for box in boxes]
    valid = [
        i
        for i, box in enumerate(clipped)
        if np.isfinite(boxes[i]).all() and area(box) > 0
    ]
    results = [
        SnapResult(box, False, "empty_or_degenerate_prompt", 0.0) for box in clipped
    ]
    if not valid:
        return tuple(results)
    padded = [prompt(clipped[i], bounds) for i in valid]
    predictions = model().predict(
        image,
        bboxes=padded,
        device=0,
        conf=0.0,
        verbose=False,
        save=False,
    )
    masks = masks_from(predictions)
    if len(masks) != len(valid):
        raise SnapError("prompt_mask_count_mismatch")
    for i, padding, mask in zip(valid, padded, masks, strict=True):
        results[i] = mask_result(mask, clipped[i], padding)
    return tuple(results)


def snap(image: Image, xyxy: Box) -> SnapResult:
    """Snap one box with the same guards as batched GPU inference."""
    return snap_batch(image, [xyxy])[0]


def snap_point(image: Image, xy: tuple[float, float]) -> SnapResult:
    """Segment a positive point; reject empty, outside or over-border masks."""
    bounds = content_box(image)
    x, y = xy
    fallback: Box = clip((x, y, x, y), bounds)
    if not (bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3]):
        return SnapResult(fallback, False, "point_outside_content", 0.0)
    predictions = model().predict(
        image,
        points=[xy],
        labels=[1],
        device=0,
        conf=0.0,
        verbose=False,
        save=False,
    )
    for mask in masks_from(predictions):
        if np.any(mask):
            box = extent(mask)
            return mask_result(mask, clip(box, bounds), bounds)
    return SnapResult(fallback, False, "empty_point_mask", 0.0)


def auto_masks(image: Image, region: Box) -> tuple[SnapResult, ...]:
    """Run SAM automatic generation on a content-clipped region; offset to image."""
    bounds = clip(region, content_box(image))
    x1, y1, x2, y2 = (int(v) for v in bounds)
    if x2 <= x1 or y2 <= y1:
        return ()
    crop = image[y1:y2, x1:x2]
    predictions = model().predict(crop, device=0, verbose=False, save=False)
    results: list[SnapResult] = []
    for raw in masks_from(predictions):
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        mask[y1:y2, x1:x2] = raw
        if np.any(mask):
            box = extent(mask)
            results.append(mask_result(mask, box, bounds))
    return tuple(results)


if __name__ == "__main__":
    if sys.argv[1:] != ["check"]:
        raise SnapError("usage: python -B -m public_label_rework.snap check")
    from importlib import import_module

    import_module("public_label_rework.smoke_snap").check()
