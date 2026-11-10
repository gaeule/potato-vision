"""Narrow Ultralytics detection results at the inference boundary."""

from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from stone_experiment.common import NAMES, PipelineError

if TYPE_CHECKING:
    from PIL.Image import Image
    from ultralytics import YOLO
    from ultralytics.engine.results import Results


def check_classes(model: "YOLO", *, two_classes: bool = False) -> None:
    """Reject COCO or reordered class mappings before any inference."""
    expected = dict(enumerate(NAMES[:len(model.names)]))
    if model.names != expected or len(expected) not in (1, 2):
        raise PipelineError(f"Expected class 0 potato, optional class 1 stone: {model.names}")
    if two_classes and len(expected) != 2:
        raise PipelineError("The new experiment requires a two-class checkpoint")


def predict(model: "YOLO", source: "str | Path | Image", confidence: float) -> "Results":
    """Run fixed-size, fixed-NMS detection without writing prediction caches."""
    from torch import Tensor
    from ultralytics.engine.results import Results

    prediction = next(iter(model.predict(
        source=source, imgsz=640, conf=confidence, iou=0.7, device=0,
        save=False, save_txt=False, verbose=False, stream=False,
    )))
    match prediction:
        case Results():
            if prediction.boxes is None:
                raise PipelineError("Detection result has no box container")
            return prediction
        case Tensor():
            raise PipelineError("Unexpected embedding tensor instead of detections")
        case unexpected:
            assert_never(unexpected)
