"""Fixed model and isolated runtime for the follow-up analysis."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, assert_never

from PIL.Image import Image

from scripts.baseline_common import ROOT, SOURCE, PipelineError, sha256, write_json
from scripts.run_baseline import pinned_environment

if TYPE_CHECKING:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

BASE: Final = ROOT / "postprocess_review"
OUTPUT: Final = BASE / "outputs"
DATASET: Final = ROOT / "stone_experiment/outputs/dataset"
WEIGHTS: Final = ROOT / "stone_experiment/outputs/selection/selected.pt"
NAMES: Final = ("potato", "stone")


@dataclass(frozen=True, slots=True)
class Postprocess:
    """Independent confidence and NMS overlap thresholds."""

    confidence: float
    nms_iou: float


def load_model() -> YOLO:
    """Use the pinned checkpoint without altering older runtime settings."""
    plan = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    if sha256(WEIGHTS) != plan["checkpoint_sha256"]:
        raise PipelineError("Checkpoint changed since protocol creation")
    runtime = OUTPUT / "runtime" / f"process-{os.getpid()}"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime)
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    os.environ["TORCH_HOME"] = str(runtime / "torch")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    pinned_environment()
    from ultralytics import YOLO, settings

    settings.update({"sync": False, "runs_dir": str(runtime / "runs"),
                     "weights_dir": str(runtime / "weights")})
    model = YOLO(str(WEIGHTS))
    if model.names != {0: "potato", 1: "stone"}:
        raise PipelineError("Expected the frozen potato/stone model")
    return model


def infer(model: YOLO, source: str | Image, parameters: Postprocess) -> Results:
    """Rerun actual model NMS, never approximate it from already suppressed boxes."""
    from torch import Tensor
    from ultralytics.engine.results import Results

    prediction = next(iter(model.predict(
        source, imgsz=640, device=0, conf=parameters.confidence,
        iou=parameters.nms_iou, agnostic_nms=False,
        save=False, save_txt=False, stream=False, verbose=False,
    )))
    match prediction:
        case Results():
            if prediction.boxes is None:
                raise PipelineError("Missing detection boxes")
            return prediction
        case Tensor():
            raise PipelineError("Unexpected embedding output")
        case unexpected:
            assert_never(unexpected)


__all__ = [
    "BASE", "DATASET", "NAMES", "OUTPUT", "ROOT", "SOURCE", "WEIGHTS",
    "PipelineError", "Postprocess", "infer", "load_model", "sha256", "write_json",
]
