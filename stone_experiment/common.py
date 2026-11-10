"""Read-only baseline helpers and strictly new experiment outputs."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Reused baseline modules must never create bytecode in their source tree.
sys.dont_write_bytecode = True

from scripts.baseline_common import (
    BASE,
    ROOT,
    SOURCE,
    PipelineError,
    sha256,
    write_json,
)
from scripts.run_baseline import environment, pinned_environment

EXPERIMENT: Final = ROOT / "stone_experiment"
OUTPUTS: Final = EXPERIMENT / "outputs"
ANNOTATIONS: Final = EXPERIMENT / "annotations.json"
DATASET: Final = OUTPUTS / "dataset"
CONFIDENCE: Final = 0.45
NMS_IOU: Final = 0.7
MATCH_IOU: Final = 0.5
NAMES: Final = ("potato", "stone")


@dataclass(init=False, repr=False, eq=False)
class Options(argparse.Namespace):
    """Mutable command fields initialized externally by argparse, never by defaults."""

    weights: Path
    baseline: Path
    confidence: float
    output: Path
    dataset: Path
    annotations: Path
    training: Path
    split: str
    external: bool


def new_output(path: Path) -> Path:
    """Atomically reserve a never-used directory inside experiment outputs."""
    resolved = path.resolve()
    if not resolved.is_relative_to(OUTPUTS.resolve()) or resolved == OUTPUTS.resolve():
        raise PipelineError(f"Output must be a new subdirectory of {OUTPUTS}: {path}")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def runtime() -> None:
    """Isolate mutable library settings between concurrent experiment processes."""
    config = OUTPUTS / "runtime" / f"process-{os.getpid()}"
    config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["MPLCONFIGDIR"] = str(config / "matplotlib")
    os.environ["TORCH_HOME"] = str(config / "torch")
    os.environ["XDG_CACHE_HOME"] = str(config / "cache")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    pinned_environment()
    from ultralytics import settings

    settings.update({
        "sync": False,
        "weights_dir": str(config / "weights"),
        "runs_dir": str(config / "runs"),
        "datasets_dir": str(config / "datasets"),
    })


__all__ = [
    "ANNOTATIONS", "BASE", "CONFIDENCE", "DATASET", "EXPERIMENT", "MATCH_IOU",
    "NAMES", "NMS_IOU", "OUTPUTS", "ROOT", "SOURCE", "Options", "PipelineError",
    "environment", "new_output", "runtime", "sha256", "write_json",
]
