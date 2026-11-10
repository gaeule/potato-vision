"""Fixed protocol, new-only outputs and lazy, process-private framework runtime."""

import os
import sys
from collections.abc import Generator
from contextlib import chdir, contextmanager
from pathlib import Path
from typing import Final
from uuid import uuid4

from postprocess_review.common import Postprocess
from retraining_data.common import digest
from retraining_data.schema import require
from scripts.baseline_common import ROOT, SOURCE, sha256, write_json
from scripts.run_baseline import environment, pinned_environment
from stone_experiment.schema import document, text

__all__ = [
    "BASELINE",
    "DATASET",
    "OUTPUTS",
    "PARAMETERS",
    "PROTOCOL",
    "ROOT",
    "SELECTED",
    "SOURCE",
    "check_frozen",
    "check_selected",
    "new_output",
    "record_context",
    "runtime",
    "sha256",
    "write_json",
]

OUTPUTS: Final = ROOT / "retraining_experiment/outputs"
DATASET: Final = ROOT / "retraining_data/outputs/dataset_v2"
BASELINE: Final = ROOT / "stone_experiment/outputs/selection/selected.pt"
SELECTED: Final = OUTPUTS / "selection/selected.pt"
PARAMETERS: Final = Postprocess(0.45, 0.5)
PROTOCOL: Final = {
    "confidence": 0.45,
    "nms_iou": 0.5,
    "matching_iou": 0.5,
    "matching": "confidence_ranked_class_agnostic_one_to_one",
    "count_unit": "frame_observations_not_unique_objects",
}


def check_frozen(path: Path, field: str) -> None:
    """Bind an input to the chief-owned, predeclared protocol without editing it."""
    plan = document(ROOT / "retraining_experiment/review/protocol.json")
    require(
        all(plan[k] == PROTOCOL[k] for k in ("confidence", "nms_iou", "matching_iou")),
        "Frozen operational parameters differ",
    )
    require(sha256(path) == text(plan[field]), f"Frozen input hash differs: {path}")


def new_output(path: Path) -> Path:
    """Reserve an unused real directory strictly below this package's outputs."""
    path = path.absolute()
    require(OUTPUTS.resolve() == OUTPUTS, "Outputs root must not be a symlink")
    resolved = path.resolve()
    require(
        resolved.is_relative_to(OUTPUTS) and resolved != OUTPUTS,
        f"Output must be below {OUTPUTS}",
    )
    require(path == resolved, "Output symlink or noncanonical path refused")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def check_selected(weights: Path) -> None:
    """Bind inference to the frozen old model or the validation-selected new one."""
    if weights.resolve() == BASELINE.resolve():
        check_frozen(weights, "baseline_weights_sha256")
        return
    selection = document(weights.parent / "selection.json")
    require(selection["selection_data"] == "val11_only", "Expected val-only selection")
    require(
        sha256(weights) == text(selection["selected_sha256"]),
        "Selected checkpoint hash differs",
    )


@contextmanager
def runtime(output: Path) -> Generator[None]:
    """Set every mutable location before importing Ultralytics, including CWD."""
    require("ultralytics" not in sys.modules, "Use a fresh process for each command")
    pinned_environment()
    private = output / "runtime" / f"process-{os.getpid()}-{uuid4().hex}"
    private.mkdir(parents=True, exist_ok=False)
    locations = {
        "YOLO_CONFIG_DIR": private / "yolo",
        "MPLCONFIGDIR": private / "matplotlib",
        "TORCH_HOME": private / "torch",
        "XDG_CACHE_HOME": private / "cache",
        "XDG_CONFIG_HOME": private / "config",
        "TORCHINDUCTOR_CACHE_DIR": private / "inductor",
        "TRITON_CACHE_DIR": private / "triton",
    }
    for name, location in locations.items():
        location.mkdir()
        os.environ[name] = str(location)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    with chdir(private):
        from ultralytics import settings

        settings.update(
            {
                "sync": False,
                "weights_dir": str(private / "weights"),
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
        yield


def record_context(output: Path, weights: tuple[Path, ...] = ()) -> None:
    """Record exact source, config, environment, checkpoint and source-video bytes."""
    sources = [
        *sorted((ROOT / "retraining_experiment").glob("*.py")),
        *sorted((ROOT / "retraining_data").glob("*.py")),
        *sorted((ROOT / "stone_experiment").glob("*.py")),
        ROOT / "postprocess_review/common.py",
        ROOT / "scripts/baseline_common.py",
        ROOT / "scripts/run_baseline.py",
        ROOT / "baseline/config/train.json",
        ROOT / "stone_experiment/config/train.json",
        ROOT / "retraining_experiment/review/protocol.json",
        ROOT / "retraining_experiment/review/baseline_reference.json",
        ROOT / "retraining_experiment/review/protected_before.json",
    ]
    write_json(output / "environment.json", environment())
    write_json(
        output / "provenance.json",
        {
            "protocol": PROTOCOL,
            "source_video_sha256": sha256(SOURCE),
            "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
            "checkpoints": {str(p): sha256(p) for p in weights},
            "environment_sha256": sha256(output / "environment.json"),
            "python_executable": sys.executable,
            "python_executable_sha256": sha256(Path(sys.executable).resolve()),
            "protocol_sha256": digest(repr(sorted(PROTOCOL.items())).encode()),
        },
    )
