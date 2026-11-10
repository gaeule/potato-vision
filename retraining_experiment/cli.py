"""Repository-style stdlib CLI; no optional packages or GPU imports on --help."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from retraining_experiment.common import BASELINE, DATASET, OUTPUTS, SELECTED

type Command = Literal["train", "select", "evaluate", "video"]


@dataclass(init=False, repr=False, eq=False)
class Options(argparse.Namespace):
    """Mutable fields initialized by each command's parser, not shared defaults."""

    dataset: Path
    output: Path
    training: Path
    weights: Path
    baseline: Path


def arguments(command: Command) -> Options:
    """Parse one executable module without importing its implementation back."""
    parser = argparse.ArgumentParser(description=f"Fixed v2 retraining: {command}")
    outputs = {
        "train": "train",
        "select": "selection",
        "evaluate": "comparison",
        "video": "video",
    }
    _ = parser.add_argument("--output", type=Path, default=OUTPUTS / outputs[command])
    fields = {
        "train": (("dataset", DATASET),),
        "select": (("dataset", DATASET), ("training", OUTPUTS / "train")),
        "evaluate": (
            ("dataset", DATASET),
            ("weights", SELECTED),
            ("baseline", BASELINE),
        ),
        "video": (("weights", SELECTED),),
    }
    for name, default in fields[command]:
        _ = parser.add_argument(f"--{name}", type=Path, default=default)
    return parser.parse_args(namespace=Options())
