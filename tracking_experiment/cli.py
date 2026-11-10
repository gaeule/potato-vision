"""Stdlib command boundary; help does not load a model or a framework runtime."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tracking_experiment.cache import OUTPUTS, Inputs
from tracking_experiment.model import Direction, Settings


@dataclass(init=False, repr=False, eq=False)
class Parsed(argparse.Namespace):
    """Mutable argparse boundary with explicitly declared parsed field types."""

    source: Path
    cache: Path
    weights: Path
    cached_video: Path
    output: Path
    line_y: float
    x_min: float
    x_max: float
    direction: Direction
    hysteresis: float
    max_gap: float
    max_distance: float
    min_hits: int


@dataclass(frozen=True, slots=True)
class Options:
    inputs: Inputs
    settings: Settings
    output: Path


def arguments(argv: Sequence[str] | None = None) -> Options:
    """Parse configurable line geometry without changing frozen detector parameters."""
    parser = argparse.ArgumentParser(
        description="Replay cached potato IDs and line crossings"
    )
    inputs, settings = Inputs(), Settings()
    for name, default in (
        ("source", inputs.source),
        ("cache", inputs.cache),
        ("weights", inputs.weights),
        ("cached-video", inputs.cached_video),
        ("output", OUTPUTS / "video"),
    ):
        _ = parser.add_argument(f"--{name}", type=Path, default=default)
    for name, default in (
        ("line-y", settings.line_y),
        ("x-min", settings.x_min),
        ("x-max", settings.x_max),
        ("hysteresis", settings.hysteresis),
        ("max-gap", settings.max_gap),
        ("max-distance", settings.max_distance),
    ):
        _ = parser.add_argument(f"--{name}", type=float, default=default)
    _ = parser.add_argument(
        "--direction", choices=("up", "down"), default=settings.direction
    )
    _ = parser.add_argument("--min-hits", type=int, default=settings.min_hits)
    parsed = parser.parse_args(argv, namespace=Parsed())
    return Options(
        Inputs(parsed.source, parsed.cache, parsed.weights, parsed.cached_video),
        Settings(
            parsed.line_y,
            parsed.x_min,
            parsed.x_max,
            parsed.direction,
            parsed.hysteresis,
            parsed.max_gap,
            parsed.max_distance,
            parsed.min_hits,
        ),
        parsed.output,
    )
