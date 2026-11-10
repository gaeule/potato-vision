"""Typed JSON boundaries for chief annotations and baseline provenance."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from stone_experiment.common import SOURCE, PipelineError

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Box = tuple[float, float, float, float]
RANGES: Final = {"train": (0, 84), "val": (92, 112), "test": (120, 147)}
PROVENANCE: Final = "agent_visual_review_not_human_expert"


def document(path: Path) -> dict[str, Json]:
    """Decode a JSON object; all consumers narrow values before using them."""
    value: Json = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PipelineError(f"Expected JSON object: {path}")
    return value


def rows(value: Json) -> list[dict[str, Json]]:
    """Parse a list of JSON objects."""
    if not isinstance(value, list):
        raise PipelineError("Expected an array of objects")
    result: list[dict[str, Json]] = []
    for row in value:
        if not isinstance(row, dict):
            raise PipelineError("Expected object in array")
        result.append(row)
    return result


def integer(value: Json) -> int:
    """Accept JSON integers but not booleans."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise PipelineError(f"Expected integer: {value}")
    return value


def text(value: Json) -> str:
    """Parse a JSON string."""
    if not isinstance(value, str):
        raise PipelineError(f"Expected string: {value}")
    return value


def boxes(value: Json, width: int = 1280, height: int = 720) -> tuple[Box, ...]:
    """Parse finite, nondegenerate pixel xyxy boxes inside the image."""
    if not isinstance(value, list):
        raise PipelineError("Expected box array")
    parsed: list[Box] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise PipelineError(f"Expected four coordinates: {row}")
        coordinates: list[float] = []
        for number in row:
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
                raise PipelineError(f"Expected finite coordinate: {number}")
            coordinates.append(float(number))
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise PipelineError(f"Box outside image: {row}")
        parsed.append((x1, y1, x2, y2))
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class Frame:
    """One independently counted frame, not a unique-object observation."""

    second: int
    split: str
    boxes: tuple[Box, ...]
    stones: tuple[Box, ...]

    @property
    def filename(self) -> str:
        return f"frame_{self.second:06d}.png"


def annotations(path: Path, *, baseline: bool = False) -> tuple[Frame, ...]:
    """Parse the fixed annotation contract and temporal isolation boundaries."""
    data = document(path)
    if (data.get("source"), data.get("width"), data.get("height"), data.get("provenance")) != (
        SOURCE.name, 1280, 720, PROVENANCE,
    ):
        raise PipelineError(f"Unexpected source, size, or provenance: {path}")
    parsed: list[Frame] = []
    seen: set[int] = set()
    for row in rows(data["frames"]):
        second, split = integer(row["second"]), text(row["split"])
        if split not in RANGES or second in seen or not RANGES[split][0] <= second <= RANGES[split][1]:
            raise PipelineError(f"Duplicate second or invalid temporal split: {row}")
        seen.add(second)
        stones = () if baseline else boxes(row["stones"])
        parsed.append(Frame(second, split, boxes(row["boxes"]), stones))
    return tuple(sorted(parsed, key=lambda frame: frame.second))
