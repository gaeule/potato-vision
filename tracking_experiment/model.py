"""Immutable tracking configuration and source-clock observations."""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Literal, NewType

from retraining_data.schema import require
from stone_experiment.metrics import Detection
from stone_experiment.schema import Box

type Direction = Literal["up", "down"]
TrackId = NewType("TrackId", int)


@dataclass(frozen=True, slots=True)
class Settings:
    """Finite horizontal counting line and time-based association policy."""

    line_y: float = 360
    x_min: float = 220
    x_max: float = 1000
    direction: Direction = "up"
    hysteresis: float = 8
    max_gap: float = 0.5
    max_distance: float = 100
    min_hits: int = 2

    def __post_init__(self) -> None:
        require(
            all(
                isfinite(v)
                for v in (
                    self.line_y,
                    self.x_min,
                    self.x_max,
                    self.hysteresis,
                    self.max_gap,
                    self.max_distance,
                )
            ),
            "Tracking parameters must be finite",
        )
        require(0 <= self.x_min < self.x_max, "Expected x_min < x_max")
        require(self.hysteresis > 0 and self.max_gap > 0, "Positive gap/hysteresis")
        require(self.max_distance > 0 and self.min_hits >= 2, "Distance/hit limits")

    def distance(self, y: float) -> float:
        """Positive means downstream in the requested direction."""
        return {"up": 1, "down": -1}[self.direction] * (self.line_y - y)


@dataclass(frozen=True, slots=True)
class Frame:
    frame: int
    pts: int
    time_base: Fraction
    duration: int
    detections: tuple[Detection, ...]

    @property
    def second(self) -> float:
        return float(self.pts * self.time_base)


@dataclass(frozen=True, slots=True)
class Observation:
    """An observed box, never an unobserved motion prediction."""

    track_id: TrackId
    xyxy: Box
    confidence: float
    hits: int
    confirmed: bool
    counted: bool


@dataclass(frozen=True, slots=True)
class Event:
    """PTS is the confirming source frame, not a fabricated crossing frame."""

    track_id: TrackId
    frame: int
    pts: int
    time_base: str
    second: float
    direction: Direction
    crossing_x: float
    crossing_second_estimate: float
    crossing_previous_pts: int
    crossing_next_pts: int
    observation_gap_seconds: float


def center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
