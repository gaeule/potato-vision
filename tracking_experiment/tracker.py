"""Deterministic gated motion assignment and once-per-ID line crossings."""

from dataclasses import dataclass
from math import hypot

from scipy.optimize import linear_sum_assignment

from retraining_data.schema import require
from stone_experiment.metrics import Detection, iou
from stone_experiment.schema import Box
from tracking_experiment.model import (
    Event,
    Frame,
    Observation,
    Settings,
    TrackId,
    center,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    x: float
    second: float
    previous_pts: int
    next_pts: int
    gap: float


@dataclass(frozen=True, slots=True)
class Track:
    id: TrackId
    box: Box
    pts: int
    second: float
    velocity: tuple[float, float] = (0, 0)
    hits: int = 1
    armed: bool = False
    counted: bool = False
    candidate: Candidate | None = None


class Tracker:
    """Mutable accumulator; only actual matched observations update crossing state."""

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        self.active: list[Track] = []
        self.created: int = 0
        self.last_second: float | None = None

    def cost(self, track: Track, frame: Frame, detection: Detection) -> float:
        """Gate predicted centers and box scale before global one-to-one assignment."""
        dt = float((frame.pts - track.pts) * frame.time_base)
        dx, dy = (v * dt for v in track.velocity)
        predicted = (
            track.box[0] + dx,
            track.box[1] + dy,
            track.box[2] + dx,
            track.box[3] + dy,
        )
        px, py = center(predicted)
        x, y = center(detection.xyxy)
        distance = hypot(px - x, py - y)
        area = (track.box[2] - track.box[0]) * (track.box[3] - track.box[1])
        box = detection.xyxy
        ratio = (box[2] - box[0]) * (box[3] - box[1]) / area
        if distance > self.settings.max_distance or not 0.25 <= ratio <= 4:
            return 1_000_000
        return distance / self.settings.max_distance + 1 - iou(predicted, box)

    def advance(
        self, track: Track, frame: Frame, detection: Detection
    ) -> tuple[Track, Event | None]:
        """Observe hysteresis and finite-line intersection before emitting an event."""
        settings = self.settings
        x0, y0 = center(track.box)
        x, y = center(detection.xyxy)
        dt = float((frame.pts - track.pts) * frame.time_base)
        measured = ((x - x0) / dt, (y - y0) / dt)
        velocity = (
            measured
            if track.hits == 1
            else (
                (track.velocity[0] + measured[0]) / 2,
                (track.velocity[1] + measured[1]) / 2,
            )
        )
        before, after = settings.distance(y0), settings.distance(y)
        armed, counted, candidate = track.armed, track.counted, track.candidate
        hits = track.hits + 1
        event: Event | None = None
        if after < 0:
            candidate = None
        if after <= -settings.hysteresis:
            armed = True
        if armed and before < 0 <= after:
            fraction = -before / (after - before)
            candidate = Candidate(
                x0 + fraction * (x - x0),
                track.second + fraction * dt,
                track.pts,
                frame.pts,
                dt,
            )
        if armed and after >= settings.hysteresis and candidate is not None:
            inside = settings.x_min <= candidate.x <= settings.x_max
            if inside and hits >= settings.min_hits and not counted:
                event = Event(
                    track.id,
                    frame.frame,
                    frame.pts,
                    str(frame.time_base),
                    frame.second,
                    settings.direction,
                    candidate.x,
                    candidate.second,
                    candidate.previous_pts,
                    candidate.next_pts,
                    candidate.gap,
                )
                counted = True
            if counted or not inside:
                armed, candidate = False, None
        updated = Track(
            track.id,
            detection.xyxy,
            frame.pts,
            frame.second,
            velocity,
            hits,
            armed,
            counted,
            candidate,
        )
        return updated, event

    def update(self, frame: Frame) -> tuple[tuple[Observation, ...], tuple[Event, ...]]:
        """Canonical ordering fixes assignment ties; dummy columns allow unmatched IDs."""
        require(
            self.last_second is None or frame.second > self.last_second,
            "Tracker requires strictly increasing source times",
        )
        self.last_second = frame.second
        settings = self.settings
        active = sorted(
            (
                t
                for t in self.active
                if float((frame.pts - t.pts) * frame.time_base) <= settings.max_gap
            ),
            key=lambda t: t.id,
        )
        detections = sorted(
            (d for d in frame.detections if d.class_id == 0),
            key=lambda d: (*d.xyxy, -d.confidence),
        )
        assignments: dict[int, int] = {}
        if active and detections:
            costs = [
                [self.cost(t, frame, d) for d in detections] + [10.0] * len(active)
                for t in active
            ]
            rows, columns = linear_sum_assignment(costs)
            assignments = {
                int(row): int(column)
                for row, column in zip(rows, columns, strict=True)
                if column < len(detections) and costs[int(row)][int(column)] < 10
            }
        observations: list[Observation] = []
        events: list[Event] = []
        retained: list[Track] = []
        used = set(assignments.values())
        for index, track in enumerate(active):
            if index not in assignments:
                retained.append(track)
                continue
            detection = detections[assignments[index]]
            updated, event = self.advance(track, frame, detection)
            retained.append(updated)
            if event is not None:
                events.append(event)
            observations.append(
                Observation(
                    updated.id,
                    updated.box,
                    detection.confidence,
                    updated.hits,
                    updated.hits >= settings.min_hits,
                    updated.counted,
                )
            )
        for index, detection in enumerate(detections):
            if index in used:
                continue
            self.created += 1
            track = Track(
                TrackId(self.created),
                detection.xyxy,
                frame.pts,
                frame.second,
                armed=settings.distance(center(detection.xyxy)[1])
                <= -settings.hysteresis,
            )
            retained.append(track)
            observations.append(
                Observation(
                    track.id,
                    track.box,
                    detection.confidence,
                    1,
                    False,
                    False,
                )
            )
        self.active = sorted(retained, key=lambda t: t.id)
        return (
            tuple(sorted(observations, key=lambda o: o.track_id)),
            tuple(events),
        )
