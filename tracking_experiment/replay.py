"""Pure replay of every cached source observation into auditable tracking rows."""

from dataclasses import dataclass

from tracking_experiment.cache import Cache
from tracking_experiment.model import Event, Observation, Settings, TrackId
from tracking_experiment.tracker import Tracker


@dataclass(frozen=True, slots=True)
class FrameTracks:
    frame: int
    pts: int
    time_base: str
    duration: int
    second: float
    on_screen_potato: int
    on_screen_stone: int
    cumulative_crossings: int
    tracks: tuple[Observation, ...]
    retained_unobserved_ids: tuple[TrackId, ...]


@dataclass(frozen=True, slots=True)
class Replay:
    frames: tuple[FrameTracks, ...]
    events: tuple[Event, ...]
    created: int


def replay(cache: Cache, settings: Settings) -> Replay:
    """Do not count predicted boxes during gaps or call a detector."""
    tracker = Tracker(settings)
    rows: list[FrameTracks] = []
    events: list[Event] = []
    for frame in cache.frames:
        observed, crossings = tracker.update(frame)
        events.extend(crossings)
        ids = {o.track_id for o in observed}
        rows.append(
            FrameTracks(
                frame.frame,
                frame.pts,
                str(frame.time_base),
                frame.duration,
                frame.second,
                sum(d.class_id == 0 for d in frame.detections),
                sum(d.class_id == 1 for d in frame.detections),
                len(events),
                observed,
                tuple(t.id for t in tracker.active if t.id not in ids),
            )
        )
    return Replay(tuple(rows), tuple(events), tracker.created)
