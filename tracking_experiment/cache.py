"""Strict cached-observation boundary and immutable input-byte receipts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from retraining_data.schema import (
    array,
    document,
    integer,
    mapping,
    number,
    require,
    text,
)
from retraining_experiment.common import PROTOCOL, SELECTED
from retraining_experiment.video import TIME_BASE
from scripts.baseline_common import ROOT, SOURCE, sha256
from stone_experiment.metrics import Detection
from stone_experiment.schema import boxes
from tracking_experiment.model import Frame

OUTPUTS: Final = ROOT / "tracking_experiment/outputs"


@dataclass(frozen=True, slots=True)
class Inputs:
    source: Path = SOURCE
    cache: Path = ROOT / "retraining_experiment/outputs/video/counts.json"
    weights: Path = SELECTED
    cached_video: Path = ROOT / "retraining_experiment/outputs/video/annotated.mp4"


@dataclass(frozen=True, slots=True)
class Hashes:
    source_sha256: str
    checkpoint_sha256: str
    cached_video_sha256: str
    cache_sha256: str


@dataclass(frozen=True, slots=True)
class Cache:
    frames: tuple[Frame, ...]
    hashes: Hashes


def load(inputs: Inputs) -> Cache:
    """Verify all three referenced files and parse every consumed schema field."""
    data = document(inputs.cache.read_bytes())
    required = {
        "source_sha256",
        "checkpoint_sha256",
        "video_sha256",
        "frames",
        "stream_time_base",
        "meaning",
        *PROTOCOL,
    }
    require(required <= data.keys(), "Incomplete cached counts schema")
    for key, value in PROTOCOL.items():
        require(data[key] == value, f"Cached detector protocol differs: {key}")
    require(
        data["stream_time_base"] == str(TIME_BASE)
        and data["meaning"] == "per_frame_on_screen_not_unique_objects",
        "Expected original observation cache and 90 kHz clock",
    )
    hashes = Hashes(
        sha256(inputs.source),
        sha256(inputs.weights),
        sha256(inputs.cached_video),
        sha256(inputs.cache),
    )
    for key, actual in (
        ("source_sha256", hashes.source_sha256),
        ("checkpoint_sha256", hashes.checkpoint_sha256),
        ("video_sha256", hashes.cached_video_sha256),
    ):
        require(text(data[key]) == actual, f"Cached input hash differs: {key}")
    frames: list[Frame] = []
    for item in array(data["frames"]):
        row = mapping(item)
        require(
            {
                "frame",
                "pts",
                "time_base",
                "duration",
                "second",
                "potato",
                "stone",
                "raw_detections",
            }
            <= row.keys(),
            "Incomplete frame schema",
        )
        index, pts, duration = (integer(row[k]) for k in ("frame", "pts", "duration"))
        require(index == len(frames), "Cache frame indices must be consecutive")
        require(pts >= 0 and duration > 0, "Invalid source PTS/duration")
        require(text(row["time_base"]) == str(TIME_BASE), "Changed frame clock")
        require(
            abs(number(row["second"]) - float(pts * TIME_BASE)) < 1e-9,
            "Frame second differs from source PTS",
        )
        if frames:
            previous = frames[-1]
            require(
                pts == previous.pts + previous.duration,
                "Cache PTS/duration sequence differs",
            )
        detections: list[Detection] = []
        for raw in array(row["raw_detections"]):
            detection = mapping(raw)
            require(
                {"class_id", "xyxy", "confidence"} <= detection.keys(),
                "Incomplete detection schema",
            )
            class_id = integer(detection["class_id"])
            confidence = number(detection["confidence"])
            require(class_id in (0, 1), "Unknown detection class")
            require(0.45 <= confidence <= 1, "Invalid cached detection confidence")
            box = boxes([detection["xyxy"]])[0]
            detections.append(Detection(class_id, box, confidence))
        for class_id, key in enumerate(("potato", "stone")):
            require(
                integer(row[key]) == sum(d.class_id == class_id for d in detections),
                f"Cached class total differs: {key}",
            )
        frames.append(Frame(index, pts, TIME_BASE, duration, tuple(detections)))
    require(bool(frames) and frames[0].pts == 0, "Expected complete source cache")
    return Cache(tuple(frames), hashes)


def new_output(path: Path) -> Path:
    """Reserve a new real output directory; never reuse existing artifacts."""
    absolute = path.absolute()
    resolved = absolute.resolve()
    require(OUTPUTS.resolve() == OUTPUTS, "Output root must not be a symlink")
    require(
        absolute == resolved
        and resolved.is_relative_to(OUTPUTS)
        and resolved != OUTPUTS,
        f"Output must be strictly below {OUTPUTS}, without symlinks",
    )
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved
