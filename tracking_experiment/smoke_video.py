"""Real CPU CLI/encoding proofs and strict cache-boundary rejection fixtures."""

import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import av

from retraining_data.schema import DataError, array, document, mapping, require
from retraining_experiment.common import PROTOCOL
from retraining_experiment.video import (
    ENCODER_OPTIONS,
    TIME_BASE,
    preserve_packet_duration,
)
from scripts.baseline_common import ROOT, PipelineError, sha256, write_json
from stone_experiment.metrics import Detection
from tracking_experiment.cache import Inputs, load, new_output


def rejected[T](action: Callable[[], T], error: type[Exception] = DataError) -> None:
    try:
        _ = action()
    except error:
        return
    raise DataError(f"Expected {error.__name__}")


def fixture(root: Path) -> Inputs:
    """Encode four original-size black frames with unequal and final VFR intervals."""
    source = root / "source.mp4"
    durations = {0: 7500, 7500: 3750, 11250: 7500, 18750: 7500}
    with av.open(str(source), "w") as target:
        stream = target.add_stream("libx264")
        stream.width, stream.height = 1280, 720
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = TIME_BASE
        stream.options = dict(ENCODER_OPTIONS)
        for pts, duration in durations.items():
            frame = av.VideoFrame(1280, 720, "rgb24")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts, frame.time_base, frame.duration = pts, TIME_BASE, duration
            for packet in stream.encode(frame):
                preserve_packet_duration(packet, durations)
                target.mux(packet)
        for packet in stream.encode():
            preserve_packet_duration(packet, durations)
            target.mux(packet)
    digest = sha256(source)
    cache = root / "counts.json"
    write_json(
        cache,
        {
            **PROTOCOL,
            "meaning": "per_frame_on_screen_not_unique_objects",
            "source_sha256": digest,
            "checkpoint_sha256": digest,
            "video_sha256": digest,
            "stream_time_base": str(TIME_BASE),
            "frames": [
                {
                    "frame": index,
                    "pts": pts,
                    "time_base": str(TIME_BASE),
                    "duration": duration,
                    "second": float(pts * TIME_BASE),
                    "potato": 1,
                    "stone": 0,
                    "raw_detections": [
                        asdict(Detection(0, (480, y - 20, 520, y + 20), 0.9))
                    ],
                }
                for index, ((pts, duration), y) in enumerate(
                    zip(durations.items(), (400, 370, 350, 330), strict=True)
                )
            ],
        },
    )
    # The checkpoint is opaque hash-only input; no weights parser is involved.
    return Inputs(source, cache, source, source)


def cache_rejection_proof(inputs: Inputs) -> None:
    # Given: a valid cache and byte receipts for each required input.
    original = inputs.cache.read_bytes()
    require(len(load(inputs).frames) == 4, "Fixture cache parses")
    # When/Then: each tampered input receipt is refused.
    for key in ("source_sha256", "checkpoint_sha256", "video_sha256"):
        data = document(original)
        data[key] = "0" * 64
        write_json(inputs.cache, data)
        rejected(lambda: load(inputs))
    # When/Then: invalid duration, class totals and boxes are refused independently.
    for key, bad in (("duration", 0), ("potato", 2), ("pts", 1), ("second", 7.0)):
        data = document(original)
        first = mapping(array(data["frames"])[0])
        first[key] = bad
        write_json(inputs.cache, data)
        rejected(lambda: load(inputs))
    data = document(original)
    first = mapping(array(data["frames"])[0])
    detection = mapping(array(first["raw_detections"])[0])
    detection["xyxy"] = [-1, 20, 100, 80]
    write_json(inputs.cache, data)
    rejected(lambda: load(inputs), PipelineError)
    write_json(inputs.cache, document(original))


def output_rejection_proof(root: Path) -> None:
    # Given: one successfully reserved output.
    existing = new_output(root / "reserved")
    # When/Then: existing paths, escaped paths and symlink aliases are refused.
    rejected(lambda: new_output(existing), FileExistsError)
    rejected(lambda: new_output(ROOT / "tracking_experiment"))
    link = root / "alias"
    link.symlink_to(existing, target_is_directory=True)
    rejected(lambda: new_output(link / "nested"))


def integration_proof(root: Path) -> None:
    # Given: actual source frames, valid cached boxes, and a new owned destination.
    inputs = fixture(root)
    cache_rejection_proof(inputs)
    output_rejection_proof(root)
    output = root / "render"
    command = [
        sys.executable,
        "-B",
        "-m",
        "tracking_experiment.video",
        "--source",
        str(inputs.source),
        "--cache",
        str(inputs.cache),
        "--weights",
        str(inputs.weights),
        "--cached-video",
        str(inputs.cached_video),
        "--output",
        str(output),
    ]
    # When: the real public CLI loads cached detections and renders on CPU.
    result = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    require(result.returncode == 0, "Real render CLI")
    # Then: exported evidence and decoded source timing agree, including final interval.
    events = array(document((output / "events.json").read_bytes())["events"])
    require(
        len(events) == 1 and mapping(events[0])["pts"] == 11250, "Real crossing JSON"
    )
    rows = array(document((output / "tracks.json").read_bytes())["frames"])
    require(
        [mapping(row)["cumulative_crossings"] for row in rows] == [0, 0, 1, 1],
        "Per-frame cumulative output",
    )
    with av.open(str(output / "annotated.mp4")) as video:
        actual = [(f.pts, f.duration) for f in video.decode(video=0)]
    require(
        actual == [(0, 7500), (7500, 3750), (11250, 7500), (18750, 7500)],
        "Exact four-frame VFR output",
    )
    summary = document((output / "summary.json").read_bytes())
    require(
        summary["cumulative_crossings"] == 1 and summary["frames"] == 4,
        "Summary matches exported actual output",
    )
    for module in ("tracking_experiment.video", "tracking_experiment.cli"):
        _ = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    f"import {module}; import sys; "
                    + "assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules"
                ),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    _ = subprocess.run(
        [sys.executable, "-B", "-m", "tracking_experiment.video", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
