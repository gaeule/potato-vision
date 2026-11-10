"""CPU-only cached replay on original source pixels with exact VFR preservation."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retraining_data.schema import require
from retraining_experiment.video import (
    ENCODER_OPTIONS,
    TIME_BASE,
    preserve_packet_duration,
)
from scripts.baseline_common import ROOT, sha256, write_json
from tracking_experiment.cache import Inputs, load, new_output
from tracking_experiment.model import Settings
from tracking_experiment.replay import FrameTracks, replay


def overlay(image: Image.Image, row: FrameTracks, settings: Settings) -> None:
    """Show observed potato IDs, finite line, direction and distinct count meanings."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    for observation in row.tracks:
        color = "lime" if observation.confirmed else "yellow"
        draw.rectangle(observation.xyxy, outline=color, width=3)
        x, y = observation.xyxy[:2]
        label = f"P{observation.track_id}" + ("" if observation.confirmed else "?")
        draw.text((x, max(0, y - 24)), label, fill=color, font=font, stroke_width=1)
    draw.line(
        (settings.x_min, settings.line_y, settings.x_max, settings.line_y),
        fill="cyan",
        width=4,
    )
    for offset in (-settings.hysteresis, settings.hysteresis):
        draw.line(
            (
                settings.x_min,
                settings.line_y + offset,
                settings.x_max,
                settings.line_y + offset,
            ),
            fill="cyan",
            width=1,
        )
    arrow_x = min(image.width - 18, settings.x_max + 18)
    sign = -1 if settings.direction == "up" else 1
    tip_y = settings.line_y + 40 * sign
    draw.line(
        (arrow_x, settings.line_y - 30 * sign, arrow_x, tip_y), fill="cyan", width=4
    )
    draw.polygon(
        (
            (arrow_x, tip_y),
            (arrow_x - 8, tip_y - 14 * sign),
            (arrow_x + 8, tip_y - 14 * sign),
        ),
        fill="cyan",
    )
    top = image.height - 80
    draw.rectangle((0, top, image.width, image.height), fill="black")
    draw.text(
        (8, top + 5),
        f"ON SCREEN potato: {row.on_screen_potato}  stone: {row.on_screen_stone}"
        + f"    {row.second:.3f}s  frame {row.frame}",
        font=font,
        fill="white",
    )
    draw.text(
        (8, top + 40),
        f"{settings.direction.upper()} LINE CUMULATIVE: {row.cumulative_crossings}"
        + "   (track events; IDs may split/switch)",
        font=font,
        fill="cyan",
    )


def render(inputs: Inputs, settings: Settings, output: Path) -> None:
    """Hash-check before reserving output, replay, encode, then decode every frame."""
    cache = load(inputs)
    result = replay(cache, settings)
    output = new_output(output)
    durations = {f.pts: f.duration for f in cache.frames}
    video = output / "annotated.mp4"
    with av.open(str(inputs.source)) as source, av.open(str(video), "w") as target:
        original = source.streams.video[0]
        require(
            original.width == 1280
            and original.height == 720
            and original.time_base == TIME_BASE,
            "Expected original 1280x720 source with 90 kHz clock",
        )
        require(
            settings.x_max <= original.width
            and settings.hysteresis <= settings.line_y
            and settings.line_y + settings.hysteresis < original.height,
            "Counting line/hysteresis must fit the source image",
        )
        stream = target.add_stream("libx264")
        stream.width, stream.height = original.width, original.height
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = TIME_BASE
        stream.options = dict(ENCODER_OPTIONS)
        for frame, expected, tracked in zip(
            source.decode(video=0), cache.frames, result.frames, strict=True
        ):
            require(
                frame.pts == expected.pts
                and frame.time_base == expected.time_base
                and frame.duration == expected.duration,
                f"Cache/source timing differs at frame {expected.frame}",
            )
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            overlay(image, tracked, settings)
            annotated = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            annotated.pts, annotated.time_base = expected.pts, expected.time_base
            annotated.duration = expected.duration
            for packet in stream.encode(annotated):
                preserve_packet_duration(packet, durations)
                target.mux(packet)
        for packet in stream.encode():
            preserve_packet_duration(packet, durations)
            target.mux(packet)
    with av.open(str(video)) as encoded:
        require(len(encoded.streams.audio) == 0, "Unexpected output audio")
        require(
            encoded.streams.video[0].codec_context.name == "h264"
            and encoded.streams.video[0].time_base == TIME_BASE,
            "Output codec or time base differs",
        )
        for actual, expected in zip(encoded.decode(video=0), cache.frames, strict=True):
            require(
                actual.pts == expected.pts
                and actual.time_base == expected.time_base
                and actual.duration == expected.duration,
                f"Output timing differs at frame {expected.frame}",
            )
    write_json(
        output / "tracks.json",
        {
            "meaning": "observed_potato_boxes_with_track_ids_not_perfect_physical_identity",
            "frames": [asdict(f) for f in result.frames],
        },
    )
    write_json(
        output / "events.json",
        {
            "meaning": "once_per_track_directed_finite_line_crossings",
            "timestamp": "confirming_source_frame_pts; crossing_second_estimate_is_interpolated",
            "events": [asdict(event) for event in result.events],
        },
    )
    source_paths = [
        *sorted((ROOT / "tracking_experiment").glob("*.py")),
        ROOT / "retraining_experiment/video.py",
        ROOT / "retraining_experiment/common.py",
        ROOT / "retraining_data/schema.py",
        ROOT / "stone_experiment/schema.py",
        ROOT / "stone_experiment/metrics.py",
        ROOT / "scripts/baseline_common.py",
    ]
    write_json(
        output / "summary.json",
        {
            "meaning": "directed_track_crossings_not_guaranteed_unique_physical_potatoes",
            "configuration": asdict(settings),
            "inputs": {
                name: str(path.resolve())
                for name, path in (
                    ("source", inputs.source),
                    ("cache", inputs.cache),
                    ("weights_hash_only", inputs.weights),
                    ("cached_video_hash_only", inputs.cached_video),
                )
            },
            "hashes": asdict(cache.hashes),
            "video_sha256": sha256(video),
            "source_code_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in source_paths
            },
            "packages": {
                name: version(name) for name in ("scipy", "numpy", "av", "pillow")
            },
            "frames": len(result.frames),
            "track_ids_created": result.created,
            "cumulative_crossings": len(result.events),
            "stream_time_base": str(TIME_BASE),
            "codec": "h264",
            "encoder_options": ENCODER_OPTIONS,
            "audio": False,
            "verification": "decoded_all_output_frames_exact_pts_timebase_duration",
            "association": "canonical_order_hungarian_predicted_center_plus_iou_with_unmatched_dummies",
            "detection_policy": "cached_class_0_only; no_predictions_counted_during_dropouts",
        },
    )


if __name__ == "__main__":
    from tracking_experiment.cli import arguments

    options = arguments()
    render(options.inputs, options.settings, options.output)
