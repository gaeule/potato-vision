"""Full-source H.264 rendering with exact source PTS and visible frame counts."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Final

from postprocess_review.common import infer
from retraining_data.schema import require
from retraining_experiment.common import (
    OUTPUTS,
    PARAMETERS,
    PROTOCOL,
    SELECTED,
    SOURCE,
    check_selected,
    new_output,
    record_context,
    runtime,
    sha256,
    write_json,
)
from stone_experiment.common import PipelineError
from stone_experiment.evaluate import detections
from stone_experiment.metrics import Detection
from stone_experiment.predict import check_classes

if TYPE_CHECKING:
    from av.packet import Packet
    from av.video.stream import VideoStream

DIAGNOSTIC_SECONDS: Final = (20, 35, 65, 70, 71, 78, 80, 83, 130, 134)
TIME_BASE: Final = Fraction(1, 90000)
ENCODER_OPTIONS: Final = {"crf": "20", "preset": "medium", "bf": "0"}


def preserve_packet_duration(
    packet: "Packet[VideoStream]", durations: Mapping[int, int]
) -> None:
    """No B-frames plus explicit packet duration preserves the final VFR interval."""
    if packet.pts is None or packet.time_base != TIME_BASE:
        raise PipelineError("Encoded packet lacks the source presentation clock")
    packet.duration = durations[packet.pts]


@dataclass(frozen=True, slots=True)
class FrameCount:
    """One frame observation, never a count of distinct physical objects."""

    frame: int
    pts: int
    time_base: str
    duration: int
    second: float
    potato: int
    stone: int
    raw_detections: tuple[Detection, ...]


def render(weights: Path = SELECTED, output: Path = OUTPUTS / "video") -> None:
    """Render all 2224 frames; preserve VFR clock, durations, and no-audio policy."""
    weights = weights.resolve()
    require(weights.is_file(), "Existing selected two-class checkpoint required")
    check_selected(weights)
    output = new_output(output)
    with runtime(output):
        import av
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from ultralytics import YOLO

        model = YOLO(str(weights))
        check_classes(model, two_classes=True)
        record_context(output, (weights,))
        diagnostic = output / "diagnostics"
        diagnostic.mkdir()
        counts: list[FrameCount] = []
        source_durations: dict[int, int] = {}
        captured: dict[int, int] = {}
        font = ImageFont.load_default(size=26)
        with (
            av.open(str(SOURCE)) as source,
            av.open(str(output / "annotated.mp4"), "w") as target,
        ):
            original = source.streams.video[0]
            require(
                original.time_base == TIME_BASE, "Expected source time base 1/90000"
            )
            stream = target.add_stream("libx264")
            stream.time_base = TIME_BASE
            stream.codec_context.time_base = TIME_BASE
            stream.width, stream.height = original.width, original.height
            stream.pix_fmt = "yuv420p"
            stream.options = {key: value for key, value in ENCODER_OPTIONS.items()}
            previous: Fraction | None = None
            for index, frame in enumerate(source.decode(video=0)):
                require(
                    frame.pts is not None and frame.time_base == TIME_BASE,
                    "Source frame lacks 90 kHz PTS",
                )
                if frame.pts is None:
                    raise RuntimeError("Missing source PTS")
                timestamp = frame.pts * frame.time_base
                require(
                    previous is None or timestamp > previous,
                    "Non-increasing source PTS",
                )
                previous = timestamp
                source_image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                result = infer(model, source_image, PARAMETERS)
                raw = detections(result, potato_only=False)
                potato, stone = (sum(d.class_id == c for d in raw) for c in (0, 1))
                image = Image.fromarray(result.plot()[..., ::-1])
                draw = ImageDraw.Draw(image)
                footer_top = original.height - 44
                draw.rectangle(
                    (0, footer_top, original.width, original.height), fill="black"
                )
                draw.text(
                    (8, footer_top + 6),
                    f"ON SCREEN  potato: {potato}  stone: {stone} (not unique)",
                    font=font,
                    fill="white",
                )
                annotated = av.VideoFrame.from_ndarray(
                    np.asarray(image), format="rgb24"
                )
                annotated.pts, annotated.time_base = frame.pts, frame.time_base
                annotated.duration = frame.duration
                source_durations[frame.pts] = frame.duration
                for packet in stream.encode(annotated):
                    preserve_packet_duration(packet, source_durations)
                    target.mux(packet)
                counts.append(
                    FrameCount(
                        index,
                        frame.pts,
                        str(frame.time_base),
                        frame.duration,
                        float(timestamp),
                        potato,
                        stone,
                        raw,
                    )
                )
                for second in DIAGNOSTIC_SECONDS:
                    if second not in captured and timestamp >= second:
                        captured[second] = frame.pts
                        image.save(diagnostic / f"second_{second:03d}_annotated.png")
                        source_image.save(
                            diagnostic / f"second_{second:03d}_source.png"
                        )
                        write_json(
                            diagnostic / f"second_{second:03d}.json",
                            {
                                "requested_second": second,
                                **asdict(counts[-1]),
                                "sampling": "first_source_frame_at_or_after_fixed_second",
                            },
                        )
            for packet in stream.encode():
                preserve_packet_duration(packet, source_durations)
                target.mux(packet)
        require(
            len(counts) == 2224 and len(captured) == len(DIAGNOSTIC_SECONDS),
            "Source frame count or diagnostic coverage differs",
        )
        # Decode the finished file: header inspection alone cannot prove VFR preservation.
        with av.open(str(output / "annotated.mp4")) as encoded:
            require(len(encoded.streams.audio) == 0, "Output must not contain audio")
            require(
                encoded.streams.video[0].time_base == TIME_BASE,
                "Output time base changed",
            )
            require(
                encoded.streams.video[0].codec_context.name == "h264",
                "Output codec changed",
            )
            for actual, expected in zip(encoded.decode(video=0), counts, strict=True):
                require(
                    actual.pts == expected.pts and actual.time_base == TIME_BASE,
                    f"Encoded PTS changed at frame {expected.frame}",
                )
                require(
                    actual.duration == expected.duration,
                    f"Encoded duration changed at frame {expected.frame}",
                )
        write_json(
            output / "counts.json",
            {
                **PROTOCOL,
                "meaning": "per_frame_on_screen_not_unique_objects",
                "source_sha256": sha256(SOURCE),
                "checkpoint_sha256": sha256(weights),
                "video_sha256": sha256(output / "annotated.mp4"),
                "stream_time_base": str(TIME_BASE),
                "codec_time_base": str(TIME_BASE),
                "codec": "h264",
                "encoder_options": ENCODER_OPTIONS,
                "audio": False,
                "diagnostic_seconds": DIAGNOSTIC_SECONDS,
                "frames": [asdict(c) for c in counts],
                "verification": "decoded_all_2224_frames_exact_pts_timebase_duration",
            },
        )


if __name__ == "__main__":
    from retraining_experiment.cli import arguments

    args = arguments("video")
    render(args.weights, args.output)
