"""Render the entire source with exact VFR timestamps and per-frame counts."""

import argparse
from fractions import Fraction
from pathlib import Path

from stone_experiment.common import (
    CONFIDENCE,
    NMS_IOU,
    OUTPUTS,
    SOURCE,
    Options,
    PipelineError,
    environment,
    new_output,
    runtime,
    sha256,
    write_json,
)
from stone_experiment.predict import check_classes, predict


def render(weights: Path, confidence: float, output: Path) -> None:
    """Encode H.264 without audio; neither tracking nor unique counting is used."""
    if not weights.is_file() or not 0 < confidence <= 1:
        raise PipelineError("Existing weights and confidence in (0, 1] are required")
    output = new_output(output)
    runtime()
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO

    model = YOLO(str(weights))
    check_classes(model)
    counts: list[dict[str, int | float | str | dict[str, int]]] = []
    font = ImageFont.load_default(size=26)
    with av.open(str(SOURCE)) as source, av.open(str(output / "annotated.mp4"), "w") as target:
        original = source.streams.video[0]
        time_base = original.time_base
        if time_base != Fraction(1, 90000):
            raise PipelineError(f"Expected original time_base 1/90000, got {time_base}")
        # Both assignments are essential: average-rate codec ticks collapse VFR
        # timestamps onto the same DTS even when the muxer uses 90 kHz ticks.
        stream = target.add_stream("libx264")
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.width, stream.height = original.width, original.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        previous: Fraction | None = None
        for index, frame in enumerate(source.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise PipelineError("Source frame has no timestamp")
            timestamp = frame.pts * frame.time_base
            if previous is not None and timestamp <= previous:
                raise PipelineError("Non-increasing source presentation timestamp")
            previous = timestamp
            result = predict(model, frame.to_image(), confidence)
            if result.boxes is None:
                raise PipelineError("Missing detection boxes")
            classes = [int(result.boxes.cls[i].item()) for i in range(len(result.boxes))]
            visible = {name: classes.count(class_id) for class_id, name in model.names.items()}
            image = Image.fromarray(result.plot()[..., ::-1])
            draw = ImageDraw.Draw(image)
            # The foreground guard is below the belt; keep top-edge detections visible.
            footer_top = original.height - 44
            draw.rectangle((0, footer_top, original.width, original.height), fill="black")
            text = "  ".join(f"{name}: {count}" for name, count in visible.items())
            draw.text((8, footer_top + 6), f"ON SCREEN  {text} (not unique)", fill="white", font=font)
            annotated = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            annotated.pts, annotated.time_base = frame.pts, frame.time_base
            annotated.duration = frame.duration
            for packet in stream.encode(annotated):
                target.mux(packet)
            counts.append({
                "frame": index, "pts": frame.pts, "time_base": str(frame.time_base),
                "second": float(timestamp), "counts_by_class": visible,
            })
        for packet in stream.encode():
            target.mux(packet)
    with av.open(str(output / "annotated.mp4")) as encoded:
        if len(encoded.streams.audio) or encoded.streams.video[0].time_base != time_base:
            raise PipelineError("Output audio/time-base verification failed")
        if encoded.streams.video[0].codec_context.name != "h264":
            raise PipelineError("Output is not H.264")
    write_json(output / "counts.json", {
        "meaning": "per_frame_visible_detections_not_unique_objects",
        "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "checkpoint": str(weights.resolve()), "checkpoint_sha256": sha256(weights),
        "video_sha256": sha256(output / "annotated.mp4"),
        "confidence": confidence, "nms_iou": NMS_IOU,
        "stream_time_base": str(time_base), "codec_time_base": str(time_base),
        "audio": False, "codec": "h264", "environment": environment(), "frames": counts,
    })


def main() -> None:
    """Parse a standalone video command without starting training."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--weights", type=Path, required=True)
    _ = parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    _ = parser.add_argument("--output", type=Path, default=OUTPUTS / "video")
    args = parser.parse_args(namespace=Options())
    render(args.weights, args.confidence, args.output)


if __name__ == "__main__":
    main()
