"""Real CPU-only VFR encoding regression, including the final interval."""

from pathlib import Path

import av

from retraining_data.schema import require
from retraining_experiment.video import (
    ENCODER_OPTIONS,
    TIME_BASE,
    preserve_packet_duration,
)


def vfr_proof(root: Path) -> None:
    """Decode actual output rather than inferring timing from container headers."""
    # Given: unequal intervals and a last interval longer than the codec default.
    expected = {0: 3750, 3750: 7500, 11250: 3750, 15000: 7500}
    path = root / "vfr.mp4"
    # When: the production encoding policy writes deterministic black frames.
    with av.open(str(path), "w") as target:
        stream = target.add_stream("libx264")
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = TIME_BASE
        stream.options = {key: value for key, value in ENCODER_OPTIONS.items()}
        for pts, duration in expected.items():
            frame = av.VideoFrame(32, 32, "rgb24")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts, frame.time_base, frame.duration = pts, TIME_BASE, duration
            for packet in stream.encode(frame):
                preserve_packet_duration(packet, expected)
                target.mux(packet)
        for packet in stream.encode():
            preserve_packet_duration(packet, expected)
            target.mux(packet)
    # Then: all decoded PTS and durations, including the final duration, match.
    with av.open(str(path)) as encoded:
        actual = [(f.pts, f.duration) for f in encoded.decode(video=0)]
    require(actual == list(expected.items()), "Exact VFR PTS and durations")
