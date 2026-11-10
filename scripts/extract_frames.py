"""Sequential PyAV extraction, independent of the annotation file."""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import av

from .baseline_common import BASE, SOURCE, PipelineError, sha256, write_json


class FrameRecord(TypedDict):
    """Identity and exact decode timestamp of one extracted frame."""

    second: int
    actual_second: float
    pts: int
    time_base: str
    file: str
    width: int
    height: int


def extract(seconds: list[int], output: Path, regenerate: bool = False) -> list[FrameRecord]:
    """Select the first decoded frame with timestamp >= each requested second."""
    if not seconds or any(type(second) is not int or second < 0 for second in seconds):
        raise PipelineError("Seconds must be a nonempty list of nonnegative integers")
    requested = sorted(set(seconds))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "extraction_manifest.json"
    owned = set()
    previous_rows: list[FrameRecord] = []
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["source_sha256"] == sha256(SOURCE):
            previous_rows = previous["frames"]
            owned = {row["file"] for row in previous["frames"]}
    for second in requested:
        name = f"frame_{second:06d}.png"
        if (output / name).exists() and not (regenerate and name in owned):
            raise PipelineError(f"Refusing to overwrite {output / name}; only manifest-owned frames can be regenerated")
    if manifest.exists() and not regenerate:
        raise PipelineError(f"Refusing to overwrite {manifest}; use --regenerate")
    rows: list[FrameRecord] = []
    index = 0
    with av.open(str(SOURCE)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise PipelineError("Decoded frame has no timestamp")
            timestamp = frame.pts * frame.time_base
            while index < len(requested) and timestamp >= requested[index]:
                second = requested[index]
                name = f"frame_{second:06d}.png"
                frame.to_image().save(output / name)
                rows.append({"second": second, "actual_second": float(timestamp), "pts": frame.pts,
                             "time_base": str(frame.time_base), "file": name,
                             "width": frame.width, "height": frame.height})
                index += 1
            if index == len(requested):
                break
    if index != len(requested):
        raise PipelineError(f"Source ended before requested second {requested[index]}")
    retained = {row["second"]: row for row in previous_rows}
    retained.update({row["second"]: row for row in rows})
    write_json(manifest, {"source": SOURCE.name, "source_sha256": sha256(SOURCE),
                          "frames": [retained[second] for second in sorted(retained)]})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", required=True, type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=BASE / "frames")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    extract(args.seconds, args.output, args.regenerate)
