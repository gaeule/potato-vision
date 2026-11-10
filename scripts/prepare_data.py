"""Build internal YOLO splits or import the separate evaluation-only ZIP."""

import argparse
import io
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Final

from PIL import Image

from .baseline_common import (
    BASE,
    PROVENANCE,
    ROOT,
    SOURCE,
    PipelineError,
    label_line,
    output_directory,
    sha256,
    write_json,
)
from .extract_frames import extract

RANGES: Final = {"train": (0, 84), "val": (92, 112), "test": (120, 147)}


def valid_box(points: list[float], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Reject malformed, degenerate, nonfinite, or out-of-image boxes."""
    if len(points) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in points):
        raise PipelineError(f"Invalid box: {points}")
    x1, y1, x2, y2 = points
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise PipelineError(f"Box outside {size}: {points}")
    return x1, y1, x2, y2


def dataset_yaml(output: Path, external: bool = False) -> None:
    """Write absolute-root YAML, with a legal but evaluation-only external train path."""
    paths = {"train": "images/test", "val": "images/test", "test": "images/test"} if external else {
        split: f"images/{split}" for split in RANGES
    }
    # JSON strings are valid YAML scalars and safely quote paths containing spaces.
    text = f"path: {json.dumps(str(output.resolve()), ensure_ascii=False)}\n"
    text += "".join(f"{key}: {value}\n" for key, value in paths.items())
    (output / "data.yaml").write_text(text + "names:\n  0: potato\n", encoding="utf-8")


def prepare_internal(annotations: Path, regenerate: bool) -> None:
    """Validate the chief's annotation contract before generating any artifacts."""
    document = json.loads(annotations.read_text(encoding="utf-8"))
    if (document["source"], document["width"], document["height"], document["provenance"]) != (
        SOURCE.name, 1280, 720, PROVENANCE
    ):
        raise PipelineError("Unexpected source, dimensions, or annotation provenance")
    frames = document["frames"]
    if not 1 <= len(frames) <= 70:
        raise PipelineError("Expected 1..70 annotated frames")
    seen = set()
    splits = set()
    for row in frames:
        second, split = row["second"], row["split"]
        if type(second) is not int or split not in RANGES:
            raise PipelineError(f"Invalid second/split: {row}")
        lower, upper = RANGES[split]
        if second in seen or not lower <= second <= upper:
            raise PipelineError(f"Duplicate second or temporal split violation: {row}")
        seen.add(second)
        splits.add(split)
        for box in row["boxes"]:
            valid_box(box, (1280, 720))
    if splits != set(RANGES):
        raise PipelineError("All three internal splits must contain frames")
    output = output_directory(BASE / "internal", regenerate)
    # Resolve extraction failures before removing any existing dataset labels.
    decoded = extract(sorted(seen), output / "decoded", regenerate)
    # Clear only files enumerated by our own previous manifest, never chief frames.
    previous_manifest = output / "manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        for row in previous["frames"]:
            for directory, extension in (("images", "png"), ("labels", "txt")):
                (output / directory / row["split"] / f"frame_{row['second']:06d}.{extension}").unlink(missing_ok=True)
    metadata = {row["second"]: row for row in decoded}
    for row in frames:
        second, split = row["second"], row["split"]
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source = output / "decoded" / f"frame_{second:06d}.png"
        if (metadata[second]["width"], metadata[second]["height"]) != (1280, 720):
            raise PipelineError("Decoded dimensions differ from annotations")
        shutil.copyfile(source, image_dir / source.name)
        text = "".join(label_line(valid_box(box, (1280, 720)), (1280, 720)) for box in row["boxes"])
        (label_dir / f"frame_{second:06d}.txt").write_text(text, encoding="utf-8")
    write_json(previous_manifest, {
        "source": SOURCE.name, "source_sha256": sha256(SOURCE),
        "annotations_sha256": sha256(annotations), "provenance": PROVENANCE,
        "frames": [{**metadata[row["second"]], "split": row["split"],
                    "box_count": len(row["boxes"])} for row in frames],
    })
    dataset_yaml(output)


def import_external(archive: Path, regenerate: bool) -> None:
    """Validate every ZIP pair in memory before writing an evaluation-only dataset."""
    records = []
    corrections = []
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
            raise PipelineError("ZIP requires unique flat filenames")
        images = {Path(name).stem for name in names if name.endswith(".png")}
        annotations = {Path(name).stem for name in names if name.endswith(".json")}
        if not images or images != annotations or len(names) != 2 * len(images):
            raise PipelineError("ZIP must contain exactly one PNG and JSON for every stem")
        for stem in sorted(images):
            doc = json.loads(zipped.read(f"{stem}.json"))
            data = zipped.read(f"{stem}.png")
            with Image.open(io.BytesIO(data)) as image:
                size = image.size
                image.verify()
            if size != (doc["imageWidth"], doc["imageHeight"]) or doc["imagePath"] != f"{stem}.png":
                raise PipelineError(f"Image dimensions or imagePath mismatch: {stem}")
            labels = []
            for index, shape in enumerate(doc["shapes"]):
                points = shape["points"]
                if shape["label"] != "potato" or shape["shape_type"] != "rectangle" or len(points) not in (2, 4):
                    raise PipelineError(f"Unsupported shape/class in {stem}")
                if any(len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                                              for v in point) for point in points):
                    raise PipelineError(f"Malformed points in {stem}")
                xs, ys = [point[0] for point in points], [point[1] for point in points]
                if len(points) == 4 and set(map(tuple, points)) != {
                    (min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))
                }:
                    raise PipelineError(f"Four points are not rectangle corners: {stem}")
                box = [min(xs), min(ys), max(xs), max(ys)]
                if box[0] < 0:
                    if stem != "extra2_t000151000ms_f002265" or not -0.14 < box[0] < 0:
                        raise PipelineError(f"Unexpected negative x in {stem}: {box[0]}")
                    corrections.append({"file": f"{stem}.json", "shape": index,
                                        "coordinate": "x1", "before": box[0], "after": 0})
                    box[0] = 0
                labels.append(label_line(valid_box(box, size), size))
            records.append((stem, data, "".join(labels), size, len(labels)))
    output = output_directory(BASE / "external", regenerate)
    for directory in ("images/test", "labels/test"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    previous = output / "manifest.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        for row in old["frames"]:
            (output / "images/test" / f"{row['stem']}.png").unlink(missing_ok=True)
            (output / "labels/test" / f"{row['stem']}.txt").unlink(missing_ok=True)
    for stem, data, labels, _, _ in records:
        (output / "images/test" / f"{stem}.png").write_bytes(data)
        (output / "labels/test" / f"{stem}.txt").write_text(labels, encoding="utf-8")
    write_json(previous, {
        "archive": archive.name, "archive_sha256": sha256(archive),
        "provenance": "supplied_annotations_not_expert_gold", "evaluation_only": True,
        "corrections": corrections,
        "frames": [{"stem": stem, "width": size[0], "height": size[1], "box_count": count}
                   for stem, _, _, size, count in records],
    })
    dataset_yaml(output, external=True)
    print(json.dumps({"pairs": len(records), "corrections": corrections}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    internal = commands.add_parser("internal")
    internal.add_argument("--annotations", type=Path, default=BASE / "annotations.json")
    internal.add_argument("--regenerate", action="store_true")
    external = commands.add_parser("external")
    external.add_argument("--archive", type=Path, default=ROOT / "추가영상2_돌검수.zip")
    external.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    match args.command:
        case "internal":
            prepare_internal(args.annotations, args.regenerate)
        case "external":
            import_external(args.archive, args.regenerate)
