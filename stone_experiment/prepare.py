"""Prepare two-class YOLO data from verified, read-only baseline frame PNGs."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from stone_experiment.common import (
    ANNOTATIONS,
    BASE,
    DATASET,
    SOURCE,
    Options,
    PipelineError,
    new_output,
    sha256,
    write_json,
)
from stone_experiment.schema import (
    PROVENANCE,
    RANGES,
    Frame,
    annotations,
    document,
    integer,
    rows,
)


def verified_frames(path: Path) -> tuple[Frame, ...]:
    """Require unchanged baseline potatoes, all 63 frames, and both train/val classes."""
    frames = annotations(path)
    original = annotations(BASE / "annotations.json", baseline=True)
    if len(frames) != 63 or len(original) != 63:
        raise PipelineError("Expected exactly 63 baseline frames")
    for frame, old in zip(frames, original, strict=True):
        if (frame.second, frame.split, frame.boxes) != (old.second, old.split, old.boxes):
            raise PipelineError(f"Potato labels or split changed at second {frame.second}")
    for split, expected in (("train", (40, 222)), ("val", (11, 60)), ("test", (12, 33))):
        subset = [frame for frame in frames if frame.split == split]
        if (len(subset), sum(len(frame.boxes) for frame in subset)) != expected:
            raise PipelineError(f"Baseline counts changed for {split}")
        if split in ("train", "val") and not sum(len(frame.stones) for frame in subset):
            raise PipelineError(f"Both classes must occur in {split}")
    return frames


def prepare(path: Path, output: Path) -> None:
    """Check provenance and hashes before reserving any output directory."""
    frames = verified_frames(path)
    manifest_path = BASE / "internal/manifest.json"
    manifest = document(manifest_path)
    if (manifest["source"], manifest["source_sha256"], manifest["annotations_sha256"], manifest["provenance"]) != (
        SOURCE.name, sha256(SOURCE), sha256(BASE / "annotations.json"), PROVENANCE,
    ):
        raise PipelineError("Baseline manifest source/annotation hashes or provenance differ")
    metadata = {integer(row["second"]): row for row in rows(manifest["frames"])}
    if set(metadata) != {frame.second for frame in frames}:
        raise PipelineError("Baseline manifest frame membership differs")
    records: list[dict[str, str | int | float]] = []
    for frame in frames:
        row = metadata[frame.second]
        if (row["split"], row["file"], row["width"], row["height"], row["box_count"], row["time_base"]) != (
            frame.split, frame.filename, 1280, 720, len(frame.boxes), "1/90000",
        ):
            raise PipelineError(f"Baseline frame provenance differs: {frame.filename}")
        image = BASE / "frames" / frame.filename
        reference = BASE / "internal/images" / frame.split / frame.filename
        digest = sha256(image)
        # The old manifest has timestamps but no PNG digest: compare the verified
        # baseline dataset copy, then record the digest in the new manifest.
        if digest != sha256(reference):
            raise PipelineError(f"PNG differs from baseline dataset copy: {image}")
        with Image.open(image) as decoded:
            if decoded.size != (1280, 720):
                raise PipelineError(f"Unexpected PNG size: {image}")
            decoded.verify()
        records.append({
            "second": frame.second, "split": frame.split, "file": frame.filename,
            "sha256": digest, "pts": integer(row["pts"]), "time_base": "1/90000",
            "potato_count": len(frame.boxes), "stone_count": len(frame.stones),
        })
    output = new_output(output)
    for frame in frames:
        image_dir, label_dir = output / "images" / frame.split, output / "labels" / frame.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(BASE / "frames" / frame.filename, image_dir / frame.filename)
        lines: list[str] = []
        for class_id, group in enumerate((frame.boxes, frame.stones)):
            for x1, y1, x2, y2 in group:
                lines.append(f"{class_id} {(x1+x2)/2560:.9f} {(y1+y2)/1440:.9f} " +
                             f"{(x2-x1)/1280:.9f} {(y2-y1)/720:.9f}\n")
        _ = (label_dir / Path(frame.filename).with_suffix(".txt")).write_text("".join(lines), encoding="utf-8")
    yaml = f"path: {json.dumps(str(output), ensure_ascii=False)}\n"
    yaml += "".join(f"{split}: images/{split}\n" for split in RANGES)
    _ = (output / "data.yaml").write_text(yaml + "names:\n  0: potato\n  1: stone\n", encoding="utf-8")
    # Snapshot chief input: later selection/evaluation cannot silently use edits.
    write_json(output / "annotations.json", {
        "source": SOURCE.name, "width": 1280, "height": 720, "provenance": PROVENANCE,
        "frames": [asdict(frame) for frame in frames],
    })
    write_json(output / "manifest.json", {
        "source": SOURCE.name, "source_sha256": sha256(SOURCE),
        "annotations_sha256": sha256(path), "baseline_manifest_sha256": sha256(manifest_path),
        "baseline_annotations_sha256": sha256(BASE / "annotations.json"),
        "snapshot_sha256": sha256(output / "annotations.json"),
        "data_yaml_sha256": sha256(output / "data.yaml"),
        "provenance": PROVENANCE, "counts_are": "frame_observations_not_unique_objects",
        "frames": records,
        "label_hashes": {str(p.relative_to(output)): sha256(p) for p in sorted(output.glob("labels/*/*.txt"))},
    })


def verify_dataset(dataset: Path) -> tuple[Frame, ...]:
    """Refuse stale or changed data before training, selection, or evaluation."""
    manifest = document(dataset / "manifest.json")
    if manifest["snapshot_sha256"] != sha256(dataset / "annotations.json"):
        raise PipelineError("Dataset annotation snapshot changed")
    if manifest["data_yaml_sha256"] != sha256(dataset / "data.yaml"):
        raise PipelineError("Dataset YAML changed")
    frames = verified_frames(dataset / "annotations.json")
    expected_images = {f"images/{frame.split}/{frame.filename}" for frame in frames}
    actual_images = {str(p.relative_to(dataset)) for p in dataset.glob("images/*/*") if p.is_file()}
    if actual_images != expected_images:
        raise PipelineError("Dataset image membership changed")
    manifest_rows = rows(manifest["frames"])
    if len(manifest_rows) != len(frames):
        raise PipelineError("Dataset manifest membership changed")
    for row in manifest_rows:
        split, filename = row["split"], row["file"]
        if not isinstance(split, str) or not isinstance(filename, str) or Path(filename).name != filename:
            raise PipelineError("Invalid dataset manifest image path")
        if row["sha256"] != sha256(dataset / "images" / split / filename):
            raise PipelineError(f"Dataset image changed: {filename}")
    hashes = manifest["label_hashes"]
    if not isinstance(hashes, dict):
        raise PipelineError("Expected label hashes")
    actual = {str(p.relative_to(dataset)): sha256(p) for p in dataset.glob("labels/*/*.txt")}
    if actual != hashes:
        raise PipelineError("Dataset labels changed")
    return frames


def main() -> None:
    """Prepare labels without importing a model or touching the original ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    _ = parser.add_argument("--output", type=Path, default=DATASET)
    args = parser.parse_args(namespace=Options())
    prepare(args.annotations, args.output)


if __name__ == "__main__":
    main()
