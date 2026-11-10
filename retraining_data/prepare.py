# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow"]
# ///
# How to run: /home/sc/.cache/potato-baseline/.venv/bin/python -B -m retraining_data.prepare
"""Build new experimental YOLO data only; refuse all existing outputs."""

import argparse
from pathlib import Path

from retraining_data.common import (
    OUTPUTS,
    ROOT,
    digest,
    file_hash,
    output_path,
    yaml_bytes,
)
from retraining_data.schema import (
    V1,
    V2,
    Manifest,
    Profile,
    json_bytes,
    profile_named,
    require,
)
from retraining_data.sources import Plan, build_plan


class Options(argparse.Namespace):
    """Mutable argparse namespace with a typed command-specific default."""

    def __init__(self) -> None:
        super().__init__()
        self.output: Path | None = None
        self.profile: Profile = V1


def publish(plan: Plan, output: Path) -> None:
    """Reserve a fresh directory, publish verified bytes, then bind their hashes."""
    destination = output_path(output)
    # mkdir(exist_ok=False) also handles a competing creator after validation.
    destination.mkdir(parents=True, exist_ok=False)
    label_hashes: dict[str, str] = {}
    for payload in plan.payloads:
        record = payload.record
        image = destination / "images" / record.split / record.filename
        label = destination / "labels" / record.split / Path(record.filename).with_suffix(".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / record.source_image).read_bytes()
        require(digest(data) == record.image_sha256, f"Source image changed during build: {record.filename}")
        require(digest(payload.label) == record.label_sha256, f"Planned label changed: {record.filename}")
        _ = image.write_bytes(data)
        _ = label.write_bytes(payload.label)
        label_hashes[label.relative_to(destination).as_posix()] = record.label_sha256
    # Detect source changes between collection and publication. Partial output is
    # deliberately left in place on failure and can never be overwritten/reused.
    for source in plan.snapshot.source_files:
        require(file_hash(ROOT / source.path) == source.sha256, f"Source changed during build: {source.path}")
    _ = (destination / "annotations.json").write_bytes(json_bytes(plan.snapshot))
    _ = (destination / "data.yaml").write_bytes(yaml_bytes())
    manifest = Manifest(
        snapshot_sha256=file_hash(destination / "annotations.json"),
        data_yaml_sha256=file_hash(destination / "data.yaml"), label_hashes=label_hashes,
        schema_version=f"retraining-manifest-{plan.profile.name}",
    )
    _ = (destination / "manifest.json").write_bytes(json_bytes(manifest))


def prepare(output: Path, profile: Profile = V1) -> None:
    """Reject occupied paths before scanning any sources."""
    destination = output_path(output)
    plan = build_plan(profile)
    publish(plan, destination)


def main() -> None:
    """CPU-only command with no dependency installation or model imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--profile", type=profile_named, choices=(V1, V2), default=V1,
                            metavar="{v1,v2}")
    _ = parser.add_argument("--output", type=Path)
    args = parser.parse_args(namespace=Options())
    prepare(args.output if args.output is not None else OUTPUTS / args.profile.dataset_directory, args.profile)
    frames, potatoes, stones = args.profile.train_counts
    print(f"Prepared {args.profile.name}: train{frames}/P{potatoes}/S{stones}, val11/P60/S8, test12/P33/S7")


if __name__ == "__main__":
    main()
