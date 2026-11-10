# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow"]
# ///
# How to run: /home/sc/.cache/potato-baseline/.venv/bin/python -B -m retraining_data.verify
"""Verify the new dataset against fresh read-only source reconstruction."""

import argparse
from pathlib import Path

from retraining_data.common import (
    OUTPUTS,
    ROOT,
    digest,
    file_hash,
    image_pixels,
    yaml_bytes,
)
from retraining_data.schema import (
    V1,
    V2,
    Profile,
    Snapshot,
    json_bytes,
    profile_named,
    read_manifest,
    require,
    validate_records,
)
from retraining_data.sources import Plan, build_plan


class Options(argparse.Namespace):
    """Mutable argparse namespace with a typed command-specific default."""

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Path | None = None
        self.profile: Profile = V1


def verify_files(dataset: Path, plan: Plan) -> Snapshot:
    """Check actual membership and bytes, not just self-reported manifest hashes."""
    manifest = read_manifest((dataset / "manifest.json").read_bytes(), plan.profile)
    require(file_hash(dataset / "annotations.json") == manifest.snapshot_sha256, "Snapshot hash differs")
    require(file_hash(dataset / "data.yaml") == manifest.data_yaml_sha256, "YAML hash differs")
    require((dataset / "data.yaml").read_bytes() == yaml_bytes(), "YAML configuration differs")
    require((dataset / "annotations.json").read_bytes() == json_bytes(plan.snapshot),
            "Snapshot differs from source reconstruction")
    snapshot = plan.snapshot
    validate_records(snapshot.frames, plan.profile)
    expected = {"annotations.json", "manifest.json", "data.yaml"}
    labels: dict[str, str] = {}
    for payload in plan.payloads:
        record = payload.record
        image_path = f"images/{record.split}/{record.filename}"
        label_path = f"labels/{record.split}/{Path(record.filename).with_suffix('.txt')}"
        expected.update((image_path, label_path))
        image, label = dataset / image_path, dataset / label_path
        require(file_hash(image) == record.image_sha256, f"Output image changed: {image_path}")
        require(image_pixels(image) == record.pixel_sha256, f"Output RGB pixels changed: {image_path}")
        require(
            label.read_bytes() == payload.label and file_hash(label) == record.label_sha256,
            f"Output label changed: {label_path}",
        )
        require(digest(payload.label) == record.label_sha256, "Source label hash differs")
        labels[label_path] = record.label_sha256
    actual = {p.relative_to(dataset).as_posix() for p in dataset.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in dataset.rglob("*")), "Output symlink refused")
    require(actual == expected, "Output file membership differs")
    require(labels == manifest.label_hashes, "Output label inventory differs")
    for source in snapshot.source_files:
        require(file_hash(ROOT / source.path) == source.sha256, f"Source changed: {source.path}")
    return snapshot


def verify_dataset(dataset: Path, profile: Profile = V1) -> Snapshot:
    """Re-run legacy verification and native eligibility/provenance before checking output."""
    return verify_files(dataset, build_plan(profile))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--profile", type=profile_named, choices=(V1, V2), default=V1,
                            metavar="{v1,v2}")
    _ = parser.add_argument("--dataset", type=Path)
    args = parser.parse_args(namespace=Options())
    snapshot = verify_dataset(
        args.dataset if args.dataset is not None else OUTPUTS / args.profile.dataset_directory, args.profile,
    )
    print(f"Verified {len(snapshot.frames)} frames; source hashes and output bytes match")


if __name__ == "__main__":
    main()
