"""Verified v2 labels as Examples and physically independent trainer-only copies."""

import math
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from retraining_data.schema import V2, Snapshot, Split, require
from retraining_data.verify import verify_dataset
from retraining_experiment.common import check_frozen, sha256, write_json
from stone_experiment.evaluate import Example
from stone_experiment.metrics import Detection


def verified(dataset: Path) -> Snapshot:
    """Use the v2 verifier, not the old 63-image dataset interface."""
    snapshot = verify_dataset(dataset, V2)
    check_frozen(dataset / "annotations.json", "dataset_snapshot_sha256")
    check_frozen(dataset / "manifest.json", "dataset_manifest_sha256")
    require(
        {
            s: sum(r.split == s for r in snapshot.frames)
            for s in ("train", "val", "test")
        }
        == {"train": 88, "val": 11, "test": 12},
        "Expected original 111-frame v2 snapshot",
    )
    return snapshot


def label_truth(label: Path, width: int, height: int) -> tuple[Detection, ...]:
    """Parse verified nine-decimal YOLO boxes with rounding-only edge clipping."""
    truth: list[Detection] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        values = tuple(float(part) for part in line.split())
        require(
            len(values) == 5 and all(math.isfinite(v) for v in values),
            f"Invalid YOLO row: {label}",
        )
        class_id, cx, cy, w, h = values
        require(
            class_id in (0, 1) and w > 0 and h > 0,
            f"Invalid YOLO class/extent: {label}",
        )
        raw = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        require(
            raw[0] >= -1e-9
            and raw[1] >= -1e-9
            and raw[2] <= 1 + 1e-9
            and raw[3] <= 1 + 1e-9,
            f"YOLO edge exceeds nine-decimal rounding: {label}",
        )
        box = (
            max(0.0, raw[0]) * width,
            max(0.0, raw[1]) * height,
            min(1.0, raw[2]) * width,
            min(1.0, raw[3]) * height,
        )
        require(box[0] < box[2] and box[1] < box[3], f"Empty YOLO box: {label}")
        truth.append(Detection(int(class_id), box))
    return tuple(truth)


def examples(dataset: Path, snapshot: Snapshot, split: Split) -> tuple[Example, ...]:
    """Build Examples directly from already validated v2 image and label bytes."""
    require(split in ("val", "test"), "Scoring accepts validation/test only")
    result: list[Example] = []
    for record in snapshot.frames:
        if record.split != split:
            continue
        image = dataset / "images" / split / record.filename
        label = dataset / "labels" / split / Path(record.filename).with_suffix(".txt")
        require(
            sha256(image) == record.image_sha256
            and sha256(label) == record.label_sha256,
            f"Verified input changed: {record.filename}",
        )
        with Image.open(image) as decoded:
            truth = label_truth(label, *decoded.size)
        require(
            (sum(d.class_id == 0 for d in truth), sum(d.class_id == 1 for d in truth))
            == (record.potato_count, record.stone_count),
            "Label counts differ from snapshot",
        )
        result.append(Example(image, truth))
    return tuple(result)


def private_copy(dataset: Path, snapshot: Snapshot, output: Path) -> Path:
    """Copy, never link, train/val only so label caches cannot touch source data."""
    private = output / "training_dataset"
    private.mkdir(exist_ok=False)
    hashes: dict[str, str] = {}
    for record in snapshot.frames:
        if record.split not in ("train", "val"):
            continue
        for kind, filename, expected in (
            ("images", record.filename, record.image_sha256),
            (
                "labels",
                str(Path(record.filename).with_suffix(".txt")),
                record.label_sha256,
            ),
        ):
            relative = Path(kind) / record.split / filename
            source, target = dataset / relative, private / relative
            require(
                not source.is_symlink() and sha256(source) == expected,
                f"Source changed before copy: {source}",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(source, target)
            require(
                sha256(target) == expected and not target.samefile(source),
                f"Private copy not independent: {target}",
            )
            hashes[relative.as_posix()] = sha256(target)
    yaml = private / "data.yaml"
    _ = yaml.write_text(
        "train: images/train\nval: images/val\nnames:\n  0: potato\n  1: stone\n",
        encoding="utf-8",
    )
    write_json(
        output / "data-provenance.json",
        {
            "original_dataset": str(dataset),
            "original111snapshot": asdict(snapshot),
            "original_snapshot_sha256": sha256(dataset / "annotations.json"),
            "original_manifest_sha256": sha256(dataset / "manifest.json"),
            "original_yaml_sha256": sha256(dataset / "data.yaml"),
            "private_dataset": str(private),
            "private_copy_hashes": hashes,
            "private_yaml_sha256": sha256(yaml),
            "trainer_splits": ["train", "val"],
            "copy_policy": "physical_copy_no_symlinks_no_hardlinks_test_external_excluded",
        },
    )
    return yaml
