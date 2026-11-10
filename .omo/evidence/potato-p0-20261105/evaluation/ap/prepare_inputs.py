"""One-shot, standard-library-only private input preparation; never loads models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Final

HERE: Final = Path(__file__).resolve().parent
ROOT: Final = HERE.parents[4]
EXPECTED: Final = {
    "current": (
        "retraining_experiment/outputs/selection/selected.pt",
        "8b2cb7982fcd11f61a33cab8659de23128d0065cfa8e8966093be297a7e216c8",
    ),
    "stone40": (
        "stone_experiment/outputs/selection/selected.pt",
        "06d46e0dc728ffbabf4078f79acaf9d32d809a57a46810acdf39a7927aa2bbd1",
    ),
}


def sha(path: Path) -> str:
    """Hash file bytes without deserializing a checkpoint."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Create new copies, verifying both endpoints and stored historical bindings."""
    sys.dont_write_bytecode = True
    receipt = HERE / "inputs-manifest.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    copies: list[dict[str, str]] = []
    sources: dict[str, str] = {}

    def copy(source: Path, destination: Path) -> None:
        if not destination.resolve().is_relative_to(HERE):
            raise RuntimeError("Private destination escaped AP directory")
        if destination.exists() or source.is_symlink():
            raise FileExistsError(destination)
        before = sha(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if before != sha(destination) or before != sha(source):
            raise RuntimeError(f"Copy changed: {source}")
        sources[str(source)] = before
        copies.append({"source": str(source), "private": str(destination), "sha256": before})

    historical = ROOT / "retraining_experiment/review/protocol.json"
    old = json.loads(historical.read_text())
    dataset = ROOT / "retraining_data/outputs/dataset_v2"
    if sha(dataset / "manifest.json") != old["dataset_manifest_sha256"]:
        raise RuntimeError("dataset_v2 historical manifest mismatch")
    if sha(dataset / "annotations.json") != old["dataset_snapshot_sha256"]:
        raise RuntimeError("dataset_v2 historical annotation mismatch")
    annotations = json.loads((dataset / "annotations.json").read_text())
    protected = json.loads(
        (ROOT / "retraining_experiment/review/protected_before.json").read_text()
    )
    for name, (relative, expected) in EXPECTED.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Checkpoint mismatch: {name}")
        copy(ROOT / relative, HERE / "private/models" / f"{name}.pt")

    membership: dict[str, list[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for name, split, count, source in (
        ("val11", "val", 11, dataset),
        ("test12", "test", 12, dataset),
        ("external166", "test", 166, ROOT / "baseline/external"),
    ):
        images = sorted((source / "images" / split).glob("*.png"))
        labels = sorted((source / "labels" / split).glob("*.txt"))
        if len(images) != count or {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f"Image/label membership mismatch: {name}")
        frames = {f["filename"]: f for f in annotations["frames"] if f["split"] == split}
        if name != "external166":
            if set(frames) != {p.name for p in images}:
                raise RuntimeError(f"Snapshot membership mismatch: {name}")
        membership[name] = [p.name for p in images]
        counts[name] = {"0": 0, "1": 0}
        for image in images:
            label = source / "labels" / split / image.with_suffix(".txt").name
            for original in (image, label):
                if name == "external166":
                    expected = protected[str(original.relative_to(ROOT))]
                else:
                    field = "image_sha256" if original == image else "label_sha256"
                    expected = frames[image.name][field]
                if sha(original) != expected:
                    raise RuntimeError(f"Historical input hash mismatch: {original}")
                copy(original, HERE / "private" / name / original.relative_to(source))
            for line in label.read_text().splitlines():
                fields = line.split()
                allowed = ("0",) if name == "external166" else ("0", "1")
                if len(fields) != 5 or fields[0] not in allowed:
                    raise RuntimeError(f"Invalid class/label: {label}")
                values = [float(v) for v in fields[1:]]
                if not all(0 <= v <= 1 for v in values) or min(values[2:]) <= 0:
                    raise RuntimeError(f"Invalid normalized box: {label}")
                counts[name][fields[0]] += 1
    for rel in (
        "retraining_experiment/review/protocol.json",
        "retraining_experiment/outputs/comparison/comparison.json",
        "retraining_experiment/outputs/comparison/environment.json",
        "retraining_experiment/outputs/comparison/provenance.json",
        "retraining_experiment/review/protected_before.json",
        "retraining_data/outputs/dataset_v2/manifest.json",
        "retraining_data/outputs/dataset_v2/annotations.json",
        "retraining_data/outputs/dataset_v2/data.yaml",
        "baseline/external/manifest.json",
        "baseline/external/data.yaml",
        "stone_experiment/review/protocol.json",
    ):
        copy(ROOT / rel, HERE / "private/reference" / rel)
    copy(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), HERE / "private/font.ttf")
    for package in ("retraining_experiment", "stone_experiment"):
        for p in sorted((ROOT / package).glob("*.py")):
            sources[str(p)] = sha(p)
    for rel in (
        "postprocess_review/common.py", "scripts/run_baseline.py",
        "scripts/baseline_common.py", "retraining_data/common.py",
        "retraining_data/schema.py", "stone_experiment/schema.py",
    ):
        sources[str(ROOT / rel)] = sha(ROOT / rel)
    installed = Path(str(importlib.metadata.distribution("ultralytics").locate_file("ultralytics")))
    library = {
        str(p): sha(p)
        for p in sorted(installed.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
    }
    config_hashes = {
        str(p): sha(p) for p in sorted((HERE / "private").glob("*/data.yaml"))
    }
    document = {
        "schema": 1,
        "python": sys.version,
        "executable": sys.executable,
        "executable_sha256": sha(Path(sys.executable).resolve()),
        "installed_versions": {
            d.metadata["Name"]: d.version
            for d in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"])
        },
        "ultralytics_root": str(installed),
        "installed_source_hashes": library,
        "source_hashes": sources,
        "private_copies": copies,
        "config_hashes": config_hashes,
        "membership": membership,
        "gt_instances": counts,
    }
    with receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "manifest_sha256": sha(receipt), "copies": len(copies),
        "installed_sources": len(library), "gt_instances": counts,
        "model_imported": "ultralytics" in sys.modules,
    }))


if __name__ == "__main__":
    main()
