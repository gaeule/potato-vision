"""Build a verified, reproducible delivery package from an applied tree."""

import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final

from public_label_rework.common import new_output
from public_label_rework.labelio import parse
from public_label_rework.package_gate import delivery_gate
from public_label_rework.package_inputs import (
    ASCII_NAME,
    PackageInput,
    SourceCredit,
    StatusRow,
    read_metrics,
    read_status,
    source_credits,
    source_hashes,
    source_id,
)
from public_label_rework.package_readme import readme
from retraining_data.schema import (
    DataError,
    require,
)

IMAGE_NAME: Final = re.compile(r"[A-Za-z0-9_-]+\Z")
__all__ = ["PackageInput", "build"]


def preflight(
    inputs: PackageInput, rows: tuple[StatusRow, ...]
) -> tuple[SourceCredit, ...]:
    """Verify source pairs, manifest hashes, applied membership, and credits."""
    image_ids = {row["image"] for row in rows}
    require(
        all(IMAGE_NAME.fullmatch(image) is not None for image in image_ids)
        and image_ids <= {path.stem for path in inputs.source.glob("*.jpg")}
        and image_ids <= {path.stem for path in inputs.source.glob("*.json")},
        "Verified source membership mismatch",
    )
    hashes = source_hashes(inputs.source_metadata)
    require(image_ids <= set(hashes), "Source manifest membership mismatch")
    for image in image_ids:
        source_jpg = inputs.source / f"{image}.jpg"
        source_json = inputs.source / f"{image}.json"
        require(
            not source_jpg.is_symlink()
            and not source_json.is_symlink()
            and hashlib.sha256(source_jpg.read_bytes()).hexdigest() == hashes[image],
            f"Source image hash mismatch: {image}",
        )
    kept_ids = {row["image"] for row in rows if row["status"] == "kept"}
    human_ids = {row["image"] for row in rows if row["status"] == "needs_human"}
    require(
        {path.stem for path in (inputs.applied / "needs_human").glob("*.json")}
        == human_ids
        and {
            path.stem
            for path in (inputs.applied / "labels/images/public").glob("*.json")
        }
        == kept_ids,
        "Applied kept or needs_human membership mismatch",
    )
    human_jpgs = {p.stem for p in (inputs.applied / "needs_human").glob("*.jpg")}
    require(human_jpgs == human_ids, "Applied needs_human JPG membership mismatch")
    for image in kept_ids | human_ids:
        folder = (
            inputs.applied / "labels/images/public"
            if image in kept_ids
            else inputs.applied / "needs_human"
        )
        doc = parse((folder / f"{image}.json").read_bytes())
        require(doc.imagePath == f"{image}.jpg", f"Renamed image reference: {image}")
        if image in kept_ids:
            require(
                all(shape.flags.get("tight") is True for shape in doc.shapes),
                f"Untight kept label: {image}",
            )
    require(
        not any((inputs.applied / "labels/images/public").glob("*.jpg")),
        "Unexpected JPG stored in applied kept labels",
    )
    for name in ("exclusions.csv", "changes.csv"):
        require((inputs.applied / name).is_file(), f"Missing apply artifact: {name}")
    require(inputs.renders.is_dir(), "Missing render sample directory")
    for path in inputs.renders.iterdir():
        require(
            path.is_file()
            and not path.is_symlink()
            and ASCII_NAME.fullmatch(path.name) is not None,
            f"Unsafe render sample: {path.name}",
        )
    return source_credits(inputs.source_metadata)


def copy_artifacts(
    inputs: PackageInput, root: Path, rows: tuple[StatusRow, ...]
) -> None:
    """Assemble package files with JPG bytes exclusively from verified source."""
    kept = root / "images/public"
    human = root / "needs_human"
    kept.mkdir(parents=True)
    if any(row["status"] == "needs_human" for row in rows):
        human.mkdir()
    for row in rows:
        image = row["image"]
        source_jpg = inputs.source / f"{image}.jpg"
        if row["status"] == "kept":
            applied_json = inputs.applied / f"labels/images/public/{image}.json"
            _ = shutil.copyfile(source_jpg, kept / f"{image}.jpg")
            _ = shutil.copyfile(applied_json, kept / f"{image}.json")
        elif row["status"] == "needs_human":
            label = inputs.applied / f"needs_human/{image}.json"
            _ = shutil.copyfile(source_jpg, human / f"{image}.jpg")
            _ = shutil.copyfile(label, human / f"{image}.json")
    for name in ("exclusions.csv", "changes.csv"):
        artifact = inputs.applied / name
        _ = shutil.copyfile(artifact, root / name)
    _ = shutil.copyfile(inputs.applied / "status.csv", root / "status.csv")
    before_after = root / "before_after"
    before_after.mkdir()
    for image in sorted(p for p in inputs.renders.iterdir() if p.is_file()):
        _ = shutil.copyfile(image, before_after / image.name)
    _ = shutil.copytree(inputs.source_metadata / "sources", root / "sources")


def write_zip(root: Path, archive_path: Path) -> str:
    """Write stable ASCII members and return the archive SHA-256."""
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            name = path.relative_to(root).as_posix()
            zip_path = PurePosixPath(name)
            require(
                ASCII_NAME.fullmatch(name) is not None
                and not zip_path.is_absolute()
                and ".." not in zip_path.parts,
                f"Unsafe ZIP name: {name}",
            )
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return digest


def build(inputs: PackageInput) -> tuple[Path, str]:
    """Validate all source contracts, then build one fresh package tree and ZIP."""
    metrics = read_metrics(inputs.metrics)
    rows = read_status(inputs.applied / "status.csv")
    delivery_gate(inputs, metrics, rows)
    require(
        set(metrics["per_source"]) == {source_id(row["image"]) for row in rows},
        "Per-source metrics membership mismatch",
    )
    credits = preflight(inputs, rows)
    kept_count = sum(row["status"] == "kept" for row in rows)
    human_count = sum(row["status"] == "needs_human" for row in rows)
    archive_path = inputs.output.parent / f"{inputs.output.name}.zip"
    receipt_path = (
        inputs.output.parent / f"{inputs.output.name}_archive_verification.json"
    )
    require(
        all(
            not path.exists() and not path.is_symlink()
            for path in (archive_path, receipt_path)
        ),
        "Refusing existing ZIP output",
    )
    root = new_output(inputs.output)
    copy_artifacts(inputs, root, rows)
    _ = (root / "README.md").write_text(
        readme(rows, metrics, credits), encoding="utf-8"
    )
    digest = write_zip(root, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "ZIP CRC verification failed")
        names = archive.namelist()
        require(names == sorted(names), "ZIP ordering is not deterministic")
        parsed = 0
        reparsed = 0
        for name in names:
            if name.endswith(".json"):
                _ = parse(archive.read(name))
                reparsed += 1
                if name.startswith("images/public/"):
                    parsed += 1
        require(parsed == kept_count, "ZIP label reparse count mismatch")
        require(
            reparsed == kept_count + human_count,
            "ZIP JSON reparse total mismatch",
        )
        member_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in names
        }
        verification = {
            "archive": archive_path.name,
            "sha256": digest,
            "bytes": archive_path.stat().st_size,
            "members": len(names),
            "crc": "passed",
            "reparsed_public_labels": parsed,
            "member_sha256": member_hashes,
        }
    _ = receipt_path.write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return archive_path, digest


def main() -> None:
    """CLI entry point; production callers pass explicit build inputs."""
    require(
        len(sys.argv) in (7, 9)
        and (len(sys.argv) == 7 or sys.argv[7] == "--verification"),
        "Usage: package.py SOURCE_DIR SOURCE_METADATA APPLIED_TREE METRICS_JSON RENDERS_DIR OUTPUT_DIR [--verification verification.json]",
    )
    inputs = PackageInput(
        *(Path(value) for value in sys.argv[1:7]),
        verification=Path(sys.argv[8]) if len(sys.argv) == 9 else None,
    )
    archive, digest = build(inputs)
    print(f"PACKAGE {archive}\nSHA256 {digest}")


if __name__ == "__main__":
    try:
        main()
    except (DataError, OSError, zipfile.BadZipFile) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
