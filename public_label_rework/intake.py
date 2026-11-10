"""Extract and verify the public image subset from the frozen dataset ZIP."""

import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Final, TypedDict

from public_label_rework.common import OUTPUTS, ZIP, new_output, sha256, write_json
from public_label_rework.protect import snapshot
from retraining_data.schema import (
    DataError,
    Json,
    array,
    document,
    mapping,
    require,
    text,
)

PUBLIC_PREFIX: Final = "images/public/"
SOURCE_PREFIX: Final = "sources/"
SOURCE_NAMES: Final = (
    "original01",
    "original02",
    "original03",
    "original06",
    "original07",
)
EXPECTED_IMAGES: Final = 2060
EXPECTED_BOXES: Final = 9498


class ManifestRecord(TypedDict):
    """Manifest identity fields used by intake."""

    image: str
    label: str
    source_prefix: str
    image_sha256: str


def member_path(name: str, destination: Path) -> Path:
    """Map a safe ZIP member to a path under destination."""
    relative = Path(name)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"Unsafe ZIP member {name}",
    )
    target = destination / relative
    require(
        target.resolve().is_relative_to(destination.resolve()),
        f"ZIP path escapes output: {name}",
    )
    return target


def copy_members(archive: zipfile.ZipFile, destination: Path) -> None:
    """Copy only public images, source attribution, and required metadata."""
    chosen = [
        item
        for item in archive.infolist()
        if item.filename.startswith((PUBLIC_PREFIX, SOURCE_PREFIX))
        or item.filename in {"manifest.json", "사용안내.txt"}
    ]
    for item in chosen:
        if item.is_dir():
            member_path(item.filename, destination).mkdir(parents=True, exist_ok=True)
            continue
        target = member_path(item.filename, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(item) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def license_receipts(source_root: Path) -> dict[str, dict[str, str]]:
    """Extract URL and CC BY 4.0 license lines from each source's metadata."""
    receipts: dict[str, dict[str, str]] = {}
    for source in SOURCE_NAMES:
        source_path = source_root / source
        files = ("README.roboflow.txt", "data.yaml")
        contents = {
            name: (source_path / name).read_text(encoding="utf-8", errors="replace")
            for name in files
        }
        urls = [
            line.strip()
            for text in contents.values()
            for line in text.splitlines()
            if re.search(r"https?://", line)
        ]
        licenses = [
            line.strip()
            for text in contents.values()
            for line in text.splitlines()
            if "CC BY 4.0" in line or "CC-BY-4.0" in line
        ]
        require(bool(urls), f"{source} is missing a URL")
        require(bool(licenses), f"{source} is missing a CC BY 4.0 license line")
        receipts[source] = {"url": urls[0], "license": licenses[0]}
    return receipts


def manifest_records(manifest: dict[str, Json]) -> tuple[ManifestRecord, ...]:
    """Parse public manifest entries into typed records."""
    records: list[ManifestRecord] = []
    for raw_value in array(manifest.get("records")):
        raw = mapping(raw_value)
        image = text(raw.get("image"))
        if image.startswith(PUBLIC_PREFIX):
            records.append(
                ManifestRecord(
                    image=image,
                    label=text(raw.get("label")),
                    source_prefix=text(raw.get("source_prefix")),
                    image_sha256=text(raw.get("image_sha256")),
                )
            )
    return tuple(records)


def label_box_count(path: Path) -> int:
    """Count label shapes at the JSON input boundary."""
    value = document(path.read_bytes())
    return len(array(value.get("shapes")))


def intake_from_zip(
    archive_path: Path,
    outputs: Path,
    *,
    protect: bool,
    expected_images: int = EXPECTED_IMAGES,
    expected_boxes: int = EXPECTED_BOXES,
) -> dict[str, object]:
    """Produce a verified public-only copy and its intake receipt."""
    require(archive_path.is_file(), f"Dataset ZIP not found: {archive_path}")
    outputs.mkdir(parents=True, exist_ok=True)
    require(outputs.resolve() == outputs, "Outputs root must not be a symlink")
    require(
        not (outputs / "source").exists() and not (outputs / "source").is_symlink(),
        "Refusing existing source output",
    )
    if protect:
        snapshot()
    source_output = new_output(outputs / "source")
    with zipfile.ZipFile(archive_path) as archive:
        copy_members(archive, source_output)
    manifest_value = document((source_output / "manifest.json").read_bytes())
    all_records = manifest_records(manifest_value)
    records = tuple(
        record for record in all_records if record["image"].startswith(PUBLIC_PREFIX)
    )
    images = sorted((source_output / PUBLIC_PREFIX).glob("*.jpg"))
    labels = sorted((source_output / PUBLIC_PREFIX).glob("*.json"))
    require(
        len(images) == expected_images,
        f"Expected {expected_images} JPGs, found {len(images)}",
    )
    require(
        len(labels) == expected_images,
        f"Expected {expected_images} labels, found {len(labels)}",
    )
    require(
        len(records) == expected_images,
        f"Expected {expected_images} public manifest records, found {len(records)}",
    )
    mismatches: list[str] = []
    by_image = {record["image"]: record for record in records}
    boxes = 0
    source_counts: Counter[str] = Counter()
    for image in images:
        relative_image = image.relative_to(source_output).as_posix()
        label_name = by_image[relative_image]["label"]
        label = source_output / label_name
        require(label.is_file(), f"Missing label for {relative_image}")
        if sha256(image) != by_image[relative_image]["image_sha256"]:
            mismatches.append(relative_image)
        boxes += label_box_count(label)
        source_id = image.stem.split("_")[1]
        source_counts[source_id] += 1
    require(not mismatches, f"Image hash mismatches: {mismatches[:5]}")
    require(boxes == expected_boxes, f"Expected {expected_boxes} boxes, found {boxes}")
    expected_labels = {record["label"] for record in records}
    require(
        len(expected_labels) == expected_images
        and {label.relative_to(source_output).as_posix() for label in labels}
        == expected_labels,
        "Public image/label pairing is not one-to-one",
    )
    licenses = license_receipts(source_output / SOURCE_PREFIX)
    public_sources = mapping(manifest_value.get("public_sources"))
    require(
        set(SOURCE_NAMES) <= set(public_sources),
        "manifest.public_sources is missing an expected source",
    )
    for source, details_value in public_sources.items():
        if source in SOURCE_NAMES:
            details = mapping(details_value)
            url = details.get("url")
            license_name = details.get("license")
            source_url, source_license = text(url), text(license_name)
            require(source_url.startswith("https://"), f"{source} URL is missing")
            require(source_license == "CC BY 4.0", f"{source} license is not CC BY 4.0")
            licenses[source] = {"url": source_url, "license": source_license}
    receipt: dict[str, object] = {
        "public_images": len(images),
        "public_labels": len(labels),
        "boxes": boxes,
        "per_source": {source: source_counts[source] for source in SOURCE_NAMES},
        "hash_mismatches": len(mismatches),
        "zip_sha256": sha256(archive_path),
        "licenses": licenses,
    }
    write_json(outputs / "intake.json", receipt)
    return receipt


def main() -> None:
    """Run the real dataset intake once."""
    receipt = intake_from_zip(ZIP, OUTPUTS, protect=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (DataError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
