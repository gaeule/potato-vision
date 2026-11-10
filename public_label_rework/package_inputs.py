"""Typed file boundary for package inputs, metrics, statuses and source credits."""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

from retraining_data.schema import (
    DataError,
    array,
    boolean,
    document,
    integer,
    mapping,
    require,
    text,
)

type PackageMode = Literal["pilot", "final"]
type ImageStatus = Literal["kept", "excluded", "needs_human"]
ASCII_NAME: Final = re.compile(r"[A-Za-z0-9._/-]+\Z")

METRICS_SCHEMA: Final = (
    "mode,verification_status,gfull_evaluations,thresholds,harvest_convention,"
    "fixture_only,per_source"
)


class SourceMetrics(TypedDict):
    background_potatoes: int
    watermark_stock: int
    thresholds: dict[str, str]


class PackageMetrics(TypedDict):
    """Exact metrics JSON fields consumed by the package README."""

    mode: PackageMode
    verification_status: str
    gfull_evaluations: int
    thresholds: dict[str, str]
    background_potatoes: int
    watermark_stock: int
    harvest_convention: str
    fixture_only: bool
    per_source: dict[str, SourceMetrics]


class SourceCredit(TypedDict):
    """One source attribution parsed from unchanged metadata."""

    source: str
    url: str
    license: str


class StatusRow(TypedDict):
    """Apply status fields consumed by packaging and README rendering."""

    image: str
    status: ImageStatus


def read_metrics(path: Path) -> PackageMetrics:
    """Parse mode, verification, PASS/FAIL tokens, and displayed counts."""
    try:
        row = mapping(document(path.read_bytes()))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DataError(f"Invalid package metrics JSON: {path}") from error
    require(set(row) == set(METRICS_SCHEMA.split(",")), "Metrics fields mismatch")
    mode = text(row["mode"])
    modes: dict[str, PackageMode] = {"pilot": "pilot", "final": "final"}
    require(mode in modes, f"Unknown package mode: {mode}")
    parsed_mode = modes[mode]
    thresholds = mapping(row["thresholds"])
    parsed_thresholds = {text(key): text(value) for key, value in thresholds.items()}
    require(
        all(value in ("PASS", "FAIL") for value in parsed_thresholds.values()),
        "Metrics tokens must be PASS or FAIL",
    )
    evaluations = integer(row["gfull_evaluations"])
    require(evaluations >= 0, "G-full evaluation count must be nonnegative")
    require(bool(parsed_thresholds), "At least one metric token is required")
    source_metrics: dict[str, SourceMetrics] = {}
    for source, source_value in mapping(row["per_source"]).items():
        details = mapping(source_value)
        require(
            set(details) == {"background_potatoes", "watermark_stock", "thresholds"},
            f"{source} metric fields mismatch",
        )
        local_thresholds = {
            text(key): text(value)
            for key, value in mapping(details["thresholds"]).items()
        }
        require(
            bool(local_thresholds)
            and all(value in ("PASS", "FAIL") for value in local_thresholds.values()),
            f"{source} metric tokens must be PASS or FAIL",
        )
        local_background = integer(details["background_potatoes"])
        local_watermark = integer(details["watermark_stock"])
        require(min(local_background, local_watermark) >= 0, "Negative source metric")
        source_metrics[text(source)] = {
            "background_potatoes": local_background,
            "watermark_stock": local_watermark,
            "thresholds": local_thresholds,
        }
    require(bool(source_metrics), "Per-source metrics are required")
    return PackageMetrics(
        mode=parsed_mode,
        verification_status=text(row["verification_status"]),
        gfull_evaluations=evaluations,
        thresholds=parsed_thresholds,
        background_potatoes=sum(
            source["background_potatoes"] for source in source_metrics.values()
        ),
        watermark_stock=sum(
            source["watermark_stock"] for source in source_metrics.values()
        ),
        harvest_convention=text(row["harvest_convention"]),
        fixture_only=boolean(row["fixture_only"]),
        per_source=source_metrics,
    )


def read_status(path: Path) -> tuple[StatusRow, ...]:
    """Parse unique task9 status rows into the closed status type."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames == ["image", "status", "reason", "batch"],
            "Unexpected status.csv header",
        )
        rows = tuple(
            StatusRow(
                image=text(row["image"]),
                status=parse_status(text(row["status"])),
            )
            for row in reader
        )
    require(bool(rows), "status.csv is empty")
    require(len({row["image"] for row in rows}) == len(rows), "Duplicate status ID")
    return rows


def source_hashes(metadata: Path) -> dict[str, str]:
    """Parse the verified intake manifest into source image hashes."""
    manifest = document((metadata / "manifest.json").read_bytes())
    hashes: dict[str, str] = {}
    for value in array(manifest["records"]):
        record = mapping(value)
        image = text(record["image"])
        if image.startswith("images/public/") and image.endswith(".jpg"):
            stem = Path(image).stem
            require(stem not in hashes, f"Duplicate source manifest image: {stem}")
            hashes[stem] = text(record["image_sha256"])
    return hashes


def parse_status(value: str) -> ImageStatus:
    """Parse a status.csv value into the closed package status set."""
    statuses: dict[str, ImageStatus] = {
        "kept": "kept",
        "excluded": "excluded",
        "needs_human": "needs_human",
    }
    require(value in statuses, f"Unknown image status: {value}")
    return statuses[value]


def source_credits(metadata: Path) -> tuple[SourceCredit, ...]:
    """Extract attribution for every preserved source from the intake manifest."""
    source_root = metadata / "sources"
    require(
        source_root.is_dir() and not source_root.is_symlink(),
        f"Missing source metadata directory: {source_root}",
    )
    for path in source_root.rglob("*"):
        require(
            not path.is_symlink()
            and all(
                ASCII_NAME.fullmatch(part) for part in path.relative_to(metadata).parts
            ),
            "Unsafe source metadata path",
        )
    manifest = document((metadata / "manifest.json").read_bytes())
    registry = mapping(manifest["public_sources"])
    source_directories = {path.name for path in source_root.iterdir() if path.is_dir()}
    require(bool(source_directories), "Source metadata directories are missing")
    credits: list[SourceCredit] = []
    for source in sorted(source_directories):
        details = mapping(registry.get(source))
        url = text(details["url"])
        license_name = text(details["license"])
        require(url.startswith("https://"), f"{source} source URL missing")
        require(license_name == "CC BY 4.0", f"{source} license mismatch")
        credits.append({"source": source, "url": url, "license": license_name})
    return tuple(credits)


def source_id(image: str) -> str:
    """Get a public source identifier from the fixed image naming convention."""
    parts = image.split("_")
    require(
        len(parts) >= 3 and parts[0] == "public" and bool(parts[1]),
        f"Invalid public image name: {image}",
    )
    return parts[1]


@dataclass(frozen=True, slots=True)
class PackageInput:
    """Explicit inputs for one package build."""

    source: Path
    source_metadata: Path
    applied: Path
    metrics: Path
    renders: Path
    output: Path
    verification: Path | None = None
