"""Independent immutable fractional-time output schema and strict JSON boundary."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

type Split = Literal["train", "val", "test"]
type Label = Literal["potato", "stone"]
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type JsonMap = dict[str, Json]
RANGES: Final = {"train": (0, 84), "val": (92, 112), "test": (120, 147)}
EXPECTED: Final = {"train": (79, 417, 30), "val": (11, 60, 8), "test": (12, 33, 7)}
EXCLUDED_SECONDS: Final = frozenset({44, 46, 48})


@dataclass(frozen=True, slots=True)
class Profile:
    """Bounded source selection and output identity, shared by both commands."""

    name: Literal["v1", "v2"]
    review_directory: str
    dataset_directory: str
    candidates: int
    held: int
    excluded_seconds: frozenset[int]
    train_counts: tuple[int, int, int]


V1: Final = Profile("v1", "label_review", "dataset", 42, 8, EXCLUDED_SECONDS, (79, 417, 30))
V2: Final = Profile("v2", "label_review_v2", "dataset_v2", 48, 2, frozenset(), (88, 473, 39))


class DataError(RuntimeError):
    """A named dataset contract violation."""

    detail: str

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise DataError(detail)


def profile_named(name: str) -> Profile:
    profiles = {"v1": V1, "v2": V2}
    require(name in profiles, f"Unknown dataset profile: {name}")
    return profiles[name]


def mapping(value: Json) -> JsonMap:
    if not isinstance(value, dict):
        raise DataError("Expected JSON object")
    return value


def array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise DataError("Expected JSON array")
    return value


def text(value: Json) -> str:
    if not isinstance(value, str):
        raise DataError("Expected JSON string")
    return value


def integer(value: Json) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError("Expected JSON integer")
    return value


def number(value: Json) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DataError("Expected finite JSON number")
    return float(value)


def boolean(value: Json) -> bool:
    if not isinstance(value, bool):
        raise DataError("Expected JSON boolean")
    return value


def document(data: bytes, *, decoder: Callable[[bytes], Json] = json.loads) -> JsonMap:
    """Decode at the boundary; callers explicitly parse every consumed field."""
    return mapping(decoder(data))


def split_name(value: Json) -> Split:
    name = text(value)
    names: dict[str, Split] = {"train": "train", "val": "val", "test": "test"}
    require(name in names, "Unknown split")
    return names[name]


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    sha256: str
    text: str | None


@dataclass(frozen=True, slots=True)
class Record:
    filename: str
    split: Split
    requested_second: float
    actual_second: float
    pts: int
    time_base: str
    provenance: str
    source_image: str
    source_annotation: str
    image_sha256: str
    pixel_sha256: str
    label_sha256: str
    potato_count: int
    stone_count: int


@dataclass(frozen=True, slots=True)
class Exclusion:
    source: str
    reason: str
    observations: int
    filenames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Snapshot:
    frames: tuple[Record, ...]
    exclusions: tuple[Exclusion, ...]
    source_files: tuple[SourceFile, ...]
    legacy_pixel_hashes: dict[str, str]
    schema_version: str = "retraining-dataset-v1"
    human_reviewed: bool = False
    native_checked: bool = False
    counts_are: str = "frame_observations_not_unique_objects"
    duplicate_policy: str = "reject_cross_split_and_review_vs_all_legacy_RGB_pixels"


@dataclass(frozen=True, slots=True)
class Manifest:
    snapshot_sha256: str
    data_yaml_sha256: str
    label_hashes: dict[str, str]
    schema_version: str = "retraining-manifest-v1"


def json_bytes(value: Snapshot | Manifest) -> bytes:
    """Stable bytes for manifests and snapshots; no timestamps or output paths."""
    return (json.dumps(asdict(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_manifest(data: bytes, profile: Profile = V1) -> Manifest:
    row = document(data)
    require(set(row) == {"schema_version", "snapshot_sha256", "data_yaml_sha256", "label_hashes"},
            "Unexpected manifest schema fields")
    version = f"retraining-manifest-{profile.name}"
    require(row["schema_version"] == version, "Unexpected manifest version/profile")
    return Manifest(
        snapshot_sha256=text(row["snapshot_sha256"]), data_yaml_sha256=text(row["data_yaml_sha256"]),
        label_hashes={k: text(v) for k, v in mapping(row["label_hashes"]).items()},
        schema_version=version,
    )


def safe_filename(filename: str) -> None:
    """Accept arbitrary unique PNG basenames, never relative or absolute paths."""
    require(
        Path(filename).name == filename and "\\" not in filename
        and filename.endswith(".png") and len(filename) > 4,
        f"Unsafe PNG filename: {filename}",
    )


def validate_records(records: tuple[Record, ...], profile: Profile = V1) -> None:
    """Enforce membership, time isolation, counts and decoded cross-split uniqueness."""
    require(len({r.filename for r in records}) == len(records), "Duplicate output filenames")
    seen_pixels: dict[str, Split] = {}
    for record in records:
        safe_filename(record.filename)
        low, high = RANGES[record.split]
        require(
            math.isfinite(record.requested_second) and math.isfinite(record.actual_second)
            and low <= record.requested_second <= high
            and 0 <= record.actual_second - record.requested_second < 1,
            f"Temporal split violation: {record.filename}",
        )
        previous = seen_pixels.get(record.pixel_sha256)
        require(previous is None or previous == record.split, "Exact RGB pixel duplicate across splits")
        seen_pixels[record.pixel_sha256] = record.split
    for split, counts in EXPECTED.items():
        subset = [r for r in records if r.split == split]
        actual = (len(subset), sum(r.potato_count for r in subset), sum(r.stone_count for r in subset))
        expected = profile.train_counts if split == "train" else counts
        require(actual == expected, f"Unexpected {split} frame/potato/stone counts: {actual}")
