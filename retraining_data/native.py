"""Strict standard-library parsing for native AI review and source manifests."""

from dataclasses import dataclass
from pathlib import Path

from retraining_data.schema import (
    Json,
    JsonMap,
    Label,
    Split,
    array,
    boolean,
    document,
    integer,
    mapping,
    number,
    require,
    split_name,
    text,
)


@dataclass(frozen=True, slots=True)
class Flags:
    needs_confirmation: bool


@dataclass(frozen=True, slots=True)
class Review:
    eligible_for_experimental_training: bool
    source_json_sha256: str
    source_image_sha256: str
    requested_second: float
    actual_second: float


@dataclass(frozen=True, slots=True)
class Shape:
    label: Label
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class Native:
    flags: Flags
    shapes: tuple[Shape, ...]
    imagePath: str
    imageWidth: int
    imageHeight: int
    review: Review


@dataclass(frozen=True, slots=True)
class ReviewRow:
    filename: str
    relative_image: str
    requested_second: float
    actual_second: float
    review_status: str
    eligible_for_experimental_training: bool
    after_potato: int
    after_stone: int


@dataclass(frozen=True, slots=True)
class ReviewManifest:
    source_source_video_sha256: str
    source_manifest_sha256: str
    frames: tuple[ReviewRow, ...]


@dataclass(frozen=True, slots=True)
class PackRow:
    filename: str
    requested_second: float
    actual_second: float
    pts: int
    time_base: str
    pixel_sha256: str


@dataclass(frozen=True, slots=True)
class Pack:
    source_video: str
    source_sha256: str
    frames: tuple[PackRow, ...]


@dataclass(frozen=True, slots=True)
class LegacyRow:
    second: int
    split: Split
    file: str
    sha256: str
    pts: int
    time_base: str
    potato_count: int
    stone_count: int


@dataclass(frozen=True, slots=True)
class LegacyManifest:
    source: str
    source_sha256: str
    annotations_sha256: str
    baseline_manifest_sha256: str
    baseline_annotations_sha256: str
    frames: tuple[LegacyRow, ...]


def expect(row: JsonMap, key: str, expected: str | bool | int) -> None:
    """Require native flag types as well as values (0 is not false)."""
    value = row[key]
    require(type(value) is type(expected) and value == expected, f"Unexpected native field: {key}")


def parse_shape(value: Json) -> Shape:
    row = mapping(value)
    expect(row, "shape_type", "rectangle")
    label = text(row["label"])
    require(label in ("potato", "stone"), "Unknown native class")
    parsed_label: Label = "potato" if label == "potato" else "stone"
    points: list[tuple[float, float]] = []
    for item in array(row["points"]):
        pair = array(item)
        require(len(pair) == 2, "Expected two coordinates per corner")
        points.append((number(pair[0]), number(pair[1])))
    return Shape(label=parsed_label, points=tuple(points))


def read_native(data: bytes) -> Native:
    row = document(data)
    flags, review = mapping(row["flags"]), mapping(row["review"])
    expect(row, "checked", False)
    expect(row, "imageWidth", 1280)
    expect(row, "imageHeight", 720)
    for key, value in (("model_proposal", False), ("ai_visual_reviewed", True), ("human_reviewed", False)):
        expect(flags, key, value)
    expect(review, "reviewer_type", "AI_visual_review")
    expect(review, "human_reviewed", False)
    expect(review, "training_inclusion", False)
    return Native(
        flags=Flags(needs_confirmation=boolean(flags["needs_confirmation"])),
        shapes=tuple(parse_shape(v) for v in array(row["shapes"])),
        imagePath=text(row["imagePath"]), imageWidth=1280, imageHeight=720,
        review=Review(
            eligible_for_experimental_training=boolean(review["eligible_for_experimental_training"]),
            source_json_sha256=text(review["source_json_sha256"]),
            source_image_sha256=text(review["source_image_sha256"]),
            requested_second=number(review["requested_second"]), actual_second=number(review["actual_second"]),
        ),
    )


def read_review_manifest(path: Path) -> ReviewManifest:
    row = document(path.read_bytes())
    expect(row, "source_pack", "postprocess_review/outputs/review_pack")
    expect(row, "reviewer_type", "AI_visual_review_not_human_expert")
    for key in ("human_reviewed", "native_checked", "training_inclusion"):
        expect(row, key, False)
    records: list[ReviewRow] = []
    for value in array(row["frames"]):
        item = mapping(value)
        expect(item, "human_reviewed", False)
        expect(item, "native_checked", False)
        records.append(ReviewRow(
            filename=text(item["filename"]), relative_image=text(item["relative_image"]),
            requested_second=number(item["requested_second"]), actual_second=number(item["actual_second"]),
            review_status=text(item["review_status"]),
            eligible_for_experimental_training=boolean(item["eligible_for_experimental_training"]),
            after_potato=integer(item["after_potato"]), after_stone=integer(item["after_stone"]),
        ))
    return ReviewManifest(text(row["source_source_video_sha256"]), text(row["source_manifest_sha256"]), tuple(records))


def read_pack(path: Path) -> Pack:
    row = document(path.read_bytes())
    expect(row, "frame_count", 50)
    records: list[PackRow] = []
    for value in array(row["frames"]):
        item = mapping(value)
        expect(item, "exact_pixel_duplicate", False)
        records.append(PackRow(
            filename=text(item["filename"]), requested_second=number(item["requested_second"]),
            actual_second=number(item["actual_second"]), pts=integer(item["pts"]),
            time_base=text(item["time_base"]), pixel_sha256=text(item["pixel_sha256"]),
        ))
    return Pack(text(row["source_video"]), text(row["source_sha256"]), tuple(records))


def read_legacy(path: Path) -> LegacyManifest:
    row = document(path.read_bytes())
    records: list[LegacyRow] = []
    for value in array(row["frames"]):
        item = mapping(value)
        records.append(LegacyRow(
            second=integer(item["second"]), split=split_name(item["split"]), file=text(item["file"]),
            sha256=text(item["sha256"]), pts=integer(item["pts"]), time_base=text(item["time_base"]),
            potato_count=integer(item["potato_count"]), stone_count=integer(item["stone_count"]),
        ))
    return LegacyManifest(
        text(row["source"]), text(row["source_sha256"]), text(row["annotations_sha256"]),
        text(row["baseline_manifest_sha256"]), text(row["baseline_annotations_sha256"]), tuple(records),
    )
