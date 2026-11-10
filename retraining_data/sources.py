"""Construct a reproducible preparation plan from read-only source artifacts."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from retraining_data.common import (
    LEGACY,
    PACK,
    ROOT,
    VIDEO,
    digest,
    eligibility,
    file_hash,
    image_pixels,
    yolo,
)
from retraining_data.native import (
    read_legacy,
    read_native,
    read_pack,
    read_review_manifest,
)
from retraining_data.schema import (
    V1,
    V2,
    Exclusion,
    Profile,
    Record,
    Snapshot,
    SourceFile,
    array,
    document,
    integer,
    mapping,
    number,
    require,
    safe_filename,
    text,
    validate_records,
)
from stone_experiment.prepare import verify_dataset as verify_legacy


@dataclass(frozen=True, slots=True)
class Payload:
    """Bytes to publish with one immutable output record."""

    record: Record
    label: bytes


@dataclass(frozen=True, slots=True)
class Plan:
    snapshot: Snapshot
    payloads: tuple[Payload, ...]
    profile: Profile = V1


class Inventory:
    """Mutable accumulator for files actually read, including full JSON snapshots."""

    def __init__(self) -> None:
        self.files: dict[str, SourceFile] = {}

    def add(self, path: Path) -> str:
        """Hash a workspace source; reject source-tree symlink escapes."""
        require(path.resolve().is_relative_to(ROOT), f"Source escapes workspace: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if relative not in self.files:
            self.files[relative] = SourceFile(
                path=relative, sha256=file_hash(path),
                text=path.read_text(encoding="utf-8") if path.suffix in (".json", ".yaml") else None,
            )
        return self.files[relative].sha256


def build_plan(profile: Profile = V1) -> Plan:
    """Check the unchanged 63-frame verifier, then derive the bounded profile."""
    review = ROOT / profile.review_directory
    old_frames = verify_legacy(LEGACY)
    inventory = Inventory()
    old_manifest = read_legacy(LEGACY / "manifest.json")
    review_manifest = read_review_manifest(review / "manifest.json")
    pack = read_pack(PACK / "manifest.json")
    for path in (LEGACY / "manifest.json", LEGACY / "annotations.json", LEGACY / "data.yaml",
                 review / "manifest.json", PACK / "manifest.json"):
        _ = inventory.add(path)
    if profile == V2:
        _ = document((review / "user_confirmation.json").read_bytes())
        _ = inventory.add(review / "user_confirmation.json")
    video_hash = inventory.add(VIDEO)
    require(
        old_manifest.source == pack.source_video == VIDEO.name
        and old_manifest.source_sha256 == review_manifest.source_source_video_sha256 == pack.source_sha256 == video_hash,
        "Source video identity/hash differs",
    )
    require(review_manifest.source_manifest_sha256 == inventory.add(PACK / "manifest.json"), "Review pack hash differs")
    for path, expected in (
        (ROOT / "stone_experiment/annotations.json", old_manifest.annotations_sha256),
        (ROOT / "baseline/internal/manifest.json", old_manifest.baseline_manifest_sha256),
        (ROOT / "baseline/annotations.json", old_manifest.baseline_annotations_sha256),
    ):
        require(inventory.add(path) == expected, f"Legacy provenance hash differs: {path}")
    old_metadata = {row.file: row for row in old_manifest.frames}
    baseline_document = document((ROOT / "baseline/internal/manifest.json").read_bytes())
    baseline_rows = [mapping(value) for value in array(baseline_document["frames"])]
    baseline_metadata = {text(row["file"]): row for row in baseline_rows}
    require(len(old_metadata) == len(old_frames) == len(old_manifest.frames), "Legacy manifest duplicate/missing rows")
    legacy_pixels: dict[str, str] = {}
    payloads: list[Payload] = []
    excluded_old: list[str] = []
    for frame in old_frames:
        metadata = old_metadata[frame.filename]
        image = LEGACY / "images" / frame.split / frame.filename
        label_path = LEGACY / "labels" / frame.split / Path(frame.filename).with_suffix(".txt")
        pixel = image_pixels(image)
        legacy_pixels[frame.filename] = pixel
        require(inventory.add(image) == metadata.sha256, f"Legacy image hash differs: {image}")
        _ = inventory.add(label_path)
        require(
            (metadata.second, metadata.split, metadata.potato_count, metadata.stone_count)
            == (frame.second, frame.split, len(frame.boxes), len(frame.stones)),
            f"Legacy record differs: {frame.filename}",
        )
        actual = float(metadata.pts * Fraction(metadata.time_base))
        baseline = baseline_metadata[frame.filename]
        require(
            (metadata.pts, metadata.time_base, actual)
            == (integer(baseline["pts"]), text(baseline["time_base"]), number(baseline["actual_second"])),
            f"Legacy timestamps differ from baseline: {frame.filename}",
        )
        if frame.second in profile.excluded_seconds:
            require(frame.split == "train", "Excluded legacy slab outside train")
            excluded_old.append(frame.filename)
            continue
        label = label_path.read_bytes()
        payloads.append(Payload(Record(
            filename=frame.filename, split=metadata.split, requested_second=float(frame.second),
            actual_second=actual, pts=metadata.pts, time_base=metadata.time_base,
            provenance="legacy_AI_visual_labels", source_image=image.relative_to(ROOT).as_posix(),
            source_annotation=label_path.relative_to(ROOT).as_posix(), image_sha256=metadata.sha256,
            pixel_sha256=pixel, label_sha256=digest(label),
            potato_count=len(frame.boxes), stone_count=len(frame.stones),
        ), label))
    require(len(excluded_old) == len(profile.excluded_seconds), "Unexpected legacy exclusions")
    pack_rows = {row.filename: row for row in pack.frames}
    require(len(pack_rows) == len(pack.frames) == 50, "Review pack duplicate/missing rows")
    require(
        len(review_manifest.frames) == 50 and {r.filename for r in review_manifest.frames} == set(pack_rows),
        "Review manifest must contain exactly the source pack's 50 unique frames",
    )
    selected: list[str] = []
    excluded_review: list[str] = []
    seen_pixels = set(legacy_pixels.values())
    for row in review_manifest.frames:
        # Path is validated before it is used for file I/O.
        safe_filename(row.filename)
        directory = "candidates" if row.eligible_for_experimental_training else "needs_confirmation"
        image = review / directory / row.filename
        native_path = image.with_suffix(".json")
        native = read_native(native_path.read_bytes())
        eligible = eligibility(row, native)
        source = pack_rows[row.filename]
        require(
            (source.requested_second, source.actual_second) == (row.requested_second, row.actual_second)
            and 0 <= row.requested_second <= 84 and 0 <= row.actual_second <= 84
            and abs(source.actual_second - float(source.pts * Fraction(source.time_base))) < 1e-9,
            f"Source frame time differs: {row.filename}",
        )
        image_hash = inventory.add(image)
        pixel = image_pixels(image)
        _ = inventory.add(native_path)
        require(
            image_hash == native.review.source_image_sha256 == inventory.add(PACK / "data" / row.filename)
            and native.review.source_json_sha256 == inventory.add((PACK / "data" / row.filename).with_suffix(".json"))
            and pixel == source.pixel_sha256,
            f"Native/source image, JSON, or decoded pixel hash differs: {row.filename}",
        )
        label = yolo(native)
        if not eligible:
            excluded_review.append(row.filename)
            continue
        require(pixel not in seen_pixels, f"Review RGB pixels duplicate legacy or review image: {row.filename}")
        seen_pixels.add(pixel)
        selected.append(row.filename)
        payloads.append(Payload(Record(
            filename=row.filename, split="train", requested_second=row.requested_second,
            actual_second=row.actual_second, pts=source.pts, time_base=source.time_base,
            provenance="AI_visual_review_not_human_expert",
            source_image=image.relative_to(ROOT).as_posix(),
            source_annotation=native_path.relative_to(ROOT).as_posix(), image_sha256=image_hash,
            pixel_sha256=pixel, label_sha256=digest(label),
            potato_count=row.after_potato, stone_count=row.after_stone,
        ), label))
    require(len(selected) == profile.candidates and len(excluded_review) == profile.held,
            f"Expected exactly {profile.candidates} candidates and {profile.held} held frames")
    for directory, names in (("candidates", selected), ("needs_confirmation", excluded_review)):
        actual_files = {p.name for p in (review / directory).iterdir() if p.is_file()}
        expected_files = set(names) | {str(Path(n).with_suffix(".json")) for n in names}
        require(actual_files == expected_files, f"Review folder membership differs: {directory}")
    ordered = tuple(sorted(payloads, key=lambda p: (p.record.requested_second, p.record.filename)))
    records = tuple(p.record for p in ordered)
    validate_records(records, profile)
    snapshot = Snapshot(
        frames=records, legacy_pixel_hashes=legacy_pixels,
        source_files=tuple(inventory.files[k] for k in sorted(inventory.files)),
        schema_version=f"retraining-dataset-{profile.name}",
        exclusions=((
            Exclusion(source="stone_experiment/outputs/dataset", observations=3, filenames=tuple(excluded_old),
                      reason="Whole frames 44/46/48: chief-confirmed same unconfirmed slab as held review43..49"),
        ) if excluded_old else ()) + (
            Exclusion(source=f"{profile.review_directory}/needs_confirmation",
                      observations=profile.held, filenames=tuple(excluded_review),
                      reason="Native needs_confirmation=true; ineligible AI review"),
            Exclusion(source="추가영상2_돌검수.zip", observations=166, filenames=(),
                      reason="External ZIP entirely excluded; not opened, decoded, or hashed"),
        ),
    )
    return Plan(snapshot, ordered, profile)
