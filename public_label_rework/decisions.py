"""Immutable decision protocol; strict JSONL with coordinate-bound tool receipts."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

from public_label_rework.labelio import LabelDocument, Xyxy
from retraining_data.schema import (
    DataError,
    Json,
    JsonMap,
    array,
    boolean,
    mapping,
    number,
    require,
    text,
)

type OpName = Literal[
    "keep",
    "replace",
    "delete",
    "add",
    "drop_image",
    "needs_human",
    "set_flag",
    "overlap_reviewed",
]
type SnapSource = Literal["sam", "sam_point"]
type DeleteReason = Literal["dup", "not_potato", "background"]
type DropReason = Literal["composite_unclear", "boundary_unresolvable", "junk"]
type FlagName = Literal["background_potatoes", "watermark_stock"]

TOKEN: Final = re.compile(r"[A-Za-z0-9_-]+")
BOX_ID: Final = re.compile(r"[0-9]+-[0-9a-f]{64}")
COMMON: Final = frozenset({"op", "image", "batch"})
FIELDS: Final[dict[OpName, frozenset[str]]] = {
    "keep": frozenset({"box_id", "refined_xyxy", "snap"}),
    "replace": frozenset({"box_id", "xyxy", "snap"}),
    "delete": frozenset({"box_id", "reason"}),
    "add": frozenset({"candidate_id", "xyxy", "snap"}),
    "drop_image": frozenset({"reason"}),
    "needs_human": frozenset({"reason"}),
    "set_flag": frozenset({"flag"}),
    "overlap_reviewed": frozenset({"box_a", "box_b"}),
}
SNAP_FIELDS: Final = frozenset({"src", "xyxy", "tight", "reason", "mask_area_ratio"})
SOURCES: Final[tuple[SnapSource, ...]] = ("sam", "sam_point")
DELETE_REASONS: Final[tuple[DeleteReason, ...]] = ("dup", "not_potato", "background")
DROP_REASONS: Final[tuple[DropReason, ...]] = (
    "composite_unclear",
    "boundary_unresolvable",
    "junk",
)
FLAGS: Final[tuple[FlagName, ...]] = ("background_potatoes", "watermark_stock")


@dataclass(frozen=True, slots=True)
class FixtureGate:
    root: Path


@dataclass(frozen=True, slots=True)
class ApplyInput:
    source: Path
    image_list: Path
    decision_files: tuple[Path, ...]
    fixture_gate: FixtureGate | None = None
    remaining_list: Path | None = None


@dataclass(frozen=True, slots=True)
class ImageResult:
    image: str
    label: LabelDocument
    status: str
    reason: str
    batches: str
    changes: tuple[JsonMap, ...]


@dataclass(frozen=True, slots=True)
class SnapReceipt:
    src: SnapSource
    xyxy: Xyxy
    tight: bool
    reason: str
    mask_area_ratio: float


@dataclass(frozen=True, slots=True)
class Origin:
    image: str
    batch: str


@dataclass(frozen=True, slots=True)
class Keep:
    origin: Origin
    box_id: str
    xyxy: Xyxy
    snap: SnapReceipt


@dataclass(frozen=True, slots=True)
class Replace:
    origin: Origin
    box_id: str
    xyxy: Xyxy
    snap: SnapReceipt


@dataclass(frozen=True, slots=True)
class Delete:
    origin: Origin
    box_id: str
    reason: DeleteReason


@dataclass(frozen=True, slots=True)
class Add:
    origin: Origin
    candidate_id: str
    xyxy: Xyxy
    snap: SnapReceipt


@dataclass(frozen=True, slots=True)
class DropImage:
    origin: Origin
    reason: DropReason


@dataclass(frozen=True, slots=True)
class NeedsHuman:
    origin: Origin
    reason: str


@dataclass(frozen=True, slots=True)
class SetFlag:
    origin: Origin
    flag: FlagName


@dataclass(frozen=True, slots=True)
class OverlapReviewed:
    origin: Origin
    box_a: str
    box_b: str


type Decision = (
    Keep | Replace | Delete | Add | DropImage | NeedsHuman | SetFlag | OverlapReviewed
)


def choose[T: str](value: Json, options: tuple[T, ...], what: str) -> T:
    name = text(value)
    for option in options:
        if name == option:
            return option
    raise DataError(f"Unknown {what}: {name}")


def token(value: Json, what: str) -> str:
    name = text(value)
    require(TOKEN.fullmatch(name) is not None, f"Invalid {what}: {name!r}")
    return name


def stable_box_id(value: Json) -> str:
    name = text(value)
    require(BOX_ID.fullmatch(name) is not None, f"Invalid box id: {name!r}")
    return name


def parse_xyxy(value: Json) -> Xyxy:
    values = array(value)
    require(len(values) == 4, "xyxy needs exactly 4 numbers")
    x1, y1, x2, y2 = (number(v) for v in values)
    require(x1 < x2 and y1 < y2, "Degenerate xyxy")
    return (x1, y1, x2, y2)


def parse_snap(value: Json) -> SnapReceipt:
    row = mapping(value)
    require(
        frozenset(row) == SNAP_FIELDS,
        f"Snap receipt fields must be {sorted(SNAP_FIELDS)}",
    )
    ratio = number(row["mask_area_ratio"])
    require(ratio >= 0, "mask_area_ratio must be nonnegative")
    tight = boolean(row["tight"])
    reason = text(row["reason"])
    require(bool(reason.strip()), "Snap reason must be nonempty")
    require(tight == (reason == "accepted"), "Inconsistent snap acceptance")
    require(not tight or ratio >= 0.3, "Accepted snap requires mask_area_ratio >= 0.3")
    return SnapReceipt(
        src=choose(row["src"], SOURCES, "snap src"),
        xyxy=parse_xyxy(row["xyxy"]),
        tight=tight,
        reason=reason,
        mask_area_ratio=ratio,
    )


def parse_decision(row: JsonMap) -> Decision:
    require("op" in row, "Missing op")
    op: OpName = choose(row["op"], tuple(FIELDS), "op")
    expected = COMMON | FIELDS[op]
    require(
        frozenset(row) == expected,
        f"Fields for {op} must be exactly {sorted(expected)}",
    )
    origin = Origin(token(row["image"], "image"), token(row["batch"], "batch"))
    if "snap" in row:
        receipt = parse_snap(row["snap"])
        coordinates = row.get("refined_xyxy", row.get("xyxy"))
        require(
            receipt.xyxy == parse_xyxy(coordinates), "Snap coordinates differ from op"
        )
    match op:
        case "keep":
            snap = parse_snap(row["snap"])
            require(snap.src == "sam", "keep requires src=sam")
            return Keep(
                origin,
                stable_box_id(row["box_id"]),
                parse_xyxy(row["refined_xyxy"]),
                snap,
            )
        case "replace":
            return Replace(
                origin,
                stable_box_id(row["box_id"]),
                parse_xyxy(row["xyxy"]),
                parse_snap(row["snap"]),
            )
        case "delete":
            return Delete(
                origin,
                stable_box_id(row["box_id"]),
                choose(row["reason"], DELETE_REASONS, "delete reason"),
            )
        case "add":
            return Add(
                origin,
                token(row["candidate_id"], "candidate id"),
                parse_xyxy(row["xyxy"]),
                parse_snap(row["snap"]),
            )
        case "drop_image":
            return DropImage(origin, choose(row["reason"], DROP_REASONS, "drop reason"))
        case "needs_human":
            reason = text(row["reason"])
            require(bool(reason.strip()), "needs_human reason must be nonempty")
            return NeedsHuman(origin, reason)
        case "set_flag":
            return SetFlag(origin, choose(row["flag"], FLAGS, "flag"))
        case "overlap_reviewed":
            box_a = stable_box_id(row["box_a"])
            box_b = stable_box_id(row["box_b"])
            require(box_a != box_b, "overlap_reviewed needs two different boxes")
            return OverlapReviewed(origin, box_a, box_b)
        case unreachable:
            assert_never(unreachable)


def unique_pairs(pairs: list[tuple[str, Json]]) -> JsonMap:
    keys = [key for key, _ in pairs]
    require(len(set(keys)) == len(keys), "Duplicate JSON key")
    return dict(pairs)


def parse_line(line: str) -> Decision:
    try:
        row: Json = json.loads(line, object_pairs_hook=unique_pairs)
    except json.JSONDecodeError as error:
        raise DataError(f"Invalid JSON: {error.msg}") from error
    return parse_decision(mapping(row))


def read_decisions(path: Path) -> tuple[Decision, ...]:
    """Parse a UTF-8 JSONL file; blank lines are refused except one trailing newline."""
    lines = path.read_bytes().decode("utf-8").split("\n")
    if lines[-1] == "":
        _ = lines.pop()
    decisions: list[Decision] = []
    for number_, raw in enumerate(lines, start=1):
        line = raw.removesuffix("\r")
        try:
            require(line.strip() != "", "Blank line")
            decisions.append(parse_line(line))
        except DataError as error:
            raise DataError(f"{path.name}:{number_}: {error.detail}") from error
    return tuple(decisions)
