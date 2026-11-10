"""X-AnyLabeling JSON IO: strict parse, key-order preserving, byte-faithful writer.

Source files use CRLF line endings, two-space indent, ``ensure_ascii=False`` and a
trailing CRLF. Integer versus float point values are preserved so an unmodified
document serializes back to the exact source bytes.
"""

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Final

from retraining_data.schema import (
    DataError,
    Json,
    JsonMap,
    array,
    boolean,
    document,
    integer,
    mapping,
    require,
    text,
)

type Num = int | float
type Point = tuple[Num, Num]
type Xyxy = tuple[float, float, float, float]

NEWLINE: Final = "\r\n"
SHAPE_KEYS: Final = (
    "label",
    "points",
    "group_id",
    "description",
    "difficult",
    "shape_type",
    "flags",
    "attributes",
)
DOCUMENT_KEYS: Final = (
    "version",
    "flags",
    "checked",
    "shapes",
    "imagePath",
    "imageData",
    "imageHeight",
    "imageWidth",
    "description",
)


@dataclass(frozen=True, slots=True)
class Shape:
    """A rectangle shape; ``has_score`` records whether the source carried a score key (it may be null)."""

    label: str
    points: tuple[Point, ...]
    group_id: Json
    description: str | None
    difficult: bool
    shape_type: str
    flags: JsonMap
    attributes: JsonMap
    has_score: bool
    score: Json
    extra: tuple[tuple[str, Json], ...]
    order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelDocument:
    version: str
    flags: JsonMap
    checked: bool
    shapes: tuple[Shape, ...]
    imagePath: str
    imageData: Json
    imageHeight: int
    imageWidth: int
    description: str
    extra: tuple[tuple[str, Json], ...]
    order: tuple[str, ...]


def number(value: Json) -> Num:
    """Finite JSON number, keeping int versus float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError("Expected JSON number")
    require(math.isfinite(value), "Expected finite JSON number")
    return value


def parse_point(value: Json) -> Point:
    pair = array(value)
    require(len(pair) == 2, "Expected two coordinates per corner")
    return (number(pair[0]), number(pair[1]))


def parse_shape(value: Json) -> Shape:
    row = mapping(value)
    for key in SHAPE_KEYS:
        require(key in row, f"Missing shape key: {key}")
    shape_type = text(row["shape_type"])
    require(shape_type == "rectangle", f"Unsupported shape_type: {shape_type}")
    points = tuple(parse_point(p) for p in array(row["points"]))
    require(len(points) == 4, "Rectangle needs exactly 4 points")
    known = (*SHAPE_KEYS, "score")
    score = row.get("score")
    if score is not None:
        _ = number(score)
    return Shape(
        label=text(row["label"]),
        points=points,
        group_id=row["group_id"],
        description=None if row["description"] is None else text(row["description"]),
        difficult=boolean(row["difficult"]),
        shape_type=shape_type,
        flags=mapping(row["flags"]),
        attributes=mapping(row["attributes"]),
        has_score="score" in row,
        score=score,
        extra=tuple((k, v) for k, v in row.items() if k not in known),
        order=tuple(row),
    )


def parse(data: bytes) -> LabelDocument:
    row = document(data)
    for key in DOCUMENT_KEYS:
        require(key in row, f"Missing document key: {key}")
    return LabelDocument(
        version=text(row["version"]),
        flags=mapping(row["flags"]),
        checked=boolean(row["checked"]),
        shapes=tuple(parse_shape(s) for s in array(row["shapes"])),
        imagePath=text(row["imagePath"]),
        imageData=row["imageData"],
        imageHeight=integer(row["imageHeight"]),
        imageWidth=integer(row["imageWidth"]),
        description=text(row["description"]),
        extra=tuple((k, v) for k, v in row.items() if k not in DOCUMENT_KEYS),
        order=tuple(row),
    )


def shape_json(shape: Shape) -> JsonMap:
    values: JsonMap = {
        "label": shape.label,
        "points": [[x, y] for x, y in shape.points],
        "group_id": shape.group_id,
        "description": shape.description,
        "difficult": shape.difficult,
        "shape_type": shape.shape_type,
        "flags": shape.flags,
        "attributes": shape.attributes,
        **dict(shape.extra),
    }
    if shape.has_score:
        values["score"] = shape.score
    return {key: values[key] for key in shape.order if key in values}


def document_json(doc: LabelDocument) -> JsonMap:
    values: JsonMap = {
        "version": doc.version,
        "flags": doc.flags,
        "checked": doc.checked,
        "shapes": [shape_json(s) for s in doc.shapes],
        "imagePath": doc.imagePath,
        "imageData": doc.imageData,
        "imageHeight": doc.imageHeight,
        "imageWidth": doc.imageWidth,
        "description": doc.description,
        **dict(doc.extra),
    }
    return {key: values[key] for key in doc.order}


def dumps(doc: LabelDocument) -> bytes:
    """Serialize in the source convention (indent 2, CRLF, trailing CRLF, raw UTF-8)."""
    body = json.dumps(document_json(doc), indent=2, ensure_ascii=False, allow_nan=False)
    return (body.replace("\n", NEWLINE) + NEWLINE).encode("utf-8")


def to_xyxy(shape: Shape) -> Xyxy:
    xs = [float(x) for x, _ in shape.points]
    ys = [float(y) for _, y in shape.points]
    return (min(xs), min(ys), max(xs), max(ys))


def from_xyxy(shape: Shape, xyxy: Xyxy) -> Shape:
    """Replace the points with the corners clockwise from top-left (the source order)."""
    x1, y1, x2, y2 = xyxy
    require(x1 < x2 and y1 < y2, "Degenerate box")
    return replace(shape, points=((x1, y1), (x2, y1), (x2, y2), (x1, y2)))


def box_id(index: int, shape: Shape) -> str:
    """Stable id: original index plus SHA-256 of the points rounded to 0.01."""
    canonical = ";".join(
        f"{round(x, 2) + 0.0:.2f},{round(y, 2) + 0.0:.2f}" for x, y in shape.points
    )
    return f"{index}-{hashlib.sha256(canonical.encode('ascii')).hexdigest()}"
