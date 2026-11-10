"""Fixture proofs and the 2,060-file byte round trip for the X-AnyLabeling label IO."""

import json
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from public_label_rework.labelio import (
    LabelDocument,
    box_id,
    dumps,
    from_xyxy,
    parse,
    to_xyxy,
)
from retraining_data.schema import DataError, require

ZIP = Path(__file__).resolve().parent.parent / "final-dataset.zip"
type Row = dict[str, object]
PUBLIC_PREFIX = "images/public/"
SAMPLE = "public_original06_0001.json"


def rejected[T](action: Callable[[], T]) -> None:
    try:
        _ = action()
    except DataError:
        return
    raise DataError("Expected DataError")


def shape_row(points: list[list[float]], **overrides: object) -> Row:
    row: Row = {
        "label": "potato",
        "points": points,
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
    }
    return {**row, **overrides}


def document_bytes(shapes: list[Row], **extra: object) -> bytes:
    row: Row = {
        "version": "4.1.1",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": "a.jpg",
        "imageData": None,
        "imageHeight": 640,
        "imageWidth": 640,
        "description": "한글 설명",
        **extra,
    }
    return (
        json.dumps(row, indent=2, ensure_ascii=False).replace("\n", "\r\n") + "\r\n"
    ).encode("utf-8")


RECT = [[10.5, 20.0], [100, 20.0], [100, 80.25], [10.5, 80.25]]


def unmodified_round_trip_proof() -> None:
    # Given: a CRLF document mixing int and float points and non-ASCII text.
    source = document_bytes([shape_row(RECT)])
    # When: it is parsed and written without modification.
    written = dumps(parse(source))
    # Then: the bytes are identical.
    require(written == source, "Unmodified document must round-trip byte-identically")


def unknown_keys_proof() -> None:
    # Given: unknown top-level and shape keys, a score and a custom key order.
    shape: Row = {
        **shape_row(RECT),
        "score": 0.5,
        "kie_linking": [],
        "custom": {"a": 1},
    }
    null_score: Row = {**shape_row(RECT, description=None), "score": None}
    source = document_bytes([shape, null_score], review_metadata={"z": 1, "a": [1, 2]})
    # When: parsed and written.
    doc = parse(source)
    # Then: unknown keys survive in place and order is unchanged.
    require(dumps(doc) == source, "Unknown keys and order preserved")
    require(
        [k for k, _ in doc.extra] == ["review_metadata"], "Top-level unknown key kept"
    )
    require(
        [k for k, _ in doc.shapes[0].extra] == ["kie_linking", "custom"],
        "Shape unknown keys kept",
    )
    require(
        doc.shapes[0].score == 0.5 and doc.shapes[0].has_score,
        "Score parsed when present",
    )
    require(
        doc.shapes[1].has_score and doc.shapes[1].score is None,
        "Null score kept as present",
    )


def score_absent_proof() -> None:
    # Given: a shape without score.
    doc = parse(document_bytes([shape_row(RECT)]))
    # When: its coordinates are replaced and the document is written.
    moved = replace(doc, shapes=(from_xyxy(doc.shapes[0], (1.0, 2.0, 30.0, 40.0)),))
    # Then: no score key is introduced.
    require(
        not doc.shapes[0].has_score and b'"score"' not in dumps(moved),
        "score must not be added",
    )


def invalid_shapes_proof() -> None:
    # Given: a 3-point shape, a polygon, and a missing key.
    three = document_bytes([shape_row(RECT[:3])])
    polygon = document_bytes([shape_row(RECT, shape_type="polygon")])
    missing = document_bytes(
        [{k: v for k, v in shape_row(RECT).items() if k != "flags"}]
    )
    # When / Then: parsing each raises DataError.
    rejected(lambda: parse(three))
    rejected(lambda: parse(polygon))
    rejected(lambda: parse(missing))


def xyxy_and_id_proof() -> None:
    # Given: a rectangle shape.
    shape = parse(document_bytes([shape_row(RECT)])).shapes[0]
    # When: converted to xyxy and back.
    rebuilt = from_xyxy(shape, to_xyxy(shape))
    # Then: corners run clockwise from top-left with floats equal to the input.
    require(to_xyxy(shape) == (10.5, 20.0, 100.0, 80.25), "xyxy values")
    require(
        rebuilt.points == ((10.5, 20.0), (100.0, 20.0), (100.0, 80.25), (10.5, 80.25)),
        "Corner order",
    )
    # Then: ids ignore sub-0.01 jitter and depend on index and coordinates.
    jitter = from_xyxy(shape, (10.5004, 20.0, 100.0, 80.25))
    shifted = from_xyxy(shape, (11.5, 20.0, 100.0, 80.25))
    require(box_id(3, shape) == box_id(3, jitter), "Id stable under rounding")
    require(box_id(3, shape) != box_id(4, shape), "Id depends on index")
    require(box_id(3, shape) != box_id(3, shifted), "Id depends on points")


def round_trip_public() -> tuple[int, list[str], LabelDocument]:
    with zipfile.ZipFile(ZIP) as archive:
        names = sorted(
            n
            for n in archive.namelist()
            if n.startswith(PUBLIC_PREFIX) and n.endswith(".json")
        )
        identical = 0
        mismatched: list[str] = []
        boxes = scores = 0
        for name in names:
            raw = archive.read(name)
            doc = parse(raw)
            if dumps(doc) == raw:
                identical += 1
            else:
                mismatched.append(name)
            for shape in doc.shapes:
                (x1, y1), (x2, _), (_, y3), (x4, _) = shape.points
                require(
                    x1 == x4 and x1 < x2 and y1 < y3, f"Point order differs in {name}"
                )
                boxes += 1
                scores += shape.has_score
        require(
            len(names) == 2060 and boxes == 9498,
            f"Unexpected public set: {len(names)} files, {boxes} boxes",
        )
        sample = parse(archive.read(PUBLIC_PREFIX + SAMPLE))
    print(f"files={len(names)} boxes={boxes} shapes_with_score={scores}")
    return identical, mismatched, sample


def main() -> None:
    for proof in (
        unmodified_round_trip_proof,
        unknown_keys_proof,
        score_absent_proof,
        invalid_shapes_proof,
        xyxy_and_id_proof,
    ):
        proof()
        print(f"PASS {proof.__name__}")
    identical, mismatched, sample = round_trip_public()
    print(f"round trip {identical}/2060 byte-identical")
    require(
        not mismatched and identical == 2060, f"Mismatching files: {mismatched[:10]}"
    )
    for index, shape in enumerate(sample.shapes):
        print(SAMPLE, box_id(index, shape), to_xyxy(shape))


if __name__ == "__main__":
    main()
