"""CPU apply proofs: five copied source samples, fabricated receipts, no inference."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

from public_label_rework.apply import (
    apply,
    apply_image,
)
from public_label_rework.apply_scope import SOURCE, selected_images
from public_label_rework.common import OUTPUTS, sha256
from public_label_rework.decisions import ApplyInput, FixtureGate, parse_line
from public_label_rework.labelio import box_id, document_json, dumps, parse, to_xyxy
from retraining_data.schema import DataError, Json, JsonMap, require

SAMPLES: Final = (
    "public_original06_0001",
    "public_original02_0077",
    "public_original01_0069",
    "public_original03_0001",
    "public_original07_0024",
)


def rejected[T](action: Callable[[], T], fragment: str) -> None:
    try:
        _ = action()
    except DataError as error:
        require(fragment in error.detail, f"Wrong rejection: {error.detail}")
        return
    raise DataError(f"Expected rejection: {fragment}")


def op(name: str, image: str, **fields: Json) -> JsonMap:
    """Reusable JSONL fixture envelope; fields remain intentionally untrusted JSON."""
    return {"op": name, "image": image, "batch": "fixture", **fields}


def geometry(image: str, shape_id: str, xyxy: tuple[float, ...]) -> JsonMap:
    coords: list[Json] = list(xyxy)
    snap: JsonMap = {
        "src": "sam",
        "xyxy": coords,
        "tight": True,
        "reason": "accepted",
        "mask_area_ratio": 0.8,
    }
    return op("keep", image, box_id=shape_id, refined_xyxy=coords, snap=snap)


def fixture(root: Path) -> tuple[ApplyInput, list[JsonMap]]:
    source = root / "source"
    source.mkdir()
    for image in SAMPLES:
        for ext in (".json", ".jpg"):
            _ = (source / (image + ext)).write_bytes(
                (SOURCE / (image + ext)).read_bytes()
            )
    rows: list[JsonMap] = []
    for image in SAMPLES[:2]:
        doc = parse((source / f"{image}.json").read_bytes())
        rows.extend(
            geometry(image, box_id(i, s), to_xyxy(s)) for i, s in enumerate(doc.shapes)
        )
    rows.append(op("drop_image", SAMPLES[2], reason="boundary_unresolvable"))
    doc = parse((source / f"{SAMPLES[3]}.json").read_bytes())
    row = geometry(SAMPLES[3], box_id(0, doc.shapes[0]), to_xyxy(doc.shapes[0]))
    row["snap"] = {
        "src": "sam",
        "xyxy": row["refined_xyxy"],
        "tight": False,
        "reason": "iou_below_0.5",
        "mask_area_ratio": 0.4,
    }
    rows.append(row)
    # Synthetic metadata variation on a COPY only.
    path = source / f"{SAMPLES[1]}.json"
    doc = parse(path.read_bytes())
    shape = replace(
        doc.shapes[0],
        has_score=True,
        score=None,
        order=(*doc.shapes[0].order, "score", "custom"),
        extra=(("custom", "kept"),),
    )
    _ = path.write_bytes(dumps(replace(doc, shapes=(shape,))))
    snap: JsonMap = {
        "src": "sam_point",
        "xyxy": [10, 10, 30, 30],
        "tight": True,
        "reason": "accepted",
        "mask_area_ratio": 0.9,
    }
    rows.append(op("add", SAMPLES[1], candidate_id="new", xyxy=snap["xyxy"], snap=snap))
    rows.append(op("set_flag", SAMPLES[1], flag="background_potatoes"))
    listing, decisions = root / "list.json", root / "ops.jsonl"
    _ = listing.write_text(json.dumps(SAMPLES), encoding="utf-8")
    _ = decisions.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return ApplyInput(source, listing, (decisions,), FixtureGate(root)), rows


def output_proof(inputs: ApplyInput, root: Path) -> None:
    # Given: identical sources and decisions, distinct fresh output directories.
    before = {p.name: sha256(p) for p in inputs.source.iterdir()}
    # When: the real apply entry point runs twice.
    results = apply(inputs, root / "one")
    _ = apply(inputs, root / "two")
    # Then: the entire trees match, and source bytes and metadata survive.
    trees = [
        {
            str(p.relative_to(root / n)): sha256(p)
            for p in (root / n).rglob("*")
            if p.is_file()
        }
        for n in ("one", "two")
    ]
    require(trees[0] == trees[1], "Output bytes differ")
    after = {p.name: sha256(p) for p in inputs.source.iterdir()}
    require(before == after, "Source changed")
    expected = ("kept", "kept", "excluded", "needs_human", "needs_human")
    observed = {r.image: r.status for r in results}
    require(observed == dict(zip(SAMPLES, expected, strict=True)), "Routing")
    for result in results:
        old = parse((inputs.source / f"{result.image}.json").read_bytes())
        new = result.label
        if result.status != "excluded":
            folder = (
                "labels/images/public" if result.status == "kept" else "needs_human"
            )
            new = parse((root / "one" / folder / f"{result.image}.json").read_bytes())
        require(new.order == old.order and new.extra == old.extra, "Keys")
        require(
            not new.checked and new.flags["human_reviewed"] is False,
            "False human claim",
        )
        if result.status == "kept":
            require(new.flags["ai_reviewed"] is True, "Kept without AI decision")
        for a, b in zip(old.shapes, new.shapes, strict=False):
            require(
                (a.order, a.has_score, a.score, a.extra)
                == (b.order, b.has_score, b.score, b.extra),
                "Shape metadata changed",
            )
        if result.status == "needs_human":
            jpg = f"{result.image}.jpg"
            require(
                (root / "one/needs_human" / jpg).read_bytes()
                == (inputs.source / jpg).read_bytes(),
                "Image bytes changed",
            )
        delta = {
            k: v
            for k, v in document_json(new).items()
            if document_json(old).get(k) != v
        }
        print(result.image, result.status, "JSON_DIFF", json.dumps(delta))
    added = next(r for r in results if r.image == SAMPLES[1]).label.shapes[-1]
    require(not added.has_score and "score" not in added.order, "Added score")
    for name in ("changes", "status", "exclusions"):
        require(
            (root / f"one/{name}.csv").read_bytes()[:3] == b"\xef\xbb\xbf", "CSV BOM"
        )
    try:
        _ = apply(inputs, root / "one")
    except FileExistsError:
        print("PASS existing output refused with FileExistsError")
    else:
        raise DataError("Existing output accepted")
    print(
        "PASS identical trees:", len(trees[0]), "files; source and existing output safe"
    )


def rejection_proof(inputs: ApplyInput, rows: list[JsonMap]) -> None:
    # Given: malformed decisions and a valid isolated source.
    original = rows[0]
    # When / Then: reject at the intended schema, conflict or scope boundary.
    missing = {k: v for k, v in original.items() if k != "snap"}
    rejected(lambda: parse_line(json.dumps(missing)), "Fields")
    mismatch = {**original, "refined_xyxy": [1, 2, 3, 4]}
    rejected(lambda: parse_line(json.dumps(mismatch)), "Snap coordinates")
    rejected(lambda: parse_line('{"op":"keep","op":"delete"}'), "Duplicate JSON key")
    rejected(lambda: selected_images(replace(inputs, fixture_gate=None)), "canonical")
    real = replace(inputs, source=SOURCE, fixture_gate=None)
    rejected(lambda: selected_images(real), "frozen pilot")
    rejected(lambda: selected_images(replace(inputs, source=SOURCE)), "Fixture gate")
    decisions = inputs.decision_files[0]
    invalid = (
        ("conflict", [original, original], "Conflicting ops"),
        ("unknown", [{**original, "box_id": "999-" + "a" * 64}], "Unknown box"),
        (
            "outside",
            [{**original, "image": "harvest_04_004"}],
            "outside explicit scope",
        ),
        (
            "status",
            [original, op("drop_image", SAMPLES[0], reason="junk")],
            "conflicts",
        ),
    )
    for name, bad, fragment in invalid:
        _ = decisions.write_text(
            "".join(json.dumps(r) + "\n" for r in bad), encoding="utf-8"
        )
        rejected(lambda n=name: apply(inputs, decisions.parent / n), fragment)
    _ = decisions.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    rejected(lambda: apply(inputs, inputs.source / "bad"), "inside source")
    jpg = inputs.source / f"{SAMPLES[0]}.jpg"
    jpg.unlink()
    jpg.symlink_to(SOURCE / jpg.name)
    rejected(lambda: apply(inputs, decisions.parent / "symlink"), "Source symlink")
    print("PASS conflicts, missing receipts, unknown IDs and source/scope boundaries")


def variants_proof(inputs: ApplyInput) -> None:
    # Given: two source boxes and parsed replace/delete/overlap decisions.
    image = SAMPLES[4]
    source = parse((inputs.source / f"{image}.json").read_bytes())
    doc = replace(source, shapes=source.shapes[:2])
    first, second = (box_id(i, s) for i, s in enumerate(doc.shapes))
    row = geometry(image, first, (20.0, 30.0, 80.0, 90.0))
    replacement = {**row, "op": "replace", "xyxy": row["refined_xyxy"]}
    del replacement["refined_xyxy"]
    rows = (
        replacement,
        op("delete", image, box_id=second, reason="dup"),
        op("overlap_reviewed", image, box_a=first, box_b=second),
    )
    ops = tuple(parse_line(json.dumps(r)) for r in rows)
    # When: these operations are applied.
    result = apply_image(image, doc, ops)
    # Then: only the replacement remains; deletion has before/after evidence.
    require(
        result.status == "kept" and len(result.label.shapes) == 1, "Variant routing"
    )
    require(to_xyxy(result.label.shapes[0]) == (20, 30, 80, 90), "Replacement lost")
    require(
        result.changes[1]["before"] is not None and result.changes[1]["after"] is None,
        "Delete log lost",
    )
    # Given: otherwise tight boxes but an explicit unresolved image.
    human = parse_line(json.dumps(op("needs_human", image, reason="경계 미해결")))
    # When: completeness cannot be resolved.
    unresolved = apply_image(image, doc, (*ops, human))
    # Then: accepted coordinates cannot override needs_human.
    require(unresolved.status == "needs_human", "Unresolved image kept")
    # Given: inherited tight flags without any box decision in this run.
    inherited = replace(
        doc, shapes=tuple(replace(s, flags={"tight": True}) for s in doc.shapes)
    )
    flag = parse_line(json.dumps(op("set_flag", image, flag="watermark_stock")))
    # When: only an unrelated flag is decided.
    undecided = apply_image(image, inherited, (flag,))
    # Then: inherited tight flags cannot hide undecided boxes.
    require(undecided.status == "needs_human", "Undecided boxes kept")
    print("PASS all eight ops; unresolved and inherited-tight undecided boxes")


def main() -> None:
    """Repeatable aggregate entry; only the owned TemporaryDirectory is cleaned."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="smoke-decisions-", dir=OUTPUTS) as directory:
        root = Path(directory)
        inputs, rows = fixture(root)
        output_proof(inputs, root)
        variants_proof(inputs)
        rejection_proof(inputs, rows)
    print("PASS smoke_decisions: synthetic receipts only, no SAM or real review run")


if __name__ == "__main__":
    main()
