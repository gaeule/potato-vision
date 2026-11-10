"""Deterministic decision application; no inference, downloads or original writes."""

import csv
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import assert_never

from public_label_rework import decisions as d
from public_label_rework import labelio as io
from public_label_rework.apply_scope import parse_args, selected_images
from public_label_rework.common import OUTPUTS, new_output
from retraining_data.schema import JsonMap, require, text


def refined(
    shape: io.Shape, op: d.Keep | d.Replace | d.Add, doc: io.LabelDocument
) -> io.Shape:
    """Apply a parsed tool result without dropping optional source fields."""
    x1, y1, x2, y2 = op.xyxy
    require(
        0 <= x1 < x2 <= doc.imageWidth and 0 <= y1 < y2 <= doc.imageHeight,
        "Decision coordinates outside source image",
    )
    provenance = f"src={op.snap.src};batch={op.origin.batch}"
    return replace(
        io.from_xyxy(shape, op.xyxy),
        flags={**shape.flags, "tight": op.snap.tight},
        description=";".join(filter(None, (shape.description, provenance))),
    )


def apply_image(
    image: str, doc: io.LabelDocument, ops: tuple[d.Decision, ...]
) -> d.ImageResult:
    """Resolve each original stable ID once, retaining source order and metadata."""
    originals = {io.box_id(i, shape): shape for i, shape in enumerate(doc.shapes)}
    working: dict[str, io.Shape | None] = dict(originals)
    flags = dict(doc.flags)
    terminal: d.DropImage | d.NeedsHuman | None = None
    seen: set[str] = set()
    changes: list[JsonMap] = []
    for op in ops:
        key = ""
        before: io.Shape | None = None
        after: io.Shape | None = None
        match op:
            case d.Keep() | d.Replace() | d.Delete():
                key = op.box_id
                require(key in originals, f"Unknown box id: {image}/{key}")
                before = originals[key]
                match op:
                    case d.Keep() | d.Replace():
                        after = refined(before, op, doc)
                    case d.Delete():
                        pass
                    case unreachable:
                        assert_never(unreachable)
                working[key] = after
            case d.Add():
                key = "add:" + op.candidate_id
                after = io.Shape(
                    "potato",
                    (),
                    None,
                    "",
                    False,
                    "rectangle",
                    {},
                    {},
                    False,
                    None,
                    (),
                    io.SHAPE_KEYS,
                )
                after = refined(after, op, doc)
                working[key] = after
            case d.DropImage() | d.NeedsHuman():
                terminal = op
            case d.SetFlag():
                key = "flag:" + op.flag
                flags[op.flag] = True
            case d.OverlapReviewed():
                require(
                    op.box_a in originals and op.box_b in originals,
                    "Unknown overlap ID",
                )
                key = "overlap:" + ":".join(sorted((op.box_a, op.box_b)))
            case unreachable:
                assert_never(unreachable)
        require(key not in seen, f"Conflicting ops: {image}/{key}")
        seen.add(key)
        changes.append(
            {
                "image": image,
                "batch": op.origin.batch,
                "op": type(op).__name__,
                "id": key,
                "before": None if before is None else io.shape_json(before),
                "after": None if after is None else io.shape_json(after),
                "decision": json.loads(json.dumps(asdict(op))),
            }
        )
    shapes = tuple(s for s in working.values() if s is not None)
    unresolved = any(key not in seen for key in originals) or any(
        shape.flags.get("tight") is not True for shape in shapes
    )
    status, reason = "kept", ""
    match terminal:
        case d.DropImage(reason=reason):
            require(len(ops) == 1, "drop_image conflicts with other image ops")
            status = "excluded"
        case d.NeedsHuman(reason=reason):
            status = "needs_human"
        case None:
            if unresolved or not ops:
                status, reason = "needs_human", "unresolved_or_untight"
        case unreachable:
            assert_never(unreachable)
    flags.update(
        {
            "human_reviewed": False,
            "ai_reviewed": bool(ops),
            "background_potatoes": flags.get("background_potatoes", False),
            "rework_status": status,
        }
    )
    label = replace(doc, checked=False, flags=flags, shapes=shapes)
    batches = ";".join(sorted({op.origin.batch for op in ops}))
    return d.ImageResult(image, label, status, reason, batches, tuple(changes))


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream, lineterminator="\r\n").writerows(rows)


def apply(inputs: d.ApplyInput, output: Path) -> tuple[d.ImageResult, ...]:
    """Validate the entire run before creating a new output tree."""
    images = selected_images(inputs)
    grouped: dict[str, list[d.Decision]] = {image: [] for image in images}
    for path in sorted(inputs.decision_files):
        for op in d.read_decisions(path):
            require(
                op.origin.image in grouped,
                f"Image outside explicit scope: {op.origin.image}",
            )
            grouped[op.origin.image].append(op)
    results: list[d.ImageResult] = []
    for image in images:
        label, jpg = inputs.source / f"{image}.json", inputs.source / f"{image}.jpg"
        doc = io.parse(label.read_bytes())
        require(doc.imagePath == jpg.name, "Source imagePath mismatch")
        require(
            all(shape.label == "potato" for shape in doc.shapes),
            "Non-potato source label",
        )
        ops = tuple(sorted(grouped[image], key=repr))
        results.append(apply_image(image, doc, ops))
    require(
        not output.resolve().is_relative_to(inputs.source.resolve())
        and not output.resolve().is_relative_to(OUTPUTS / "source"),
        "Output cannot be inside source",
    )
    root = new_output(output)
    statuses = [["image", "status", "reason", "batch"]]
    exclusions = [["image", "reason", "batch"]]
    csv_changes = [["image", "box_id", "batch", "change"]]
    change_rows: list[str] = []
    for result in results:
        statuses.append([result.image, result.status, result.reason, result.batches])
        if result.status == "excluded":
            exclusions.append([result.image, result.reason, result.batches])
        else:
            folder = root / (
                "labels/images/public" if result.status == "kept" else "needs_human"
            )
            folder.mkdir(parents=True, exist_ok=True)
            _ = (folder / f"{result.image}.json").write_bytes(io.dumps(result.label))
            if result.status == "needs_human":
                _ = shutil.copyfile(
                    inputs.source / f"{result.image}.jpg",
                    folder / f"{result.image}.jpg",
                )
        for change in result.changes:
            change_rows.append(
                json.dumps(change, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            csv_changes.append(
                [
                    result.image,
                    text(change["id"]),
                    text(change["batch"]),
                    "결정 적용: " + text(change["op"]),
                ]
            )
    _ = (root / "changes.jsonl").write_text(
        "".join(row + "\n" for row in change_rows), encoding="utf-8"
    )
    write_csv(root / "changes.csv", csv_changes)
    write_csv(root / "status.csv", statuses)
    write_csv(root / "exclusions.csv", exclusions)
    return tuple(results)


def main() -> None:
    """Apply the explicit pilot list, or confirmed union with --remaining-list."""
    inputs, output = parse_args(sys.argv[1:])
    _ = apply(inputs, output)


if __name__ == "__main__":
    main()
