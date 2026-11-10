"""Isolated full2060 authorization proofs using real scope/apply and synthetic IO."""

import csv
import json
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final
from unittest.mock import patch

from PIL import Image

from public_label_rework import apply_scope as scope
from public_label_rework.apply import apply
from public_label_rework.apply import main as apply_main
from public_label_rework.common import OUTPUTS, sha256, write_json
from public_label_rework.decisions import ApplyInput, FixtureGate
from public_label_rework.labelio import box_id, dumps, parse, to_xyxy
from public_label_rework.smoke_decisions import geometry, op, rejected
from public_label_rework.smoke_labelio import RECT, document_bytes, shape_row
from retraining_data.schema import Json, require

IDS: Final = tuple(f"fixture_{i:04d}" for i in range(2060))
CONFIRMED: Final = {
    "date": "2026-10-29",
    "basis": "isolated fixture only",
    "statement_verbatim": "Synthetic test authorization; not real user confirmation",
}


def frozen(path: Path, images: tuple[str, ...]) -> None:
    write_json(path, images)
    _ = path.with_suffix(".sha256").write_text(sha256(path), encoding="ascii")


def fixture(root: Path) -> ApplyInput:
    source, lists = root / "source", root / "lists"
    source.mkdir()
    lists.mkdir()
    pilot, remaining = lists / "pilot_list.json", lists / "remaining_list.json"
    frozen(pilot, IDS[:100])
    frozen(remaining, IDS[100:])
    doc = parse(document_bytes([shape_row(RECT)]))
    with (
        Image.new("RGB", (doc.imageWidth, doc.imageHeight)) as image,
        BytesIO() as stream,
    ):
        image.save(stream, format="JPEG")
        jpg = stream.getvalue()
    for image in IDS:
        _ = (source / f"{image}.json").write_bytes(
            dumps(replace(doc, imagePath=f"{image}.jpg"))
        )
        _ = (source / f"{image}.jpg").write_bytes(jpg)
    first, second = root / "pilot_ops.jsonl", root / "remaining_ops.jsonl"
    rows = (
        geometry(IDS[0], box_id(0, doc.shapes[0]), (10.0, 20.0, 90.0, 70.0)),
        geometry(IDS[100], box_id(0, doc.shapes[0]), (10.0, 20.0, 90.0, 70.0)),
        op("needs_human", IDS[1], reason="fixture_unresolved"),
    )
    _ = first.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _ = second.write_text(
        "".join(
            json.dumps(op("drop_image", i, reason="junk")) + "\n"
            for i in IDS[3:]
            if i != IDS[100]
        ),
        encoding="utf-8",
    )
    (root / "pilot").mkdir()
    return ApplyInput(source, pilot, (first, second), remaining_list=remaining)


def denied(inputs: ApplyInput, fragment: str) -> None:
    """Exercise scope through apply; no rejected call may create an output."""
    output = inputs.source.parent / "rejected"
    rejected(lambda: apply(inputs, output), fragment)
    require(not output.exists(), "Rejected scope created output")


def confirmation_proof(inputs: ApplyInput) -> None:
    # Given: correct frozen lists, a missing project confirmation, unrelated approval.
    reference = scope.CONFIRMATION.parent.parent / "label_review_v2"
    reference.mkdir()
    write_json(reference / "user_confirmation.json", CONFIRMED)
    # When / Then: neither unrelated approval nor malformed project approval works.
    denied(inputs, "Missing")
    invalid: tuple[tuple[Json, str], ...] = (
        ({}, "requires"),
        ([], "JSON object"),
        ({**CONFIRMED, "date": "2026-02-30"}, "ISO calendar"),
        ({**CONFIRMED, "date": "20261029"}, "YYYY-MM-DD"),
        ({**CONFIRMED, "basis": None}, "JSON string"),
        ({**CONFIRMED, "basis": " "}, "nonempty"),
        ({**CONFIRMED, "statement_verbatim": ""}, "nonempty"),
        ({**CONFIRMED, "statement_verbatim": False}, "JSON string"),
    )
    for value, error in invalid:
        write_json(scope.CONFIRMATION, value)
        denied(inputs, error)
    _ = scope.CONFIRMATION.write_text("{", encoding="utf-8")
    denied(inputs, "Malformed")
    scope.CONFIRMATION.unlink()
    scope.CONFIRMATION.symlink_to(reference / "user_confirmation.json")
    denied(inputs, "symlinked")
    scope.CONFIRMATION.unlink()
    write_json(scope.CONFIRMATION, CONFIRMED)
    print("PASS missing/malformed/wrong-project/symlink confirmation rejection")


def list_proof(inputs: ApplyInput) -> None:
    # Given: actual frozen list files and a valid fixture-only project confirmation.
    # When / Then: path, byte hash, duplicate, overlap, count and membership attacks fail.
    for path, baseline in ((scope.PILOT, IDS[:100]), (scope.REMAINING, IDS[100:])):
        _ = path.write_bytes(path.read_bytes() + b" ")
        denied(inputs, "SHA-256")
        frozen(path, baseline)
        for bad, error in (
            (baseline[:-1], "exactly"),
            (baseline[:-1] + baseline[:1], "duplicate"),
            (baseline[:-1] + ("../escape",), "Invalid image"),
        ):
            frozen(path, bad)
            denied(inputs, error)
        frozen(path, baseline)
    frozen(scope.REMAINING, (*IDS[100:-1], IDS[0]))
    denied(inputs, "overlap")
    frozen(scope.REMAINING, (*IDS[100:-1], "foreign"))
    denied(
        replace(inputs, decision_files=(inputs.source.parent / "missing_ops",)),
        "source membership",
    )
    frozen(scope.REMAINING, IDS[100:])
    arbitrary = scope.REMAINING.parent / "arbitrary.json"
    frozen(arbitrary, IDS[100:])
    denied(replace(inputs, remaining_list=arbitrary), "canonical remaining")
    frozen(arbitrary, IDS[:100])
    denied(replace(inputs, image_list=arbitrary), "frozen pilot")
    for path, error in (
        (scope.PILOT, "frozen pilot"),
        (scope.REMAINING, "canonical remaining"),
    ):
        raw = path.read_bytes()
        path.unlink()
        path.symlink_to(arbitrary)
        denied(inputs, error)
        path.unlink()
        _ = path.write_bytes(raw)
    sidecar = scope.REMAINING.with_suffix(".sha256")
    digest = sidecar.read_bytes()
    sidecar.unlink()
    sidecar.symlink_to(scope.PILOT.with_suffix(".sha256"))
    denied(inputs, "symlinked")
    sidecar.unlink()
    _ = sidecar.write_bytes(digest)
    image = inputs.source / f"{IDS[-1]}.jpg"
    jpg = image.read_bytes()
    image.unlink()
    denied(inputs, "source membership")
    _ = image.write_bytes(jpg)
    fixture_list = inputs.source.parent / "list.json"
    write_json(fixture_list, IDS[:100])
    escaped = replace(
        inputs, image_list=fixture_list, fixture_gate=FixtureGate(inputs.source.parent)
    )
    denied(escaped, "Fixture gate cannot authorize full")
    denied(replace(escaped, source=OUTPUTS / "source/images/public"), "Fixture gate")
    print(
        "PASS both hashes, lists, canonical paths, source membership and fixture escape"
    )


def full_proof(inputs: ApplyInput, root: Path) -> None:
    # Given: complete synthetic source, confirmed pilot100+remaining1960 and decisions.
    before = {p.name: sha256(p) for p in inputs.source.iterdir()}
    # When: API and actual CLI entry apply with opposite decision-file order.
    results = apply(inputs, root / "api")
    argv = [
        "apply",
        "--list",
        str(scope.PILOT),
        "--output",
        str(root / "cli"),
        "--remaining-list",
        str(scope.REMAINING),
        *(str(p) for p in reversed(inputs.decision_files)),
    ]
    with patch.object(sys, "argv", argv):
        apply_main()
    # Then: exactly 2060 unique serialized statuses, mixed routing and identical trees.
    with (root / "api/status.csv").open(encoding="utf-8-sig", newline="") as stream:
        statuses = list(csv.DictReader(stream))
    require(
        len(statuses) == len({r["image"] for r in statuses}) == 2060, "Status count"
    )
    require({r["image"] for r in statuses} == set(IDS), "Status membership")
    expected = {i: "excluded" for i in IDS[3:]}
    expected.update(
        {IDS[0]: "kept", IDS[100]: "kept", IDS[1]: "needs_human", IDS[2]: "needs_human"}
    )
    require({r["image"]: r["status"] for r in statuses} == expected, "Status routing")
    trees = [
        {
            str(p.relative_to(root / n)): sha256(p)
            for p in (root / n).rglob("*")
            if p.is_file()
        }
        for n in ("api", "cli")
    ]
    require(trees[0] == trees[1], "API/CLI output bytes differ")
    require(
        {p.name: sha256(p) for p in inputs.source.iterdir()} == before, "Source changed"
    )
    require(len(results) == 2060, "Result count")
    for image in (IDS[0], IDS[100]):
        doc = parse((root / "api/labels/images/public" / f"{image}.json").read_bytes())
        require(
            not doc.checked
            and doc.flags["human_reviewed"] is False
            and doc.flags["ai_reviewed"] is True,
            "Review flags",
        )
        require(to_xyxy(doc.shapes[0]) == (10, 20, 90, 70), "Snapped update lost")
    print(
        "PASS T15-APPLY-2060: API/CLI 2060 statuses, identical files:",
        len(trees[0]),
        "; source unchanged",
    )


def pilot_proof(inputs: ApplyInput) -> None:
    # Given: optional remaining input is absent, regardless of confirmation presence.
    pilot = replace(inputs, remaining_list=None)
    # When: the default selection runs; full decisions are mistakenly supplied.
    selected = scope.selected_images(pilot)
    denied(pilot, "outside explicit scope")
    # Then: only frozen pilot100 is selected and nonpilot decisions stay rejected.
    require(selected == IDS[:100], "Default pilot widened")
    print("PASS default pre-gate pilot100 unchanged; nonpilot decisions refused")


def main() -> None:
    """Only path constants and argv are redirected; authorization/apply IO is real."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="smoke-decisions-scope-", dir=OUTPUTS) as directory:
        root = Path(directory)
        inputs = fixture(root)
        with patch.multiple(
            scope,
            SOURCE=inputs.source,
            PILOT=inputs.image_list,
            REMAINING=inputs.remaining_list,
            CONFIRMATION=root / "pilot/user_confirmation.json",
        ):
            pilot_proof(inputs)
            confirmation_proof(inputs)
            list_proof(inputs)
            full_proof(inputs, root)
            pilot_proof(inputs)
    print("PASS smoke_apply_scope; only owned TemporaryDirectory cleaned")


if __name__ == "__main__":
    main()
