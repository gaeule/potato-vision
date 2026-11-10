"""CPU-only boundary fixtures; synthetic masks are never real toolkit evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from public_label_rework import instance_roster as r
from public_label_rework import snap_instances as e
from retraining_data.schema import DataError, Json, JsonMap, mapping


def rejects[T](action: Callable[[], T], reason: str) -> None:
    """Then the exact named boundary rejects, not an unrelated exception."""
    try:
        _ = action()
    except DataError as error:
        assert reason in error.detail
    else:
        raise AssertionError(reason)


def candidate_guards(root: Path) -> None:
    # Given synthetic CPU pixels and one null plus one nonnull target.
    targets = (
        r.Target("T001", None, (), "uncertain", "null fixture"),
        r.Target("T002", (4, 4), (), "identifiable", "point fixture"),
    )
    image = r.RosterImage("fixture", "source-hash", 10, 10, False, targets, ())
    folder = root / image.image_id
    folder.mkdir()
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[2:7, 2:7] = True
    saved = e.save_mask(folder / "synthetic.npy", mask)
    null = e.Candidate(
        "fixture",
        "T001",
        "source-hash",
        None,
        None,
        ((4, 4),),
        (),
        (),
        "null_body_point",
        0,
        "uncertain",
        "null fixture",
    )
    candidate = e.Candidate(
        "fixture",
        "T002",
        "source-hash",
        (2, 2, 7, 7),
        (4, 4),
        (),
        (saved,),
        (0.8,),
        "tool_candidate",
        0,
        "identifiable",
        "point fixture",
        model_called=True,
        wire_points=((4, 4),),
        wire_labels=(1,),
        positive_in_component=True,
        wire_consistent=True,
        full_consistent=True,
    )
    records: list[Json] = [null.json(), candidate.json()]
    # When binding candidate records to frozen target membership and raw geometry.
    e.verify_candidates(records, (image,), root)
    # Then both records are present including the null target.
    assert len(records) == 2
    for changed, reason in (
        (records[:1], "missing_target_membership"),
        ([*records, records[1]], "duplicate_or_unknown_candidate"),
        (
            [records[0], replace(candidate, image_sha256="wrong").json()],
            "wrong_image_association",
        ),
        (
            [records[0], replace(candidate, box=(1, 1, 7, 7)).json()],
            "candidate_box_tamper",
        ),
    ):
        # Given altered membership, image identity or a hand-authored box.
        # When checked, then the corresponding boundary rejects it.
        rejects(
            lambda changed=changed: e.verify_candidates(changed, (image,), root), reason
        )
    # Given a changed underlying candidate mask with the original digest.
    _ = (folder / "synthetic.npy").write_bytes(b"changed")
    # When checked, then mask/input tamper is rejected.
    rejects(
        lambda: e.verify_candidates(records, (image,), root), "candidate_mask_tamper"
    )
    # Given only a null record, when default evaluation is called.
    scored = e.evaluate(np.full((10, 10, 3), 255, dtype=np.uint8), [records[0]], folder)
    # Then it is a denominator-preserving failure without importing SAM.
    assert len(scored) == 1 and mapping(scored[0])["default_result"] is None
    assert mapping(scored[0])["semantic_single_body"] is None
    assert r.counts((image,)) == {"targets": 2, "uncertain": 1, "null_points": 1}
    assert null.wire_points == () and not null.model_called
    # Given a genuinely null-only roster, when generated, then SAM is never needed.
    null_image = replace(image, targets=(targets[0],))
    null_generated = e.generate(
        np.full((10, 10, 3), 255, dtype=np.uint8), null_image, folder
    )
    assert len(null_generated) == 1 and not null_generated[0].model_called
    assert (
        null_generated[0].wire_points == ()
        and null_generated[0].full_consistent is None
    )


def wire_guards(_root: Path) -> None:
    # Given distance ties with intentionally unsorted coordinates and a large set.
    positive = (4.0, 4.0)
    for full, expected in (
        ((), ()),
        (((8.0, 4.0),), ((8.0, 4.0),)),
        (((5.0, 4.0), (3.0, 4.0)), ((3.0, 4.0), (5.0, 4.0))),
        (((4.0, 5.0), (5.0, 4.0), (3.0, 4.0)), ((3.0, 4.0), (4.0, 5.0), (5.0, 4.0))),
        (
            ((4.0, 5.0), (5.0, 4.0), (4.0, 3.0), (3.0, 4.0)),
            ((3.0, 4.0), (4.0, 3.0), (4.0, 5.0)),
        ),
        (
            tuple((float(i), 4.0) for i in range(10, 138)),
            ((10.0, 4.0), (11.0, 4.0), (12.0, 4.0)),
        ),
    ):
        # When choosing wire points, then k and distance/x/y ordering are exact.
        assert e.wire_negatives(positive, full) == expected
    # Given a frozen null, when choosing wire points, then no prompt is sent.
    assert e.wire_negatives(None, ((1, 1), (2, 2))) == ()


def full_negative_guards(root: Path) -> None:
    # Given a raw component with a farther negative inside, omitted from nearest3.
    full = tuple(
        sorted(
            (
                (1, 4),
                (4, 1),
                (4, 8),
                (7, 7),
                (0, 0),
                (0, 9),
                (9, 0),
                (9, 9),
                (0, 4),
                (4, 0),
                (9, 4),
                (4, 9),
            )
        )
    )
    target = r.Target("T", (4, 4), full, "identifiable", "fixture")
    image = r.RosterImage("image", "hash", 10, 10, False, (target,), ())
    wire = e.wire_negatives(target.point, full)
    assert (7, 7) not in wire
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[2:8, 2:8] = True
    folder = root / image.image_id
    folder.mkdir()
    saved = e.save_mask(folder / "raw.npy", mask)
    # When measuring both exclusion sets, then wire success cannot waive full failure.
    flags = e.prompt_checks(e.component(mask, (4, 4)), wire, full)
    assert flags == (True, True, False)
    rejected = e.Candidate(
        "image",
        "T",
        "hash",
        None,
        (4, 4),
        full,
        (saved,),
        (0.8,),
        "negative_in_raw_component",
        0,
        "identifiable",
        "fixture",
        model_called=True,
        wire_negative=wire,
        wire_points=((4, 4),) + wire,
        wire_labels=(1, 0, 0, 0),
        positive_in_component=True,
        wire_consistent=True,
        full_consistent=False,
    )
    # Given genuine provenance, when verified, then a failed target remains accounted.
    e.verify_candidates([rejected.json()], (image,), root)
    # Given a v1-shaped record without revision, when checked, then refuse it.
    legacy = rejected.json()
    del legacy["algorithm_revision"]
    rejects(
        lambda: e.verify_candidates([legacy], (image,), root),
        "candidate_revision_mismatch",
    )
    for changed, reason in (
        (replace(rejected, box=(2, 2, 8, 8)), "candidate_prompt_inconsistent"),
        (replace(rejected, negative=wire), "negative_provenance_mismatch"),
        (replace(rejected, wire_negative=full), "wire_provenance_mismatch"),
        (replace(rejected, wire_points=((4, 4),)), "wire_group_mismatch"),
        (replace(rejected, wire_labels=(0, 0, 0, 0)), "wire_group_mismatch"),
        (replace(rejected, full_consistent=True), "prompt_check_tamper"),
        (replace(rejected, algorithm_revision="v1"), "candidate_revision_mismatch"),
    ):
        # Given altered wire/provenance/flags, when verified, then refuse that change.
        rejects(
            lambda changed=changed: e.verify_candidates(
                [changed.json()], (image,), root
            ),
            reason,
        )


def frozen_guards(root: Path) -> None:
    # Given equivalent mappings with different insertion order.
    # When serialized, then exact deterministic bytes agree.
    assert r.encoded({"b": 2, "a": 1}) == r.encoded({"a": 1, "b": 2})
    path = root / "frozen.json"
    digest = r.seal(path, {"records": [None], "denominator": 1})
    # Given frozen input, when read, then its digest and null count survive.
    assert digest == r.sha(path.read_bytes()) and r.frozen(path)["denominator"] == 1
    # Given existing outputs, when reused, then they are refused, never overwritten.
    rejects(lambda: r.seal(path, {}) and None, "existing_output")
    # Given one modified frozen input byte.
    _ = path.write_bytes(path.read_bytes() + b" ")
    # When read, then the sidecar gate refuses it.
    rejects(lambda: r.frozen(path) and None, "frozen_input_tamper")
    # Given duplicate target IDs inside one roster image.
    duplicates: list[Json] = [
        {
            "target_id": "T",
            "point": None,
            "negative_points": [],
            "status": "uncertain",
            "reason": "fixture",
        }
    ] * 2
    row: JsonMap = {
        "image_id": "fixture",
        "image_sha256": "h",
        "width": 10,
        "height": 10,
        "background_potatoes": False,
        "targets": duplicates,
    }
    # When parsed, then duplicate membership is rejected.
    rejects(lambda: r.images([row]) and None, "duplicate_target_membership")


def main() -> None:
    with TemporaryDirectory(prefix="potato-instance-smoke-") as directory:
        root = Path(directory)
        for name, check in (
            ("candidate", candidate_guards),
            ("frozen", frozen_guards),
            ("wire", wire_guards),
            ("full-negative", full_negative_guards),
        ):
            folder = root / name
            folder.mkdir()
            check(folder)
    print(
        "CPU_INSTANCE_SMOKE PASS: release, hashes, membership, nulls, tamper, deterministic, exclusive"
    )


if __name__ == "__main__":
    main()
