# /// script
# requires-python = ">=3.12"
# dependencies = ["Pillow"]
# ///
# How to run: /home/sc/.cache/potato-baseline/.venv/bin/python -B -m retraining_data.smoke
"""Small deterministic CPU proofs using real sources and disposable owned output."""

import argparse
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from retraining_data.common import (
    DATASET,
    OUTPUTS,
    REVIEW,
    ROOT,
    digest,
    eligibility,
    output_path,
    yolo,
)
from retraining_data.native import Shape, parse_shape, read_native, read_review_manifest
from retraining_data.prepare import prepare, publish
from retraining_data.schema import (
    V1,
    V2,
    DataError,
    document,
    profile_named,
    require,
    validate_records,
)
from retraining_data.sources import build_plan
from retraining_data.verify import Options, verify_dataset, verify_files


def rejected[T](
    action: Callable[[], T],
    error: type[DataError] | type[FileExistsError],
    name: str,
) -> None:
    """Require the actual operation to fail with the specified boundary error."""
    try:
        _ = action()
    except error:
        return
    raise DataError(f"Smoke expected rejection: {name}")


def conversion_proofs() -> None:
    """Cover empty, full-edge, four-corner and invalid native rectangle inputs."""
    # Given: a real empty reviewed frame.
    native = read_native((REVIEW / "candidates/train_t000001000ms.json").read_bytes())
    # When: converting its native shapes.
    empty = yolo(native)
    # Then: a negative sample is a real empty file.
    require(empty == b"", "empty label conversion")

    # Given: a two-corner potato and four-corner stone touching every image edge.
    shapes = (
        Shape(label="potato", points=((0.0, 0.0), (1280.0, 720.0))),
        Shape(label="stone",
              points=((0.0, 720.0), (1280.0, 720.0), (1280.0, 0.0), (0.0, 0.0))),
    )
    # When: converting pixel coordinates to YOLO.
    converted = yolo(replace(native, shapes=shapes))
    # Then: both class IDs and all nine fractional digits are exact.
    require(
        converted == b"0 0.500000000 0.500000000 1.000000000 1.000000000\n"
        + b"1 0.500000000 0.500000000 1.000000000 1.000000000\n",
        "edge/class conversion",
    )
    for points in (
        ((-1.0, 0.0), (10.0, 10.0)), ((0.0, 0.0), (1281.0, 10.0)),
        ((0.0, 0.0), (0.0, 10.0)),
        ((0.0, 0.0), (10.0, 0.0), (9.0, 10.0), (0.0, 10.0)),
    ):
        # Given: out-of-bounds, empty or nonrectangular corners.
        shape = Shape(label="stone", points=points)
        # When/Then: real conversion rejects instead of clipping or inventing a box.
        rejected(partial(yolo, replace(native, shapes=(shape,))), DataError, "invalid corners")
    # Given: nonfinite corner data at the JSON boundary.
    # When/Then: strict parsing rejects it before conversion.
    rejected(
        lambda: parse_shape(document(
            b'{"label":"stone","shape_type":"rectangle","points":[[NaN,0],[10,10]]}',
        )),
        DataError, "nonfinite rectangle",
    )


def eligibility_proofs() -> None:
    """Reject candidate/native disagreement without modifying any review input."""
    manifest = read_review_manifest(REVIEW / "manifest.json")
    row = manifest.frames[0]
    raw = (REVIEW / row.relative_image).with_suffix(".json").read_bytes()
    native = read_native(raw)
    # Given: the manifest claims this native candidate is eligible.
    changed = replace(native, flags=replace(native.flags, needs_confirmation=True))
    # When/Then: eligibility must reject, rather than trusting manifest alone.
    rejected(lambda: eligibility(row, changed), DataError, "needs_confirmation mismatch")
    changed_review = replace(native.review, eligible_for_experimental_training=False)
    # Given: native eligibility disagrees with manifest.
    # When/Then: native exclusion wins by refusing the inconsistent source.
    rejected(
        lambda: eligibility(row, replace(native, review=changed_review)),
        DataError, "native eligibility mismatch",
    )
    # Given: checked=true or human_reviewed=true serialized native input.
    for invalid in (
        raw.replace(b'"checked": false', b'"checked": true'),
        raw.replace(b'"human_reviewed": false', b'"human_reviewed": true'),
    ):
        # When/Then: parsing rejects automatic review promotion.
        rejected(partial(read_native, invalid), DataError, "review promotion")


def v2_proofs() -> None:
    """Exercise real v2 routing, restored legacy bytes and profile rejection."""
    # Given: the chief-created v2 source and unchanged legacy inputs.
    plan = build_plan(V2)
    records = plan.snapshot.frames
    selected = [r for r in records if r.source_image.startswith("label_review_v2/")]
    held = next(e for e in plan.snapshot.exclusions if e.source == "label_review_v2/needs_confirmation")
    # When: inspecting the actual bounded selection.
    names = {r.filename for r in records}
    # Then: all 48 candidates, but neither held native frame, enter the 111-frame plan.
    require(len(selected) == 48 and len(records) == 111, "v2 selection counts")
    require(held.observations == 2 and not names.intersection(held.filenames), "v2 held exclusion")
    manifest = read_review_manifest(ROOT / "label_review_v2/manifest.json")
    require({r.requested_second for r in manifest.frames if r.filename in held.filenames} == {65, 71},
            "65 and 71 top fragments remain held")
    require({r.requested_second for r in selected}.issuperset({43, 43.5, 45, 47, 47.5, 49}),
            "confirmed slab candidates restored")
    inventory = {s.path for s in plan.snapshot.source_files}
    require("label_review_v2/user_confirmation.json" in inventory
            and not any(p.startswith("label_review/") for p in inventory), "v2 source isolation")
    for second in (44, 46, 48):
        restored = next(p for p in plan.payloads if p.record.filename == f"frame_{second:06d}.png")
        require(restored.label == (ROOT / restored.record.source_annotation).read_bytes(),
                "restored legacy labels unchanged")
    require(plan.snapshot.human_reviewed is False and plan.snapshot.native_checked is False,
            "v2 AI status preserved")
    # Given: real v2 records passed to the v1 count boundary.
    # When/Then: v1 must not silently accept the new membership.
    rejected(partial(validate_records, records), DataError, "v2 records under v1")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="smoke_v2_", dir=OUTPUTS) as temporary:
        output = Path(temporary) / "dataset"
        # Given: a disposable destination, never the canonical dataset_v2.
        # When: driving both public commands with explicit v2 selection.
        for module, option in (("prepare", "--output"), ("verify", "--dataset")):
            _ = subprocess.run(
                [sys.executable, "-B", "-m", f"retraining_data.{module}",
                 "--profile", "v2", option, str(output)], cwd=ROOT, check=True,
            )
        # Then: the published files match the selected source and reject default v1.
        _ = verify_files(output, plan)
        rejected(partial(verify_dataset, output), DataError, "v2 output under default v1")
        # Given: the immutable canonical v1 dataset and a v2 reconstruction.
        # When/Then: the opposite profile mismatch also fails.
        rejected(partial(verify_files, DATASET, plan), DataError, "v1 output under v2")
    print("CPU v2 smoke passed: 48/2 input selection, restored legacy 44/46/48, profile isolation")


def main() -> None:
    """Run real publication, overwrite refusal and actual file tampering proofs."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--profile", type=profile_named, choices=(V1, V2), default=V1,
                            metavar="{v1,v2}")
    args = parser.parse_args(namespace=Options())
    if args.profile == V2:
        v2_proofs()
        return
    conversion_proofs()
    eligibility_proofs()
    # Given: all actual legacy/review sources, not a mock of dataset planning.
    plan = build_plan()
    names = {p.record.filename for p in plan.payloads}
    held = next(e for e in plan.snapshot.exclusions if e.observations == 8)
    # When: assembling membership and exclusions.
    selected = [p for p in plan.payloads if p.record.provenance == "AI_visual_review_not_human_expert"]
    # Then: exactly 42 selected and three whole legacy frames plus all eight held absent.
    require(len(selected) == 42 and len(plan.payloads) == 102, "42 candidates/102 total")
    require(not names.intersection(held.filenames), "all eight held frames excluded")
    require(not names.intersection({"frame_000044.png", "frame_000046.png", "frame_000048.png"}), "slabs excluded")
    require(any(p.record.requested_second == 15.5 for p in selected), "fractional requested second preserved")
    require(plan.snapshot.human_reviewed is False and plan.snapshot.native_checked is False, "AI status preserved")
    # Given: unchanged counts but an arbitrary safe name for one fractional-time candidate.
    records = tuple(p.record for p in plan.payloads)
    renamed = tuple(replace(r, filename="arbitrary_unique.png") if r == selected[0].record else r for r in records)
    # When/Then: the separate schema accepts it without filename-derived timestamps.
    validate_records(renamed)
    # Given: a validation image carrying exact training pixels.
    duplicate = tuple(
        replace(r, pixel_sha256=records[0].pixel_sha256) if r.split == "val" else r for r in records
    )
    # When/Then: exact cross-split pixels must be rejected.
    rejected(partial(validate_records, duplicate), DataError, "cross-split pixel duplicate")
    # Given: a validation record whose requested time crosses its interval.
    wrong_time = tuple(replace(r, requested_second=120.0) if r.split == "val" else r for r in records)
    # When/Then: temporal validation rejects it even though counts remain valid.
    rejected(partial(validate_records, wrong_time), DataError, "split time range")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="smoke_", dir=OUTPUTS) as temporary:
        output = Path(temporary) / "dataset"
        # Given: a never-used owned output directory.
        # When: publishing and independently reading actual output files.
        _ = subprocess.run(
            [sys.executable, "-B", "-m", "retraining_data.prepare", "--output", str(output)],
            cwd=ROOT, check=True,
        )
        snapshot = verify_files(output, plan)
        # Then: all counts, copied labels, image bytes and source hashes match.
        require(snapshot == plan.snapshot, "actual output verifies")
        _ = subprocess.run(
            [sys.executable, "-B", "-m", "retraining_data.verify", "--dataset", str(output)],
            cwd=ROOT, check=True,
        )
        # Given: the actual output now exists.
        # When/Then: the public builder refuses it before reading or changing sources.
        rejected(lambda: prepare(output), FileExistsError, "existing output")
        rejected(lambda: output_path(REVIEW / "forbidden"), DataError, "outside owned outputs")
        for relative in (
            "labels/train/train_t000001000ms.txt", "labels/val/frame_000092.txt",
            "images/train/train_t000001000ms.png", "annotations.json", "data.yaml",
        ):
            # Given: saved actual output bytes.
            path = output / relative
            original = path.read_bytes()
            _ = path.write_bytes(original + b"tamper")
            try:
                # When/Then: verifying the modified file must fail.
                rejected(lambda: verify_files(output, plan), DataError, f"actual file tamper: {relative}")
            finally:
                _ = path.write_bytes(original)
        # Given: a byte-identical valid output plus an unexpected label file.
        extra = output / "labels/train/extra.txt"
        _ = extra.write_bytes(b"")
        try:
            # When/Then: checking membership catches even an empty extra label.
            rejected(lambda: verify_files(output, plan), DataError, "extra membership")
        finally:
            extra.unlink()
        # Given: unchanged sources and a different new destination.
        second = Path(temporary) / "again"
        # When: publishing the same plan again.
        publish(plan, second)
        # Then: every artifact, including manifest and snapshot, is reproducible.
        for path in output.rglob("*"):
            if path.is_file():
                require(
                    digest(path.read_bytes()) == digest((second / path.relative_to(output)).read_bytes()),
                    "byte reproducibility",
                )
    print("CPU smoke passed: inclusion/exclusion, conversion, eligibility, overwrite, tamper, reproducibility")


if __name__ == "__main__":
    main()
