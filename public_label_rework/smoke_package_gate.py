"""Real apply-to-builder proofs for pilot subsets and frozen final2060 delivery."""

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from public_label_rework import apply_scope as scope
from public_label_rework.apply import apply
from public_label_rework.common import OUTPUTS, sha256
from public_label_rework.decisions import ApplyInput
from public_label_rework.package import build
from public_label_rework.package import main as package_main
from public_label_rework.package_fixtures import (
    IDS,
    TOKENS,
    final_artifacts,
    fixture,
    fixture_paths,
    write,
)
from public_label_rework.package_inputs import PackageInput
from public_label_rework.smoke_package import inspect_archive
from retraining_data.schema import DataError, JsonMap, array, document, mapping, require


def denied(inputs: PackageInput, expected: str) -> None:
    """Only a real rejection before creating any package counts."""
    try:
        _ = build(inputs)
    except (DataError, OSError, ValueError) as error:
        require(expected in str(error), f"Wrong rejection: {error}")
        require(not inputs.output.exists(), "Rejected proof created package")
        print(f"REJECT {expected}: {error}")
        return
    raise DataError(f"Accepted invalid package: {expected}")


def proof_rejections(inputs: PackageInput) -> None:
    """Given one valid frozen producer output, change one contract at a time."""
    path = inputs.verification
    if path is None:
        raise DataError("Missing fixture proof")
    baseline = path.read_bytes()
    proof = document(baseline)
    evaluation = mapping(array(proof["evaluations"])[0])
    variants: tuple[tuple[JsonMap, str], ...] = (
        ({"validators": "FAIL"}, "Validators"),
        ({"blind_audit_defect_free_rate": 0.94}, "Blind audit"),
        ({"unexplained_family_mismatches": 1}, "family mismatches"),
        ({"gfull_evaluations": 2}, "Exactly one"),
        ({"evaluations": [evaluation, evaluation]}, "Exactly one"),
        ({"evaluations": []}, "Exactly one"),
        ({"applied_tree": str(inputs.applied.parent)}, "Applied tree"),
        ({"frozen_at": "2026-01-03T00:00:00+00:00"}, "Freeze must precede"),
        ({"metrics_sha256": "0" * 64}, "Metrics SHA"),
        ({"final_tree_sha256": "0" * 64}, "manifest SHA"),
        (
            {"evaluations": [{**evaluation, "global": {"unrelated": 1}}]},
            "Required G-full",
        ),
        ({"evaluations": [{**evaluation, "per_source": {}}]}, "per-source"),
    )
    for changes, expected in variants:
        # When: existing artifact is invalid; Then: build refuses before writing.
        write(path, {**proof, **changes})
        denied(inputs, expected)
    for source in (None, "original02"):
        for name, value in (
            ("precision_at_0_5", 0.96),
            ("recall_at_0_5", 0.96),
            ("duplicates", 1),
            ("median_iou", 0.84),
            ("median_area_ratio", 1.06),
            ("p90_area_ratio", 1.16),
            ("iou_below_0_7_share", 0.051),
        ):
            changed = document(baseline)
            local = mapping(array(changed["evaluations"])[0])
            values = mapping(
                local["global"]
                if source is None
                else mapping(local["per_source"])[source]
            )
            values[name] = value
            write(path, changed)
            denied(inputs, "G-full threshold failed")
    _ = path.write_bytes(baseline)
    # Changed labels, logs, rendered inputs and attribution must all be bound.
    for target in (
        inputs.applied / f"labels/images/public/{IDS[0]}.json",
        inputs.applied / "changes.jsonl",
        inputs.applied / "status.csv",
        inputs.renders / "sample_before_after.png",
        inputs.source_metadata / "sources/original01/data.yaml",
    ):
        raw, stat = target.read_bytes(), target.stat()
        _ = target.write_bytes(
            raw.replace(b"synthetic", b"fixture")
            if target.name == "status.csv"
            else raw + b" "
        )
        denied(inputs, "Frozen file SHA")
        _ = target.write_bytes(raw)
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    manifest = path.parent / "final_tree.sha256"
    raw, stat = manifest.read_bytes(), manifest.stat()
    for content in (b"", b"\n".join(raw.splitlines()[1:]) + b"\n"):
        _ = manifest.write_bytes(content)
        os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        write(path, {**proof, "final_tree_sha256": sha256(manifest)})
        denied(inputs, "Incomplete frozen manifest")
    _ = manifest.write_bytes(raw)
    _ = path.write_bytes(baseline)
    denied(inputs, "Freeze must precede")
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    _ = path.write_bytes(baseline)
    manifest.unlink()
    denied(inputs, "Missing or symlinked")
    _ = manifest.write_bytes(raw)
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    denied(replace(inputs, verification=None), "requires task16")
    missing = replace(inputs, verification=path.parent / "missing/verification.json")
    denied(missing, "verification.json")


def scope_and_metrics_rejections(inputs: PackageInput) -> None:
    # Given: a valid final2060 fixture; When: scope or metric evidence changes.
    raw = inputs.metrics.read_bytes()
    baseline = document(raw)
    variants: tuple[tuple[JsonMap, str], ...] = (
        ({"verification_status": "blocked_gfull_fail"}, "successful single"),
        ({"fixture_only": True}, "successful single"),
        ({"gfull_evaluations": 0}, "successful single"),
        ({"gfull_evaluations": 2}, "successful single"),
        ({"thresholds": {"unrelated": "PASS"}}, "threshold tokens"),
        ({"thresholds": {**TOKENS, "median_iou": "FAIL"}}, "threshold tokens"),
        (
            {
                "per_source": {
                    "original01": {
                        "background_potatoes": 0,
                        "watermark_stock": 0,
                        "thresholds": {"unrelated": "PASS"},
                    }
                }
            },
            "threshold tokens",
        ),
        ({"harvest_convention": "changed metrics bytes"}, "Metrics SHA"),
    )
    for changes, expected in variants:
        write(inputs.metrics, {**baseline, **changes})
        denied(inputs, expected)
    _ = inputs.metrics.write_bytes(raw)
    for target, content, expected in (
        (scope.CONFIRMATION, b"{}", "Confirmation requires"),
        (scope.PILOT.with_suffix(".sha256"), b"0" * 64, "SHA-256 mismatch"),
        (scope.REMAINING.with_suffix(".sha256"), b"0" * 64, "SHA-256 mismatch"),
        (
            inputs.applied / "status.csv",
            (inputs.applied / "status.csv").read_bytes().splitlines(keepends=True)[0]
            + b"".join(
                (inputs.applied / "status.csv")
                .read_bytes()
                .splitlines(keepends=True)[1:4]
            ),
            "Final statuses",
        ),
    ):
        original, stat = target.read_bytes(), target.stat()
        _ = target.write_bytes(content)
        denied(inputs, expected)
        _ = target.write_bytes(original)
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    approval = scope.CONFIRMATION.read_bytes()
    scope.CONFIRMATION.unlink()
    denied(inputs, "Missing or symlinked")
    _ = scope.CONFIRMATION.write_bytes(approval)


def main() -> None:
    """Native /tmp fixture IO; path-root redirection never authorizes real data."""
    with TemporaryDirectory(prefix="package-gate-attempt2-") as directory:
        root = Path(directory)
        with fixture_paths(root):
            base = fixture(root, full=True)
            approval = scope.CONFIRMATION.read_bytes()
            scope.CONFIRMATION.unlink()
            pilot_ops = root / "pilot_ops.jsonl"
            _ = pilot_ops.write_bytes(
                b"".join(
                    (root / "ops.jsonl").read_bytes().splitlines(keepends=True)[:100]
                )
            )
            _ = apply(
                ApplyInput(base.source, scope.PILOT, (pilot_ops,)),
                root / "pilot-applied",
            )
            pilot = replace(
                base,
                applied=root / "pilot-applied",
                output=root / "package_check/pilot",
            )
            archive, _ = build(pilot)
            inspect_archive(pilot, archive)
            require(not scope.CONFIRMATION.exists(), "Pilot created approval")
            _ = scope.CONFIRMATION.write_bytes(approval)
            malicious = final_artifacts(fixture(root / "malicious-three"))
            denied(replace(malicious, verification=None), "canonical public source")
            final = final_artifacts(base)
            proof_rejections(final)
            scope_and_metrics_rejections(final)
            archive, _ = build(final)
            inspect_archive(final, archive)
            argv = [
                "package",
                *(
                    str(p)
                    for p in (
                        final.source,
                        final.source_metadata,
                        final.applied,
                        final.metrics,
                        final.renders,
                        root / "cli",
                    )
                ),
                "--verification",
                str(final.verification),
            ]
            with patch.object(sys, "argv", argv):
                package_main()
            require(
                archive.read_bytes() == (root / "cli.zip").read_bytes(), "API/CLI bytes"
            )
            qa = OUTPUTS / f"package_check/task19-attempt2/final-fixture-{root.name}"
            qa.mkdir(parents=True, exist_ok=False)
            for path in (
                archive,
                archive.with_name("fixture_archive_verification.json"),
                final.metrics,
                root / "task16/verification.json",
                root / "task16/final_tree.sha256",
                final.output / "README.md",
            ):
                _ = shutil.copyfile(path, qa / path.name)
            _ = shutil.copyfile(root / "package_check/pilot.zip", qa / "pilot100.zip")
            _ = shutil.copyfile(
                root / "package_check/pilot_archive_verification.json",
                qa / "pilot100_archive_verification.json",
            )
            _ = shutil.copyfile(
                root / "package_check/pilot/README.md", qa / "pilot100_README.md"
            )
    print("PASS smoke_package_gate; real producer, pilot subset, final2060, cleanup")


if __name__ == "__main__":
    main()
