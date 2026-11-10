"""Validate existing task16 artifacts; never run or tune a G-full evaluation."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, assert_never

from public_label_rework import apply_scope as scope
from public_label_rework.common import sha256
from public_label_rework.decisions import ApplyInput
from public_label_rework.package_inputs import (
    PackageInput,
    PackageMetrics,
    StatusRow,
    source_hashes,
)
from retraining_data.schema import (
    DataError,
    Json,
    array,
    integer,
    mapping,
    number,
    require,
    text,
)

# Only these metrics have numerical delivery thresholds in the approved plan.
LIMITS: Final = {
    "precision_at_0_5": (0.97, 1.0),
    "recall_at_0_5": (0.97, 1.0),
    "duplicates": (0.0, 0.0),
    "median_iou": (0.85, 1.0),
    "median_area_ratio": (0.95, 1.05),
    "p90_area_ratio": (0.0, 1.15),
    "iou_below_0_7_share": (0.0, 0.05),
}
REPORTS: Final = {"precision_at_0_75", "recall_at_0_75", "count_mae"}
SHA: Final = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class Evaluation:
    started_at: datetime
    completed_at: datetime
    global_values: dict[str, float]
    per_source: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class Verification:
    artifact: Path
    applied_tree: Path
    final_tree_sha256: str
    metrics_sha256: str
    frozen_at: datetime
    evaluation: Evaluation


def timestamp(value: Json) -> datetime:
    """Parse an explicit timezone-aware ISO timestamp at the artifact boundary."""
    parsed = datetime.fromisoformat(text(value))
    require(parsed.tzinfo is not None, "Verification timestamp requires timezone")
    return parsed.astimezone(UTC)


def measurements(value: Json) -> dict[str, float]:
    """Require all delivery thresholds and the task12 reporting-only metrics."""
    row = mapping(value)
    require(set(row) == set(LIMITS) | REPORTS, "Required G-full metrics mismatch")
    values = {key: number(value) for key, value in row.items()}
    for key, (low, high) in LIMITS.items():
        require(low <= values[key] <= high, f"G-full threshold failed: {key}")
    require(
        0 <= values["precision_at_0_75"] <= 1
        and 0 <= values["recall_at_0_75"] <= 1
        and values["count_mae"] >= 0,
        "Invalid reporting-only G-full metrics",
    )
    return values


def read_verification(path: Path) -> Verification:
    """Parse the small versioned task16 artifact into a typed value."""
    scope.plain_file(path)
    row = mapping(scope.json_value(path.read_bytes()))
    require(
        set(row)
        == {
            "schema_version",
            "applied_tree",
            "final_tree_sha256",
            "metrics_sha256",
            "frozen_at",
            "validators",
            "blind_audit_defect_free_rate",
            "unexplained_family_mismatches",
            "gfull_evaluations",
            "evaluations",
        },
        "Verification fields mismatch",
    )
    require(row["schema_version"] == "package-verification-v1", "Verification version")
    require(row["validators"] == "PASS", "Validators must PASS")
    require(
        0.95 <= number(row["blind_audit_defect_free_rate"]) <= 1,
        "Blind audit defect-free rate must be >= .95",
    )
    require(
        integer(row["unexplained_family_mismatches"]) == 0,
        "Unexplained family mismatches must be zero",
    )
    evaluations = array(row["evaluations"])
    require(
        integer(row["gfull_evaluations"]) == len(evaluations) == 1,
        "Exactly one G-full evaluation is required",
    )
    evaluation = mapping(evaluations[0])
    require(
        set(evaluation) == {"started_at", "completed_at", "global", "per_source"},
        "G-full evaluation fields mismatch",
    )
    return Verification(
        path,
        Path(text(row["applied_tree"])),
        text(row["final_tree_sha256"]),
        text(row["metrics_sha256"]),
        timestamp(row["frozen_at"]),
        Evaluation(
            timestamp(evaluation["started_at"]),
            timestamp(evaluation["completed_at"]),
            measurements(evaluation["global"]),
            {
                key: measurements(value)
                for key, value in mapping(evaluation["per_source"]).items()
            },
        ),
    )


def frozen_files(inputs: PackageInput) -> dict[str, Path]:
    """Enumerate every frozen file, including logs and all consumed auxiliaries."""
    files: dict[str, Path] = {}
    for prefix, root in (
        ("applied", inputs.applied),
        ("renders", inputs.renders),
        ("source_metadata", inputs.source_metadata),
    ):
        require(root.absolute() == root.resolve() and root.is_dir(), "Frozen root")
        for path in sorted(root.rglob("*")):
            require(not path.is_symlink(), "Frozen tree contains symlink")
            if path.is_file():
                name = f"{prefix}/{path.relative_to(root).as_posix()}"
                require(
                    "\\" not in name and "\n" not in name,
                    "Unsafe frozen relative path",
                )
                files[name] = path
    return files


def verify_freeze(inputs: PackageInput, proof: Verification) -> None:
    """Check exact complete relative-file manifest and hashes before any output."""
    manifest = proof.artifact.parent / "final_tree.sha256"
    scope.plain_file(manifest)
    files = frozen_files(inputs)
    for external in (manifest, proof.artifact, inputs.metrics):
        require(
            not any(
                external.resolve().is_relative_to(root.resolve())
                for root in (inputs.applied, inputs.renders, inputs.source_metadata)
            ),
            "Proof and metrics must be outside frozen roots",
        )
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ")
        require(len(parts) == 2, "Malformed frozen manifest line")
        digest, name = parts
        require(
            SHA.fullmatch(digest) is not None and name not in expected,
            "Invalid or duplicate frozen manifest entry",
        )
        expected[name] = digest
    require(set(expected) == set(files), "Incomplete frozen manifest membership")
    require(sha256(manifest) == proof.final_tree_sha256, "Frozen manifest SHA mismatch")
    require(sha256(inputs.metrics) == proof.metrics_sha256, "Metrics SHA mismatch")
    for name, path in files.items():
        require(sha256(path) == expected[name], f"Frozen file SHA mismatch: {name}")
        require(
            path.stat().st_mtime <= proof.frozen_at.timestamp(),
            f"Frozen file newer than freeze: {name}",
        )
    require(
        manifest.stat().st_mtime
        <= proof.frozen_at.timestamp()
        < proof.evaluation.started_at.timestamp()
        <= proof.evaluation.completed_at.timestamp()
        <= proof.artifact.stat().st_mtime,
        "Freeze must precede the single G-full evaluation and verification",
    )


def delivery_gate(
    inputs: PackageInput, metrics: PackageMetrics, rows: tuple[StatusRow, ...]
) -> None:
    """Keep pilot FAILs honest; final requires scope and byte-bound task16 proof."""
    match metrics["mode"]:
        case "pilot":
            require(metrics["verification_status"] == "pilot", "Pilot status mismatch")
            return
        case "final":
            pass
        case _:
            assert_never(metrics["mode"])
    require(
        metrics["verification_status"] == "passed"
        and metrics["gfull_evaluations"] == 1
        and not metrics["fixture_only"],
        "Final delivery requires successful single G-full verification",
    )
    for tokens in (
        metrics["thresholds"],
        *(value["thresholds"] for value in metrics["per_source"].values()),
    ):
        require(
            set(tokens) == set(LIMITS) and set(tokens.values()) == {"PASS"},
            "Required G-full threshold tokens mismatch",
        )
    images = scope.selected_images(
        ApplyInput(inputs.source, scope.PILOT, (), remaining_list=scope.REMAINING)
    )
    require(
        {row["image"] for row in rows} == set(images),
        "Final statuses must equal frozen 2060 union",
    )
    require(
        set(source_hashes(inputs.source_metadata)) == set(images),
        "Final source manifest membership mismatch",
    )
    artifact = inputs.verification
    if artifact is None:
        raise DataError("Final requires task16 verification")
    require(artifact.name == "verification.json", "Use verification.json")
    proof = read_verification(artifact)
    scope.canonical(inputs.applied, proof.applied_tree, "Applied tree proof mismatch")
    require(
        set(proof.evaluation.per_source) == set(metrics["per_source"]),
        "G-full per-source proof membership mismatch",
    )
    verify_freeze(inputs, proof)
