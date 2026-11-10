"""Repeatable CPU package checks using explicitly synthetic metadata and images."""

import hashlib
import shutil
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from public_label_rework.common import OUTPUTS, sha256
from public_label_rework.labelio import parse
from public_label_rework.package import PackageInput, build
from public_label_rework.package_fixtures import (
    BLOCKED_METRICS_JSON,
    DUPLICATE_STATUS,
    IDS,
    KEPT_LABELS,
    METRICS_PATH,
    RENAMED_LABEL,
    SOURCE_IMAGES,
    SOURCE_LICENSE,
    STATUS_FILE,
    SYNTHETIC_JPG,
    UNSAFE_RENDER,
    fixture,
    fixture_paths,
    label,
)
from public_label_rework.package_inputs import (
    PackageMetrics,
    StatusRow,
    read_metrics,
    read_status,
)
from retraining_data.schema import DataError, document, mapping, require


def rejected(
    action: Callable[[], tuple[Path, str]],
    expected: type[DataError] | type[FileExistsError] = DataError,
) -> None:
    try:
        _ = action()
    except expected:
        return
    raise DataError("Expected package rejection")


def mutate(case: PackageInput, relative: str, content: bytes | None) -> None:
    """Change one explicitly synthetic input file for a rejection scenario."""
    path = case.metrics.parent / relative
    if content is None:
        _ = path.unlink()
    else:
        _ = path.write_bytes(content)


def status_counts(rows: Sequence[StatusRow]) -> tuple[int, int, int]:
    """Return kept, excluded, and needs-human counts from parsed status data."""
    return (
        sum(row["status"] == "kept" for row in rows),
        sum(row["status"] == "excluded" for row in rows),
        sum(row["status"] == "needs_human" for row in rows),
    )


def check_readme(
    readme: str, status_rows: tuple[StatusRow, ...], metrics: PackageMetrics
) -> None:
    """Compare machine-readable README tables with status and metrics inputs."""
    expected = {
        f"| {source} | {kept} | {excluded} | {human} |"
        for source in {row["image"].split("_")[1] for row in status_rows}
        for kept, excluded, human in (
            status_counts(
                [row for row in status_rows if row["image"].split("_")[1] == source]
            ),
        )
    }
    expected.add("| 전체 | {} | {} | {} |".format(*status_counts(status_rows)))
    expected.update(
        f"| {source} | {values['background_potatoes']} | {values['watermark_stock']} | {name} | {result} |"
        for source, values in metrics["per_source"].items()
        for name, result in values["thresholds"].items()
    )
    observed = {
        line
        for line in readme.splitlines()
        if line.startswith(("| original", "| 전체"))
    }
    require(observed == expected, "README numeric tables differ from inputs")
    global_rows = {
        f"| {key} | {token} |" for key, token in metrics["thresholds"].items()
    }
    require(global_rows <= set(readme.splitlines()), "README global metric tokens")


def inspect_archive(inputs: PackageInput, archive: Path) -> None:
    """Independently read every member and compare CSV/README/source/label bytes."""
    receipt = document(
        archive.with_name(f"{archive.stem}_archive_verification.json").read_bytes()
    )
    hashes = mapping(receipt["member_sha256"])
    rows = read_status(inputs.applied / "status.csv")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        require(zipped.testzip() is None, "CRC")
        require(receipt["crc"] == "passed", "Receipt CRC")
        require(
            names == sorted(set(names)) and all(n.isascii() for n in names), "Names"
        )
        require(sha256(archive) == receipt["sha256"], "Archive hash")
        require(len(names) == receipt["members"] == len(hashes), "Member count")
        require(
            all(hashlib.sha256(zipped.read(n)).hexdigest() == hashes[n] for n in names),
            "Member hashes",
        )
        expected = {"README.md", "status.csv", "changes.csv", "exclusions.csv"}
        expected.update(
            f"sources/{p.relative_to(inputs.source_metadata / 'sources').as_posix()}"
            for p in (inputs.source_metadata / "sources").rglob("*")
            if p.is_file()
        )
        expected.update(f"before_after/{p.name}" for p in inputs.renders.iterdir())
        for row in rows:
            folder = {
                "kept": "images/public",
                "needs_human": "needs_human",
                "excluded": "",
            }[row["status"]]
            if folder:
                image = row["image"]
                require(
                    (inputs.source / f"{image}.jpg").read_bytes()
                    == SYNTHETIC_JPG + image.encode(),
                    "Synthetic source mutated",
                )
                expected.update(f"{folder}/{image}.{ext}" for ext in ("jpg", "json"))
                require(
                    zipped.read(f"{folder}/{image}.jpg")
                    == (inputs.source / f"{image}.jpg").read_bytes(),
                    "JPG bytes",
                )
                raw = zipped.read(f"{folder}/{image}.json")
                require(parse(raw).imagePath == f"{image}.jpg", "JSON reference")
                applied_folder = (
                    "labels/images/public" if folder == "images/public" else folder
                )
                require(
                    raw
                    == (inputs.applied / applied_folder / f"{image}.json").read_bytes(),
                    "Label bytes",
                )
        for name in names:
            if name.startswith("sources/"):
                require(
                    zipped.read(name) == (inputs.source_metadata / name).read_bytes(),
                    "Source metadata",
                )
        for name in ("status.csv", "changes.csv", "exclusions.csv"):
            require(
                zipped.read(name) == (inputs.applied / name).read_bytes(), "CSV bytes"
            )
        require(set(names) == expected, "Exact kept/human/package membership")
        require(
            all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in zipped.infolist()),
            "Deterministic timestamps",
        )
        extracted = inputs.output.parent / f"{archive.stem}-extracted"
        zipped.extractall(extracted)
        labels = list(extracted.rglob("*.json"))
        for path in labels:
            _ = parse(path.read_bytes())
        require(
            len(labels) == sum(row["status"] != "excluded" for row in rows),
            "Extracted JSON reparse total",
        )
        check_readme(
            zipped.read("README.md").decode(), rows, read_metrics(inputs.metrics)
        )
        print(
            "INSPECT",
            archive.name,
            "statuses",
            len(rows),
            "members",
            len(names),
            "sha256",
            sha256(archive),
        )


def main() -> None:
    """Run the owned package IO regressions without real labels or approval."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="smoke-package-") as directory:
        root = Path(directory)
        with fixture_paths(root):
            run(root)
    print("PASS smoke_package: deterministic IO, rejection fixtures; fixture-only")


def run(root: Path) -> None:
    base = fixture(root)
    first_output = root / "package_check/one"
    second_output = root / "package_check/two"
    inputs = replace(base, output=first_output)
    archive, digest = build(inputs)
    inspect_archive(inputs, archive)
    require(
        hashlib.sha256(archive.read_bytes()).hexdigest() == digest, "Returned digest"
    )
    second = replace(base, output=second_output)
    second_archive, _ = build(second)
    require(archive.read_bytes() == second_archive.read_bytes(), "ZIP bytes differ")
    qa_root = OUTPUTS / "package_check/task19-attempt2/pilot"
    qa_archive = qa_root / archive.name
    qa_receipt = qa_root / "archive_verification.json"
    if qa_root.exists():
        require(
            qa_archive.read_bytes() == archive.read_bytes()
            and qa_receipt.read_bytes()
            == (inputs.output.parent / "one_archive_verification.json").read_bytes(),
            "Existing QA fixture differs from this deterministic run",
        )
    else:
        qa_root.mkdir(parents=True)
        _ = shutil.copyfile(archive, qa_archive)
        _ = shutil.copyfile(
            inputs.output.parent / "one_archive_verification.json", qa_receipt
        )

    cases: tuple[tuple[str, str, bytes | None], ...] = (
        ("needs-human", f"{KEPT_LABELS}/{IDS[2]}.json", label(IDS[2], tight=False)),
        ("renamed", f"{KEPT_LABELS}/{IDS[0]}.json", RENAMED_LABEL),
        ("missing", f"{SOURCE_IMAGES}/{IDS[0]}.jpg", None),
        ("pixel-mismatch", f"{SOURCE_IMAGES}/{IDS[0]}.jpg", b"changed pixels"),
        ("untight", f"{KEPT_LABELS}/{IDS[0]}.json", label(IDS[0], tight=False)),
        ("blocked", METRICS_PATH, BLOCKED_METRICS_JSON.encode()),
        ("non-ascii", SOURCE_LICENSE, b"synthetic fixture"),
        ("unsafe", UNSAFE_RENDER, b"synthetic fixture"),
        ("duplicate", STATUS_FILE, DUPLICATE_STATUS),
    )
    for name, relative, content in cases:
        case = fixture(root / name)
        mutate(case, relative, content)
        rejected(lambda case=case: build(case))

    existing = fixture(root / "existing-output")
    existing.output.mkdir(parents=True)
    rejected(lambda: build(existing), FileExistsError)


if __name__ == "__main__":
    main()
