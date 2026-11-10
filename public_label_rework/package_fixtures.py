"""Owned synthetic package inputs; real apply, JPEGs, CSVs and frozen proofs."""

import hashlib
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Final
from unittest.mock import patch

from PIL import Image

from public_label_rework import apply_scope as scope
from public_label_rework import common
from public_label_rework.apply import apply
from public_label_rework.decisions import ApplyInput, FixtureGate
from public_label_rework.labelio import box_id, parse
from public_label_rework.package_inputs import PackageInput
from retraining_data.schema import Json, JsonMap

IDS: Final = (
    "public_original01_0001",
    "public_original01_0002",
    "public_original02_0003",
)
KEPT_LABELS: Final = "applied/labels/images/public"
SOURCE_IMAGES: Final = "source"
METRICS_PATH: Final = "metrics.json"
SOURCE_LICENSE: Final = "metadata/sources/original01/출처.txt"
STATUS_FILE: Final = "applied/status.csv"
UNSAFE_RENDER: Final = "renders/..\\escape.png"
METRICS_JSON: Final = '{"mode":"pilot","fixture_only":true,"verification_status":"pilot","gfull_evaluations":0,"thresholds":{"fixture_metric":"FAIL"},"harvest_convention":"synthetic fixture input","per_source":{"original01":{"background_potatoes":1,"watermark_stock":0,"thresholds":{"fixture_metric":"FAIL"}},"original02":{"background_potatoes":2,"watermark_stock":1,"thresholds":{"fixture_metric":"PASS"}}}}'
BLOCKED_METRICS_JSON: Final = METRICS_JSON.replace(
    '"mode":"pilot"', '"mode":"final"'
).replace('"verification_status":"pilot"', '"verification_status":"blocked_gfull_fail"')
DUPLICATE_STATUS: Final = f"\ufeffimage,status,reason,batch\r\n{IDS[0]},kept,,fixture\r\n{IDS[0]},kept,,fixture\r\n".encode()
LABEL_TEMPLATE: Final = '{"version":"4.1.1","flags":{"human_reviewed":false},"checked":false,"shapes":[{"label":"potato","points":[[1,1],[8,1],[8,8],[1,8]],"group_id":null,"description":"fixture only","difficult":false,"shape_type":"rectangle","flags":{"tight":__TIGHT__},"attributes":{}}],"imagePath":"__IMAGE__.jpg","imageData":null,"imageHeight":10,"imageWidth":10,"description":""}'
VALUES: Final[dict[str, float]] = {
    "precision_at_0_5": 1,
    "recall_at_0_5": 1,
    "duplicates": 0,
    "median_iou": 1,
    "median_area_ratio": 1,
    "p90_area_ratio": 1,
    "iou_below_0_7_share": 0,
    "precision_at_0_75": 1,
    "recall_at_0_75": 1,
    "count_mae": 0,
}
TOKENS: Final = {
    key: "PASS"
    for key in VALUES
    if key not in {"precision_at_0_75", "recall_at_0_75", "count_mae"}
}
FREEZE_TIME: Final = 1767225600  # 2026-01-01 UTC; fixture timestamps only.


def label(image: str, *, tight: bool) -> bytes:
    return (
        LABEL_TEMPLATE.replace("__IMAGE__", image)
        .replace("__TIGHT__", str(tight).lower())
        .encode()
    )


RENAMED_LABEL: Final = label("renamed_image", tight=True)


with Image.new("RGB", (10, 10), (40, 80, 100)) as _image, BytesIO() as _stream:
    _image.save(_stream, format="JPEG")
    SYNTHETIC_JPG: Final = _stream.getvalue()


def write(path: Path, value: Json) -> None:
    _ = path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@contextmanager
def fixture_paths(root: Path) -> Generator[None]:
    """Redirect owned path roots only; every producer and consumer stays real."""
    with (
        patch.object(common, "OUTPUTS", root),
        patch.multiple(
            scope,
            OUTPUTS=root,
            SOURCE=root / "source",
            PILOT=root / "lists/pilot_list.json",
            REMAINING=root / "lists/remaining_list.json",
            CONFIRMATION=root / "pilot/user_confirmation.json",
        ),
    ):
        yield


def fixture(root: Path, *, full: bool = False) -> PackageInput:
    """Build actual apply output; never construct status/label output by hand."""
    root.mkdir(parents=True, exist_ok=True)
    work = root if full else root / "smoke-decisions-fixture"
    source, metadata, renders = work / "source", work / "metadata", work / "renders"
    for directory in (
        source,
        metadata / "sources",
        renders,
        work / "lists",
        work / "pilot",
    ):
        directory.mkdir(parents=True)
    images = IDS
    if full:
        images += tuple(f"public_original02_{i:04d}" for i in range(4, 2061))
    registry: JsonMap = {}
    for name in ("original01", "original02"):
        registry[name] = {
            "url": f"https://fixture.invalid/{name}",
            "license": "CC BY 4.0",
        }
        folder = metadata / "sources" / name
        folder.mkdir()
        for filename in ("README.roboflow.txt", "data.yaml"):
            _ = (folder / filename).write_text(f"SYNTHETIC {name}; CC BY 4.0\n")
    for image in images:
        _ = (source / f"{image}.jpg").write_bytes(SYNTHETIC_JPG + image.encode())
        _ = (source / f"{image}.json").write_bytes(label(image, tight=False))
    _ = (metadata / "사용안내.txt").write_text(
        "Synthetic intake guide\n", encoding="utf-8"
    )
    write(
        metadata / "manifest.json",
        {
            "public_sources": registry,
            "records": [
                {
                    "image": f"images/public/{image}.jpg",
                    "image_sha256": hashlib.sha256(
                        (source / f"{image}.jpg").read_bytes()
                    ).hexdigest(),
                }
                for image in images
            ],
        },
    )
    listing = work / "list.json"
    write(listing, list(IDS))
    operations: list[Json] = []
    doc = parse(label(IDS[0], tight=False))
    for index, image in enumerate(images if full else IDS):
        row: JsonMap = {"image": image, "batch": "synthetic"}
        if index == 1:
            row.update(op="drop_image", reason="junk")
        elif index == 2:
            row.update(op="needs_human", reason="synthetic")
        else:
            row.update(
                op="keep",
                box_id=box_id(0, doc.shapes[0]),
                refined_xyxy=[1, 1, 8, 8],
                snap={
                    "src": "sam",
                    "xyxy": [1, 1, 8, 8],
                    "tight": True,
                    "reason": "accepted",
                    "mask_area_ratio": 0.8,
                },
            )
        operations.append(row)
    ops = work / "ops.jsonl"
    _ = ops.write_text("".join(json.dumps(row) + "\n" for row in operations))
    inputs = ApplyInput(source, listing, (ops,), FixtureGate(work))
    if full:
        for path, selected in (
            (scope.PILOT, images[:100]),
            (scope.REMAINING, images[100:]),
        ):
            write(path, list(selected))
            _ = path.with_suffix(".sha256").write_text(common.sha256(path))
        write(
            scope.CONFIRMATION,
            {
                "date": "2026-10-29",
                "basis": "Synthetic test only",
                "statement_verbatim": "Synthetic authorization; not production approval",
            },
        )
        inputs = ApplyInput(source, scope.PILOT, (ops,), remaining_list=scope.REMAINING)
    with patch.object(scope, "OUTPUTS", root):
        _ = apply(inputs, work / "applied")
    _ = (renders / "sample_before_after.png").write_bytes(b"synthetic render fixture")
    _ = (work / "metrics.json").write_text(METRICS_JSON)
    return PackageInput(
        source,
        metadata,
        work / "applied",
        work / "metrics.json",
        renders,
        work / "package_check/fixture",
    )


def final_artifacts(inputs: PackageInput) -> PackageInput:
    """Simulate task16 serialization only, not a real G-full/visual quality claim."""
    metrics: JsonMap = {
        "mode": "final",
        "fixture_only": False,
        "verification_status": "passed",
        "gfull_evaluations": 1,
        "thresholds": dict(TOKENS),
        "harvest_convention": "synthetic fixture input",
        "per_source": {
            source: {
                "background_potatoes": 0,
                "watermark_stock": 0,
                "thresholds": dict(TOKENS),
            }
            for source in ("original01", "original02")
        },
    }
    write(inputs.metrics, metrics)
    proof_dir = inputs.applied.parent / "task16"
    proof_dir.mkdir()
    entries: list[str] = []
    for prefix, tree in (
        ("applied", inputs.applied),
        ("renders", inputs.renders),
        ("source_metadata", inputs.source_metadata),
    ):
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                entries.append(
                    f"{common.sha256(path)}  {prefix}/{path.relative_to(tree).as_posix()}\n"
                )
                os.utime(path, (FREEZE_TIME, FREEZE_TIME))
    manifest = proof_dir / "final_tree.sha256"
    _ = manifest.write_text("".join(sorted(entries)), encoding="utf-8")
    os.utime(manifest, (FREEZE_TIME, FREEZE_TIME))
    verification = proof_dir / "verification.json"
    write(
        verification,
        {
            "schema_version": "package-verification-v1",
            "applied_tree": str(inputs.applied),
            "final_tree_sha256": common.sha256(manifest),
            "metrics_sha256": common.sha256(inputs.metrics),
            "frozen_at": "2026-01-01T00:00:00+00:00",
            "validators": "PASS",
            "blind_audit_defect_free_rate": 1.0,
            "unexplained_family_mismatches": 0,
            "gfull_evaluations": 1,
            "evaluations": [
                {
                    "started_at": "2026-01-02T00:00:00+00:00",
                    "completed_at": "2026-01-02T01:00:00+00:00",
                    "global": dict(VALUES),
                    "per_source": {
                        source: dict(VALUES) for source in ("original01", "original02")
                    },
                }
            ],
        },
    )
    return replace(inputs, verification=verification)
