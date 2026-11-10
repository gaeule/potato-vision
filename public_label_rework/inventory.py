"""Read-only inventory and PIL evidence sheets; never a label decision."""

import io
import json
import math
import random
import zipfile
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
from fnmatch import filter as matching
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

from public_label_rework import families as family_tools
from public_label_rework.common import OUTPUTS, ZIP, new_output, sha256, write_json
from public_label_rework.inventory_reports import sheet
from public_label_rework.labelio import LabelDocument, Xyxy, parse, to_xyxy
from retraining_data.schema import array, document, mapping, require

SOURCE: Final = OUTPUTS / "source"


@dataclass(frozen=True, slots=True)
class Overlap:
    a: int
    b: int
    iou: float
    containment: float


@dataclass(frozen=True, slots=True)
class InventoryRow:
    """Schema v1: zero-based source box indices; pixels, degrees and area ratios."""

    image_id: str
    source: str
    width: int
    height: int
    box_count: int
    box_count_band: str
    dense: bool
    letterbox_rows: tuple[int, ...]
    letterbox_cols: tuple[int, ...]
    rotation_wedge: bool
    estimated_angle: float | None
    out_of_bounds_boxes: tuple[int, ...]
    tiny_boxes: tuple[int, ...]
    huge_boxes: tuple[int, ...]
    overlap_candidates: tuple[Overlap, ...]


def overlap(a: Xyxy, b: Xyxy) -> tuple[float, float]:
    """IoU=intersection/union; containment=intersection/min(area_a,area_b)."""
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (a, b)]
    require(all(math.isfinite(v) and v > 0 for v in areas), "Invalid box area")
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return inter / (sum(areas) - inter), inter / min(areas)


def boxes(doc: LabelDocument) -> tuple[Xyxy, ...]:
    """Reject malformed rectangles instead of silently hiding geometry failures."""
    require(doc.imageWidth > 0 and doc.imageHeight > 0, "Invalid dimensions")
    result = tuple(to_xyxy(shape) for shape in doc.shapes)
    for shape, box in zip(doc.shapes, result, strict=True):
        x1, y1, x2, y2 = box
        require(
            x1 < x2 and y1 < y2 and math.isfinite((x2 - x1) * (y2 - y1)),
            "Invalid rectangle area",
        )
        require(
            set(shape.points) == {(x1, y1), (x2, y1), (x2, y2), (x1, y2)},
            "Non-axis-aligned rectangle",
        )
    return result


def inspect(image_id: str, doc: LabelDocument, image: Image.Image) -> InventoryRow:
    """Compute all inventory flags without modifying a document or its pixels."""
    require(
        image.size == (doc.imageWidth, doc.imageHeight), "Image/label size mismatch"
    )
    rects = boxes(doc)
    pairs: list[Overlap] = []
    for i, a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            iou, contained = overlap(a, rects[j])
            if iou >= 0.5 or contained >= 0.8:
                pairs.append(Overlap(i, j, iou, contained))
    width, height = image.size
    areas = [(b[2] - b[0]) * (b[3] - b[1]) / (width * height) for b in rects]
    rows, cols, angle = family_tools.picture_geometry(image)
    count = len(rects)
    return InventoryRow(
        image_id,
        image_id.split("_")[1],
        width,
        height,
        count,
        ("0", "1", "2-5", "6-10", "11+")[bisect_left((0, 1, 5, 10), count)],
        count >= 11,
        rows,
        cols,
        angle is not None,
        angle,
        tuple(
            i
            for i, b in enumerate(rects)
            if b[0] < 0 or b[1] < 0 or b[2] > width or b[3] > height
        ),
        tuple(i for i, area in enumerate(areas) if area < 0.001),
        tuple(i for i, area in enumerate(areas) if area > 0.5),
        tuple(pairs),
    )


def annotated(image: Image.Image, doc: LabelDocument) -> Image.Image:
    """Draw only evidence overlays on a new RGB canvas with stable PIL."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for i, box in enumerate(boxes(doc)):
        draw.rectangle(box, outline="#ff3030", width=max(2, image.width // 300))
        draw.text((box[0] + 2, box[1] + 2), str(i), fill="yellow")
    return canvas


def render_public(ids: list[str], path: Path) -> None:
    tiles: list[tuple[str, Image.Image]] = []
    for image_id in ids:
        label = SOURCE / f"images/public/{image_id}.json"
        with Image.open(label.with_suffix(".jpg")) as image:
            tiles.append((image_id, annotated(image, parse(label.read_bytes()))))
    sheet(tiles, path)


def harvest_report(output: Path) -> None:
    """Read all harvest labels and exactly 24 images in memory; save annotations only."""
    renders = output / "harvest_convention"
    renders.mkdir()
    with zipfile.ZipFile(ZIP) as archive:
        names = sorted(matching(archive.namelist(), "images/harvest/*.json"))
        picked = sorted(random.Random(42).sample(names, 24))
        areas: list[float] = []
        aspects: list[float] = []
        counts: list[int] = []
        tiles: list[tuple[str, Image.Image]] = []
        for name in names:
            raw = document(archive.read(name))
            _ = raw.setdefault("checked", False)
            _ = raw.setdefault("description", "")
            for shape in array(raw["shapes"]):
                _ = mapping(shape).setdefault("attributes", {})
            doc = parse(json.dumps(raw).encode())
            rects = boxes(doc)
            counts.append(len(rects))
            for x1, y1, x2, y2 in rects:
                areas.append((x2 - x1) * (y2 - y1) / (doc.imageWidth * doc.imageHeight))
                aspects.append((x2 - x1) / (y2 - y1))
            if name in picked:
                image_name = str(Path(name).parent / doc.imagePath)
                with Image.open(io.BytesIO(archive.read(image_name))) as image:
                    rendered = annotated(image, doc)
                rendered.save(renders / f"{Path(name).stem}.jpg", quality=94)
                tiles.append((Path(name).stem, rendered))
        for start in range(0, 24, 6):
            sheet(tiles[start : start + 6], renders / f"sheet_{start // 6 + 1}.jpg")
    quantiles = (0, 0.25, 0.5, 0.75, 1)
    write_json(
        output / "harvest_convention.json",
        {
            "schema_version": 1,
            "images": len(names),
            "boxes": sum(counts),
            "box_count_histogram": dict(Counter(counts)),
            "area_ratio_quantiles": np.quantile(areas, quantiles).tolist(),
            "aspect_ratio_quantiles": np.quantile(aspects, quantiles).tolist(),
            "sample_seed": 42,
            "sample": [Path(n).stem for n in picked],
            "sample_images_decoded": len(tiles),
            "raw_images_extracted": 0,
            "sprouts_verdict": None,
            "extent_verdict": None,
            "visual_status": "pending",
            "overlap_candidates_report_only": [
                "harvest_04_004",
                "harvest_05_020",
                "harvest_05_036",
            ],
        },
    )


def run(output: Path) -> None:
    """Full real-data surface, refusing a stale output before reading source images."""
    target = new_output(output)
    paths = sorted((SOURCE / "images/public").glob("*.json"))
    require(len(paths) == 2060, f"Expected 2060 source labels, got {len(paths)}")
    before = {
        str(p.relative_to(SOURCE)): sha256(p)
        for p in sorted(SOURCE.rglob("*"))
        if p.is_file()
    }
    zip_before = sha256(ZIP)
    rows: list[InventoryRow] = []
    features: list[family_tools.Feature] = []
    with (target / "inventory.jsonl").open("x", encoding="utf-8") as stream:
        for path in paths:
            doc = parse(path.read_bytes())
            with Image.open(path.with_suffix(".jpg")) as image:
                row = inspect(path.stem, doc, image)
                features.append(
                    family_tools.Feature(
                        path.stem,
                        row.source,
                        row.box_count,
                        family_tools.variant_hashes(image),
                        family_tools.appearance(image, boxes(doc)),
                    )
                )
            rows.append(row)
            _ = stream.write(json.dumps(asdict(row), allow_nan=False) + "\n")
    require(
        sum(bool(r.overlap_candidates) for r in rows) == 40,
        "Expected 40 overlap images",
    )
    require(sum(r.box_count for r in rows) == 9498, "Expected 9498 source boxes")
    families = family_tools.build_families(features)
    write_json(target / "families.json", asdict(families))
    for i, members in enumerate(families.examples):
        render_public(members, target / f"family_{i + 1}.jpg")
    render_public(
        [m for family in families.examples for m in family[:2]],
        target / "families_examples.jpg",
    )
    sanity_ids = random.Random(42).sample(
        [p.stem for p in paths if "original06" in p.stem], 12
    )
    render_public(sanity_ids, target / "conversion_sanity.jpg")
    source01 = [p.stem for p in paths if "original01" in p.stem]
    for start in range(0, len(source01), 12):
        render_public(
            source01[start : start + 12], target / f"source01_{start // 12 + 1}.jpg"
        )
    harvest_report(target)
    after = {
        str(p.relative_to(SOURCE)): sha256(p)
        for p in sorted(SOURCE.rglob("*"))
        if p.is_file()
    }
    require(before == after and zip_before == sha256(ZIP), "Source mutation")
    write_json(
        target / "summary.json",
        {
            "rows": len(rows),
            "boxes": sum(r.box_count for r in rows),
            "overlap_candidate_images": sum(bool(r.overlap_candidates) for r in rows),
            "source_files_hashed": len(before),
            "source_unchanged": True,
            "zip_sha256": zip_before,
        },
    )
    print(
        f"PASS inventory rows={len(rows)} overlap_images=40 source_unchanged={len(before)}"
    )


def main() -> None:
    run(OUTPUTS / "inventory")


if __name__ == "__main__":
    main()
