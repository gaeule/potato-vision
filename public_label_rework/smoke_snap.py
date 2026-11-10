"""CPU geometry fixtures and a separate, new-output-only GPU acceptance check."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from public_label_rework.snap import (
    ROOT,
    Box,
    SnapError,
    SnapResult,
    area,
    auto_masks,
    clip,
    content_box,
    digest,
    mask_result,
    prompt,
    snap,
    snap_batch,
    snap_point,
)
from public_label_rework.snap_prompts import Guidance, refine, refinement_result


class Shape(TypedDict):
    """Parse the four-point rectangle boundary, including rotated point order."""

    shape_type: str
    points: list[tuple[float, float]]


class Label(TypedDict):
    """Only machine-consumed rectangle fields enter the check."""

    shapes: list[Shape]


@dataclass(frozen=True, slots=True)
class Observation:
    """One actual return value, not a parsed success log."""

    image: str
    index: int
    input: Box
    result: SnapResult


def main() -> None:
    """Run deterministic CPU-only Given/When/Then geometry fixtures."""
    # Given a letterboxed image and a box entirely inside a black bar.
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[10:90, 20:100] = 180
    # When snapping an empty prompt, GPU/model construction must not be reached.
    empty = snap(image, (1, 1, 10, 5))
    # Then the input is clipped and rejected with a reason.
    assert not empty.tight and empty.reason and area(empty.xyxy) == 0
    assert content_box(image) == (20, 10, 100, 90)

    # Given a degenerate box on real image content.
    # When snapping it.
    degenerate = snap(image, (50, 50, 50, 60))
    # Then zero-area geometry is rejected before inference.
    assert not degenerate.tight and degenerate.reason
    assert not snap(image, (80, 50, 60, 70)).tight
    assert not snap(image, (0, 0, float("inf"), 70)).tight

    # Given a valid mask that expands left and right beyond its input.
    original: Box = (35, 30, 75, 70)
    padded = prompt(original, content_box(image))
    mask = np.zeros((100, 120), dtype=np.bool_)
    mask[31:69, 33:77] = True
    mask[15, 25] = True
    # When snapping, the isolated speck must not enlarge the largest component.
    result = mask_result(mask, original, padded)
    # Then expansion within the padded prompt is accepted.
    assert result.tight and result.xyxy == (33, 31, 77, 69)
    assert result.mask_area_ratio == 1672 / 1600

    # Given a larger raw component outside the prompt plus a smaller one inside.
    mask[:20, :] = True
    # When restricting the largest raw component to the prompt.
    outside = mask_result(mask, original, padded)
    # Then cropping must not promote the smaller component to a tight result.
    assert not outside.tight and outside.reason == "empty_mask_or_prompt"

    # Given a mask filling all four padded borders.
    mask[:] = False
    mask[26:74, 31:79] = True
    # When applying the guards.
    border = mask_result(mask, original, padded)
    # Then it is rejected and the original is retained.
    assert not border.tight and border.xyxy == original
    assert border.reason == "padded_border_more_than_two_sides"

    # Given a sparse connected cross with an acceptable bbox but little area.
    mask[:] = False
    mask[49, 35:75] = True
    mask[30:70, 55] = True
    # When applying the guards.
    sparse = mask_result(mask, original, padded)
    # Then the independent area guard rejects it despite high bbox IoU.
    assert not sparse.tight and sparse.reason == "mask_area_below_0.3"

    # Given a small off-centre component.
    mask[:] = False
    mask[27:37, 32:42] = True
    # When applying the guards.
    shifted = mask_result(mask, original, padded)
    # Then the independent IoU guard rejects the component.
    assert not shifted.tight and shifted.reason == "iou_below_0.5"

    # Given empty masks, all-black content, and an out-of-content point.
    mask[:] = False
    # When the public geometry entry points see those inputs.
    missing = mask_result(mask, original, padded)
    black = snap(np.zeros_like(image), original)
    point = snap_point(image, (5, 5))
    # Then all reject without downloading or importing the GPU framework.
    assert not missing.tight and not black.tight and not point.tight
    assert auto_masks(image, (0, 0, 5, 5)) == ()
    assert clip((-20, -20, 200, 200), content_box(image)) == (20, 10, 100, 90)
    coordinate_guards()
    guided_guards(image)
    guided_mask_guards()
    print("CPU_SMOKE PASS: empty, degenerate, component, expansion, IoU, area, border")


def guided_guards(image: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> None:
    """Reject unusable guided prompts before model loading, without rebasing."""
    # Given letterboxed content and invalid foreground/background evidence.
    original: Box = (35, 30, 75, 70)
    cases = (
        (Guidance(original, (), ((80, 80),)), "positive_and_negative_points_required"),
        (Guidance(original, ((50, 50),), ()), "positive_and_negative_points_required"),
        (Guidance(original, ((5, 5),), ((80, 80),)), "point_outside_content"),
        (Guidance(original, ((25, 15),), ((80, 80),)), "positive_point_outside_prompt"),
        (Guidance(original, ((50, 50),), ((50, 50),)), "contradictory_point_labels"),
        (
            Guidance(original, ((50, 50),), ((float("nan"), 50),)),
            "point_outside_content",
        ),
        (Guidance((1, 1, 10, 5), ((5, 3),), ((80, 80),)), "empty_or_degenerate_prompt"),
    )
    for guidance, reason in cases:
        # When explicit guidance crosses the public image boundary.
        receipt = refine(image, guidance)
        # Then no inference occurs and fallback remains the clipped original.
        assert receipt.raw_mask is None and not receipt.result.tight
        assert receipt.result.reason == reason
        assert receipt.result.xyxy == clip(guidance.original, content_box(image))


def guided_mask_guards() -> None:
    """A passing geometry mask cannot override contradictory observed clicks."""
    # Given a real rectangle mask and foreground/background evidence variants.
    mask = np.zeros((100, 120), dtype=np.bool_)
    mask[31:69, 33:77] = True
    original: Box = (35, 30, 75, 70)
    for positive, negative, eligible in (
        ((50, 50), (80, 80), True),
        ((76, 70), (80, 80), False),
        ((50, 50), (60, 50), False),
    ):
        # When deriving the unchanged geometry receipt and click consistency.
        receipt = refinement_result(
            mask, Guidance(original, (positive,), (negative,)), (31, 26, 79, 74)
        )
        # Then contradictory clicks withhold a candidate without changing tight.
        assert receipt.result.tight and (receipt.candidate is not None) == eligible


def coordinate_guards() -> None:
    """Exercise exclusive endpoints, fractional prompts and non-square masks."""
    # Given a non-square raster and a four-edge rectangle in pixel-edge units.
    mask = np.zeros((60, 120), dtype=np.bool_)
    mask[10:50, 30:80] = True
    # When finding its bounding box with a disjoint padded prompt.
    exact = mask_result(mask, (30, 10, 80, 50), (25, 6, 85, 54))
    # Then max occupied pixel + 1 recovers the original exclusive rectangle.
    assert exact.tight and exact.xyxy == (30, 10, 80, 50)
    assert exact.mask_area_ratio == 1.0

    # Given fractional bounds inside the same raster.
    # When clipping to full pixel extents inside the continuous prompt.
    fractional = mask_result(mask, (30, 10, 80, 50), (29.2, 9.2, 79.8, 49.8))
    # Then only the lower and right prompt borders are touched, not three.
    assert fractional.tight and fractional.xyxy == (30, 10, 79, 49)
    assert fractional.mask_area_ratio == 1911 / 2000

    # Given a one-pixel rectangle on a non-square raster.
    mask[:] = False
    mask[20, 40] = True
    # When converting occupied pixels to exclusive xyxy.
    pixel = mask_result(mask, (40, 20, 41, 21), (39, 19, 42, 22))
    # Then area and IoU stay one rather than collapsing to zero.
    assert pixel.tight and pixel.xyxy == (40, 20, 41, 21)
    assert pixel.mask_area_ratio == 1.0


def check() -> None:
    """Run first twenty original06 images and the fixed shifted-box example."""
    output = ROOT / "outputs/snap_check"
    if output.resolve() != output or (ROOT / "outputs").is_symlink():
        raise SnapError("noncanonical_output_refused")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows: list[Observation] = []
    tiles: list[Image.Image] = []
    expansion = False
    point_checks: list[bool] = []
    auto_count = 0
    with ZipFile(ROOT.parent / "final-dataset.zip") as archive:
        names = sorted(
            n
            for n in archive.namelist()
            if n.startswith("images/public/public_original06_") and n.endswith(".json")
        )[:20]
        names.append("images/public/public_original02_0177.json")
        for name in names:
            label: Label = json.loads(archive.read(name))
            with Image.open(BytesIO(archive.read(name[:-5] + ".jpg"))) as decoded:
                rgb = decoded.convert("RGB")
            image = np.asarray(rgb)[:, :, ::-1].copy()
            boxes: list[Box] = []
            for shape in label["shapes"]:
                if shape["shape_type"] != "rectangle" or len(shape["points"]) != 4:
                    raise SnapError("four_point_rectangle_required")
                xs, ys = zip(*shape["points"], strict=True)
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            results = snap_batch(image, boxes)
            print(
                f"IMAGE {Path(name).stem}: {sum(r.tight for r in results)}/{len(results)}"
            )
            for index, (box, result) in enumerate(zip(boxes, results, strict=True)):
                rows.append(Observation(Path(name).stem, index, box, result))
                extended = result.tight and (
                    result.xyxy[0] < box[0]
                    or result.xyxy[1] < box[1]
                    or result.xyxy[2] > box[2]
                    or result.xyxy[3] > box[3]
                )
                if "original02_0177" in name:
                    expansion |= extended
                if len(tiles) < 11 or ("original02_0177" in name and index == 1):
                    draw_image = rgb.copy()
                    draw = ImageDraw.Draw(draw_image)
                    draw.rectangle(box, outline="red", width=3)
                    draw.rectangle(result.xyxy, outline="lime", width=2)
                    draw.text(
                        (8, 8),
                        f"{Path(name).stem} #{index} {result.reason}",
                        fill="white",
                        stroke_width=1,
                        stroke_fill="black",
                    )
                    tiles.append(draw_image.resize((384, 384)))
            if len(point_checks) < 1:
                box = boxes[0]
                point_result = snap_point(
                    image, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                )
                point_checks.append(point_result.tight)
                # Bound automatic generation to a representative prompted region.
                auto_count = len(auto_masks(image, prompt(box, content_box(image))))
    empty = snap(np.zeros((64, 64, 3), dtype=np.uint8), (5, 5, 25, 25))
    measured = [row for row in rows if "original06" in row.image]
    ratios = np.asarray([row.result.mask_area_ratio for row in measured])
    rate = sum(row.result.tight for row in measured) / len(measured)
    stats = {
        "selection": "lexicographic_first_20_original06",
        "images": 20,
        "boxes": len(measured),
        "tight_rate": rate,
        "mask_area_ratio_quantiles": np.quantile(
            ratios, [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "mask_area_ratios": ratios.tolist(),
        "results": [asdict(row) for row in rows],
        "empty_prompt": asdict(empty),
        "shifted_example_expands": expansion,
        "point_checks": point_checks,
        "auto_mask_count": auto_count,
        "weights_sha256": digest(ROOT / "weights/sam2.1_b.pt"),
        "elapsed_seconds": time.monotonic() - started,
        "acceptance": "PASS"
        if rate >= 0.9 and not empty.tight and expansion
        else "FAIL",
    }
    _ = (output / "snap_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    sheet = Image.new("RGB", (384 * 4, 384 * 3))
    for index, tile in enumerate(tiles[:12]):
        sheet.paste(tile, ((index % 4) * 384, (index // 4) * 384))
    evidence = ROOT.parent / ".omo/evidence/public-label-rework"
    evidence.mkdir(parents=True, exist_ok=True)
    sheet.save(evidence / "snap_examples.jpg", quality=95)
    print(
        json.dumps(
            {
                key: value
                for key, value in stats.items()
                if key not in {"results", "mask_area_ratios"}
            },
            indent=2,
        )
    )
    if stats["acceptance"] != "PASS":
        raise SnapError("acceptance_failed_see_snap_stats")


if __name__ == "__main__":
    main()
