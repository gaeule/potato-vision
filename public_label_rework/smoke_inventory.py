"""Deterministic CPU fixtures; isolated outputs are cleaned by TemporaryDirectory."""

import errno
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from public_label_rework.common import OUTPUTS
from public_label_rework.families import (
    Feature,
    appearance,
    components,
    distances,
    family_id,
    picture_geometry,
    strict_groups,
    variant_hashes,
)
from public_label_rework.inventory import (
    boxes,
    inspect,
    overlap,
    run,
)
from public_label_rework.inventory_reports import sheet
from public_label_rework.labelio import LabelDocument, Xyxy, from_xyxy, parse
from retraining_data.schema import DataError, require


def fixture(rects: tuple[Xyxy, ...]) -> LabelDocument:
    """One parsed label fixture with arbitrary valid boxes and no disk dependency."""
    return parse(
        json.dumps(
            {
                "version": "4.1.1",
                "flags": {},
                "checked": False,
                "shapes": [
                    {
                        "label": "potato",
                        "points": [[a, b], [c, b], [c, d], [a, d]],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {},
                    }
                    for a, b, c, d in rects
                ],
                "imagePath": "fixture.jpg",
                "imageData": None,
                "imageHeight": 64,
                "imageWidth": 64,
                "description": "",
            }
        ).encode()
    )


def test_overlap() -> None:
    # Given: two 3x2 boxes offset by one pixel.
    a, b = (0.0, 0.0, 3.0, 2.0), (1.0, 0.0, 4.0, 2.0)
    # When: measure the boundary overlap.
    iou, contained = overlap(a, b)
    # Then: exact IoU=.5, not intersection/larger-area.
    require(iou == 0.5 and abs(contained - 2 / 3) < 1e-12, "IoU boundary math")
    # Given: a 5x5 rectangle with 80% inside a 10x10 rectangle.
    # When: measure containment.
    _, containment = overlap((0.0, 0.0, 10.0, 10.0), (-1.0, 0.0, 4.0, 5.0))
    # Then: denominator is the smaller area.
    require(containment == 0.8, "Containment boundary math")
    # Given/When: disjoint and touching rectangles.
    for box in ((4.0, 4.0, 5.0, 5.0), (3.0, 0.0, 4.0, 2.0)):
        require(overlap(a, box) == (0.0, 0.0), "Disjoint/touching overlap")


def test_geometry() -> None:
    # Given: tiny, huge, OOB and contained boxes in a known image.
    doc = fixture(((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 64.0, 64.0), (-1.0, 2.0, 5.0, 7.0)))
    # When: inspect without changing the parsed source.
    image = Image.new("RGB", (64, 64), "white")
    row = inspect("public_original01_0001", doc, image)
    # Then: flags and unique image-level overlap match the real formula.
    require(row.tiny_boxes == (0,) and row.huge_boxes == (1,), "Area flags")
    require(row.out_of_bounds_boxes == (2,), "OOB flag")
    require(len(row.overlap_candidates) == 2, "Pair overlap count")
    require(row.box_count_band == "2-5" and not row.dense, "Stratification schema")
    # Given: degenerate or non-axis-aligned geometry.
    for points in (((0.0, 0.0),) * 4, ((0.0, 0.0), (5.0, 1.0), (5.0, 5.0), (0.0, 5.0))):
        malformed = replace(doc, shapes=(replace(doc.shapes[0], points=points),))
        # When/Then: reject instead of claiming success with missing geometry.
        try:
            _ = boxes(malformed)
        except DataError:
            continue
        raise DataError("Malformed rectangle accepted")
    # Given/When/Then: mismatched real image dimensions fail.
    try:
        _ = inspect("public_original01_0001", doc, Image.new("RGB", (32, 32)))
    except DataError as error:
        require(error.detail == "Image/label size mismatch", "Wrong mismatch failure")
    else:
        raise DataError("Dimension mismatch accepted")
    require(
        from_xyxy(doc.shapes[0], (1.0, 2.0, 7.0, 9.0)).points[2] == (7.0, 9.0),
        "Conversion",
    )


def test_families() -> None:
    # Given: an asymmetric, textured image with exact D4 transforms.
    rgb = np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    image = Image.fromarray(rgb)
    transforms = [image.rotate(k * 90) for k in range(4)]
    transforms += [ImageOps.mirror(item) for item in transforms]
    features = [
        Feature(str(i), "fixture", 1, variant_hashes(im), appearance(im, ()))
        for i, im in enumerate(transforms)
    ]
    # When: compare all transforms using the minimum dihedral distance.
    distance = distances(features)
    # Then: all true transform pairs have identical hashes.
    require(bool(np.all(distance[~np.eye(8, dtype=bool)] == 0)), "D4 transform hashes")
    require(max(map(len, strict_groups(features, distance, 1))) <= 6, "Strict cap")
    # Given: equal hashes but 1 versus 5 boxes.
    incompatible = [features[0], replace(features[1], box_count=5)]
    # When: compute strict and loose groups.
    d = distances(incompatible)
    # Then: neither regime may join the incompatible scenes.
    require(len(components(d, 20)) == 2, "Loose box-count guard")
    require(len(strict_groups(incompatible, d, 12)) == 2, "Strict box-count guard")
    # Given: 13 identical scenes with compatible box counts.
    crowded = [replace(features[0], image_id=f"fixture-{i:02d}") for i in range(13)]
    # When: split a raw component larger than six.
    split = strict_groups(crowded, distances(crowded), 1)
    # Then: stable complete assignment with deterministic cap, no scene removal.
    require([len(g) for g in split] == [6, 6, 1], "Deterministic oversized split")
    require(sorted(i for g in split for i in g) == list(range(13)), "Family coverage")
    require(
        family_id("strict", ["a", "b"]) != family_id("strict", ["a", "c"]),
        "Distinct memberships",
    )
    # Given: a hash collision with a very different RGB appearance.
    different = [
        features[0],
        replace(features[1], appearance=np.zeros((8, 32, 32, 3), dtype=np.uint8)),
    ]
    # When/Then: the silhouette collision is vetoed for strict consistency.
    require(
        len(strict_groups(different, distances(different), 12)) == 2,
        "RGB collision veto",
    )


def test_pixels() -> None:
    # Given: exact rectangular bars and a separately rotated bright square.
    image = Image.new("RGB", (128, 128), "black")
    ImageDraw.Draw(image).rectangle((0, 16, 127, 111), fill="white")
    # When: measure rows and columns.
    rows, cols, angle = picture_geometry(image)
    # Then: bars are exact, an axis-aligned bar is not a rotation wedge.
    require(rows == tuple(range(16)) + tuple(range(112, 128)), "Letterbox rows")
    require(not cols and angle is None, "Bar/wedge separation")
    wedge = Image.new("RGB", (256, 256), "white").rotate(6, fillcolor="black")
    # When: estimate angle from rotated edges.
    _, _, estimated = picture_geometry(wedge)
    # Then: the known rotation is recovered within one degree.
    require(
        estimated is not None and abs(abs(estimated) - 6) < 1, "Wedge angle estimate"
    )


def main() -> None:
    test_overlap()
    test_geometry()
    test_families()
    test_pixels()
    # Given: only temporary, namespaced output ownership.
    with tempfile.TemporaryDirectory(
        prefix="smoke-inventory-", dir=OUTPUTS
    ) as directory:
        root = Path(directory)
        # When: write a real PIL sheet.
        sheet([("fixture", Image.new("RGB", (64, 64), "white"))], root / "sheet.jpg")
        # Then: it is an actual nonblank image, not just a success log.
        with Image.open(root / "sheet.jpg") as image:
            require(
                image.size == (1440, 510) and image.getbbox() is not None,
                "Sheet surface",
            )
        # Given/When: an already existing output is supplied to the real CLI function.
        try:
            run(root)
        except FileExistsError as error:
            require(error.errno == errno.EEXIST, "Wrong stale output failure reason")
        else:
            raise DataError("Stale output accepted")
    print(
        "PASS inventory fixtures: overlap geometry D4 count-guard cap6 RGB-veto bars wedge stale-output"
    )


if __name__ == "__main__":
    main()
