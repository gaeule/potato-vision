"""CPU detector contracts; repeatable private fixtures, no model/GPU substitute."""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

from PIL import Image, ImageDraw

from public_label_rework.common import new_output, sha256, write_json
from public_label_rework.detector import read_oof
from public_label_rework.detector_data import (
    DEST,
    SOURCE,
    Families,
    Fold,
    Inventory,
    Oof,
    assert_isolation,
    folds,
    inventory,
    load_fold,
    oof_json,
    original_box,
    private_dataset,
    stable,
    views,
    yolo_label,
)
from public_label_rework.labelio import dumps
from public_label_rework.smoke_inventory import fixture
from retraining_data.schema import DataError, require


def test_folds() -> None:
    # Given: 30 source images with two multi-image loose families.
    rows = tuple(
        Inventory(
            image_id=str(i),
            width=64,
            height=64,
            box_count=1,
            dense=False,
            rotation_wedge=i == 5,
            overlap_candidates=({"a": 0, "b": 1},) if i == 6 else (),
            out_of_bounds_boxes=(0,) if i == 7 else (),
        )
        for i in range(30)
    )
    groups = {
        "large": tuple(str(i) for i in range(8)),
        "pair": ("8", "9"),
        **{str(i): (str(i),) for i in range(10, 30)},
    }
    families = Families(
        loose=groups,
        image_to_loose={i: g for g, members in groups.items() for i in members},
    )
    # When: run grouping on both input orders.
    result = folds(rows, families)
    repeated = folds(tuple(reversed(rows)), families)
    # Then: deterministic, complete, balanced without splitting the largest group.
    require(result == repeated and len(result) == 5, "Fold nondeterminism")
    require(
        sorted(int(i) for fold in result for i in fold.predict) == list(range(30)),
        "Missing prediction image",
    )
    require(
        sum(set(groups["large"]) <= set(f.predict) for f in result) == 1,
        "Largest group split",
    )
    require(
        all(not {"5", "6", "7"} & (set(f.train) | set(f.validation)) for f in result),
        "Unclean source entered model fitting",
    )
    # Given: a training member deliberately injected from the predicting family.
    fold = result[0]
    leaked = replace(
        fold, train=(*fold.train, fold.predict[0]), predict=fold.predict[1:]
    )
    # When/Then: real isolation gate rejects it.
    try:
        assert_isolation(leaked)
    except DataError as error:
        pass_rejected = error.detail == "Family leakage"
    else:
        pass_rejected = False
    require(pass_rejected, "Injected family leak accepted")
    # Given: an inconsistent family reverse map.
    broken = replace(families, image_to_loose={})
    # When/Then: no fallback to a different fold count.
    try:
        _ = folds(rows, broken)
    except DataError:
        rejected = True
    else:
        rejected = False
    require(rejected, "Invalid group assignment accepted")
    # Given/When: one indivisible family makes four prediction pools empty.
    single = Families(
        loose={"all": tuple(str(i) for i in range(30))},
        image_to_loose={str(i): "all" for i in range(30)},
    )
    try:
        _ = folds(rows, single)
    except DataError:
        rejected_empty = True
    else:
        rejected_empty = False
    # Then: fail, never silently use fewer folds.
    require(rejected_empty, "Empty folds accepted")


def test_coordinates() -> None:
    # Given: an odd-size image; exact 2x2 tiles must cover each pixel once.
    crops = views((641, 479), True)
    # When: map a view-space box in the lower-right tile to the original.
    box = original_box((1.0, 2.0, 100.0, 80.0), crops[4])
    # Then: offset, not scaling or a swapped axis.
    require(box == (321.0, 241.0, 420.0, 319.0), "Tile coordinate mapping")
    require(
        sum((c - a) * (d - b) for a, b, c, d in crops[1:]) == 641 * 479,
        "Tile coverage",
    )
    require(views((641, 479), False) == crops[:1], "Unexpected regular-image tiles")
    # Given/When: a full-frame prediction.
    full = original_box((1.0, 2.0, 640.0, 478.0), crops[0])
    # Then: coordinates are unchanged.
    require(full == (1.0, 2.0, 640.0, 478.0), "Full-frame mapping")
    # Given: actual TTA output clipped to a zero-width box at the right edge.
    clipped = (640.0, 0.8784332275390625, 640.0, 275.3863525390625)
    # When/Then: the model boundary rejects the non-box, not the whole image.
    require(
        original_box(clipped, (0, 0, 640, 640)) is None, "Clipped zero-area model box"
    )


def test_output(directory: Path) -> None:
    # Given: an original JSON rectangle and a fresh private output directory.
    label = directory / "fixture.json"
    _ = label.write_bytes(dumps(fixture(((8.0, 16.0, 40.0, 48.0),))))
    # When: convert through production labelio->YOLO boundary.
    converted = yolo_label(label)
    # Then: independent known geometry gives the normalized center and extent.
    require(converted == "0 0.375 0.5 0.5 0.5\n", "Original JSON conversion")
    # Given: a real empty prediction row with its primary predicting model.
    empty = Oof(
        image_id="empty",
        fold=0,
        family="family",
        model_sha256="0" * 64,
        width=64,
        height=64,
        predictions=(),
    )
    # When: persist an empty row as JSONL and parse it back.
    output = directory / "oof.jsonl"
    _ = output.write_text(oof_json(empty) + "\n", encoding="utf-8")
    parsed = [read_oof(line) for line in output.read_bytes().splitlines()]
    # Then: the image is present even though no boxes exist.
    require(
        len(parsed) == 1
        and parsed[0].image_id == "empty"
        and parsed[0].predictions == (),
        "Empty row disappeared",
    )
    # Given: an existing private output and its digest.
    digest = sha256(output)
    # When: attempt a stale output reservation.
    try:
        _ = new_output(directory)
    except FileExistsError:
        rejected = True
    else:
        rejected = False
    # Then: refusal preserves bytes.
    require(rejected and sha256(output) == digest, "Stale output overwritten")


def test_private_dataset(directory: Path) -> None:
    # Given: original private bytes plus a cache belonging to a different process.
    source = new_output(directory / "input")
    for name in ("images", "labels"):
        (source / name).mkdir()
    for name in ("train", "val"):
        Image.new("RGB", (64, 64), "white").save(source / f"images/{name}.jpg")
        _ = (source / f"labels/{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    _ = (source / "labels.cache").write_bytes(b"foreign cache")
    fold = Fold(0, ("held",), ("train",), ("val",), {})
    # When: make process-specific YOLO data without any framework import.
    config = private_dataset(fold, directory, source)
    data = config.parent
    # Then: pixels/labels are identical; loader lists point into the new dataset.
    require(
        sha256(data / "images/train.jpg") == sha256(source / "images/train.jpg")
        and sha256(data / "labels/train.txt") == sha256(source / "labels/train.txt")
        and not (data / "labels.cache").exists(),
        "Process data copied cache or changed source bytes",
    )
    require(
        (data / "train.txt").read_text().splitlines()
        == [str(data / "images/train.jpg")]
        and (data / "val.txt").read_text().splitlines()
        == [str(data / "images/val.jpg")],
        "Process loader list points outside its own data",
    )


def main() -> None:
    test_folds()
    test_coordinates()
    # Given/When: repeated runs use distinct owned temporary directories.
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix="smoke-", dir=DEST) as name:
            test_output(Path(name))
            test_private_dataset(Path(name))
    # Then: TemporaryDirectory cleans its own fixtures.
    print("PASS detector CPU fixtures and isolated temporary outputs")


def render_qa() -> None:
    """Actual OOF proposal overlays; visual acceptance is recorded separately."""
    records = [
        read_oof(line) for line in (DEST / "oof.jsonl").read_bytes().splitlines()
    ]
    qa = new_output(DEST / "qa")
    by_id = {r.image_id: r for r in records}
    dense = [r.image_id for r in inventory() if r.dense][:3]
    regular = [r.image_id for r in inventory() if not r.dense]
    selected = dense + sorted(regular, key=stable)[:9]
    for image_id in selected:
        row = by_id[image_id]
        with Image.open(SOURCE / (image_id + ".jpg")) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in row.predictions:
            if box.confidence >= 0.25:
                draw.rectangle(
                    box.xyxy, outline="lime" if box.view == "full" else "cyan", width=2
                )
        image.save(qa / (image_id + ".jpg"))
    write_json(
        qa / "selection.json",
        {
            "images": selected,
            "dense": dense,
            "display_confidence": 0.25,
            "saved_confidence": 0.05,
            "purpose": "Detector proposal QA only, not label acceptance",
        },
    )


def audit_preparation() -> None:
    """Count actual private files and all clean train/inner-validation pools."""
    rows = inventory()
    clean = {r.image_id for r in rows if r.clean}
    split = tuple(load_fold(i) for i in range(5))
    for name, suffix in (("images", ".jpg"), ("labels", ".txt")):
        require(
            {p.stem for p in (DEST / "data" / name).glob("*" + suffix)} == clean,
            "Private dataset differs from clean pool",
        )
    counts = [
        {
            "index": f.index,
            "predict": len(f.predict),
            "train": len(f.train),
            "validation": len(f.validation),
            "clean_heldout": len(set(f.predict) & clean),
        }
        for f in split
    ]
    write_json(
        DEST / "counts.json",
        {
            "images": len(rows),
            "clean": len(clean),
            "folds": counts,
            "validation_policy": "Inner 10% whole-family holdout from other four folds; "
            "optimizer and checkpoint selection never access predicting families",
        },
    )
    print("DATA_COUNTS", len(rows), len(clean), counts)


if __name__ == "__main__":
    modes: dict[str, Literal["qa", "counts"]] = {"qa": "qa", "counts": "counts"}
    mode = modes.get(sys.argv[1]) if len(sys.argv) == 2 else None
    match mode:
        case "qa":
            render_qa()
        case "counts":
            audit_preparation()
        case None:
            main()
        case unreachable:
            assert_never(unreachable)
