"""Detector-only family folds and private YOLO dataset; no label decisions."""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from public_label_rework.common import OUTPUTS, ROOT, new_output, sha256, write_json
from public_label_rework.labelio import Xyxy, parse, to_xyxy
from retraining_data.schema import (
    Json,
    array,
    boolean,
    document,
    integer,
    mapping,
    number,
    require,
    text,
)

DEST: Final = OUTPUTS / "detector"
SOURCE: Final = OUTPUTS / "source/images/public"


@dataclass(frozen=True, slots=True)
class Inventory:
    image_id: str
    width: int
    height: int
    box_count: int
    dense: bool
    rotation_wedge: bool
    overlap_candidates: tuple[dict[str, int | float], ...]
    out_of_bounds_boxes: tuple[int, ...]

    @property
    def clean(self) -> bool:
        return not (
            self.rotation_wedge or self.overlap_candidates or self.out_of_bounds_boxes
        )


@dataclass(frozen=True, slots=True)
class Families:
    image_to_loose: dict[str, str]
    loose: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    predict: tuple[str, ...]
    train: tuple[str, ...]
    validation: tuple[str, ...]
    family_by_image: dict[str, str]


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: Xyxy
    confidence: float
    view: str
    crop_xyxy: tuple[int, int, int, int]
    label: str = "potato"
    tta: bool = True


@dataclass(frozen=True, slots=True)
class Oof:
    image_id: str
    fold: int
    family: str
    model_sha256: str
    width: int
    height: int
    predictions: tuple[Detection, ...]


def inventory() -> tuple[Inventory, ...]:
    rows = [
        document(line)
        for line in (OUTPUTS / "inventory/inventory.jsonl").read_bytes().splitlines()
    ]
    return tuple(
        Inventory(
            text(r["image_id"]),
            integer(r["width"]),
            integer(r["height"]),
            integer(r["box_count"]),
            boolean(r["dense"]),
            boolean(r["rotation_wedge"]),
            tuple(
                {k: number(v) for k, v in mapping(p).items()}
                for p in array(r["overlap_candidates"])
            ),
            tuple(integer(i) for i in array(r["out_of_bounds_boxes"])),
        )
        for r in rows
    )


def strings(value: Json) -> tuple[str, ...]:
    return tuple(text(i) for i in array(value))


def oof_json(row: Oof) -> str:
    return json.dumps(asdict(row), allow_nan=False, separators=(",", ":"))


def stable(value: str) -> str:
    return hashlib.sha256(f"42:{value}".encode()).hexdigest()


def assert_isolation(fold: Fold) -> None:
    """Both optimizer and checkpoint selection must exclude predicting families."""
    pools = [set(fold.train), set(fold.validation), set(fold.predict)]
    require(all(pools), "Empty train/validation/prediction pool")
    families = [{fold.family_by_image[i] for i in pool} for pool in pools]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        require(not pools[a] & pools[b], "Image leakage")
        require(not families[a] & families[b], "Family leakage")


def folds(rows: tuple[Inventory, ...], groups: Families) -> tuple[Fold, ...]:
    """LPT balance whole loose components; inner group holdout selects best."""
    ids = {row.image_id for row in rows}
    require(len(ids) == len(rows), "Duplicate inventory ID")
    members = [i for group in groups.loose.values() for i in group]
    require(
        len(members) == len(ids)
        and set(members) == ids
        and set(groups.image_to_loose) == ids,
        "Incomplete or duplicate family assignment",
    )
    for family, images in groups.loose.items():
        require(bool(images), "Empty loose family")
        require(
            all(groups.image_to_loose[i] == family for i in images),
            "Forward/reverse family mismatch",
        )
    assigned: list[list[str]] = [[] for _ in range(5)]
    for family in sorted(
        groups.loose, key=lambda g: (-len(groups.loose[g]), stable(g))
    ):
        target = min(range(5), key=lambda n: (len(assigned[n]), n))
        assigned[target].extend(groups.loose[family])
    clean = {row.image_id for row in rows if row.clean}
    result: list[Fold] = []
    for index, held in enumerate(assigned):
        pool = clean - set(held)
        candidates = sorted({groups.image_to_loose[i] for i in pool}, key=stable)
        validation: set[str] = set()
        for family in candidates:
            if len(validation) >= max(1, round(len(pool) * 0.1)):
                break
            validation.update(set(groups.loose[family]) & pool)
        fold = Fold(
            index=index,
            predict=tuple(sorted(held)),
            train=tuple(sorted(pool - validation)),
            validation=tuple(sorted(validation)),
            family_by_image=groups.image_to_loose,
        )
        assert_isolation(fold)
        result.append(fold)
    return tuple(result)


def yolo_label(path: Path) -> str:
    """Use verified original JSON geometry, never corrected/gold labels."""
    doc = parse(path.read_bytes())
    lines: list[str] = []
    for shape in doc.shapes:
        x1, y1, x2, y2 = to_xyxy(shape)
        require(shape.label == "potato", "Unexpected source class")
        require(
            0 <= x1 < x2 <= doc.imageWidth and 0 <= y1 < y2 <= doc.imageHeight,
            "Invalid source rectangle",
        )
        values = (
            (x1 + x2) / (2 * doc.imageWidth),
            (y1 + y2) / (2 * doc.imageHeight),
            (x2 - x1) / doc.imageWidth,
            (y2 - y1) / doc.imageHeight,
        )
        lines.append("0 " + " ".join(f"{v:.12g}" for v in values))
    return "\n".join(lines) + ("\n" if lines else "")


def prepare(output: Path = DEST) -> None:
    rows = inventory()
    raw = document((OUTPUTS / "inventory/families.json").read_bytes())
    groups = Families(
        {i: text(g) for i, g in mapping(raw["image_to_loose"]).items()},
        {g: strings(v) for g, v in mapping(raw["loose"]).items()},
    )
    split = folds(rows, groups)
    require(len(rows) == 2060, "Expected all 2060 source images")
    require(
        {p.stem for p in SOURCE.glob("*.jpg")} == {r.image_id for r in rows},
        "Source/inventory join differs",
    )
    output = new_output(output)
    data = new_output(output / "data")
    for name in ("images", "labels"):
        (data / name).mkdir()
    hashes: dict[str, str] = {}
    for row in rows:
        for suffix in (".jpg", ".json"):
            path = SOURCE / (row.image_id + suffix)
            hashes[path.name] = sha256(path)
        if row.clean:
            target = data / "images" / (row.image_id + ".jpg")
            _ = shutil.copyfile(SOURCE / target.name, target)
            _ = (data / "labels" / (row.image_id + ".txt")).write_text(
                yolo_label(SOURCE / (row.image_id + ".json")), encoding="utf-8"
            )
    hashes["root_model"] = sha256(ROOT / "yolo11s.pt")
    write_json(output / "source_hashes.json", hashes)
    write_json(output / "folds.json", [asdict(f) for f in split])
    for fold in split:
        folder = new_output(output / f"fold-{fold.index}")
        write_json(folder / "manifest.json", asdict(fold))
        for name, images in (("train", fold.train), ("val", fold.validation)):
            _ = (folder / f"{name}.txt").write_text(
                "".join(f"{data}/images/{i}.jpg\n" for i in images), encoding="utf-8"
            )
        _ = (folder / "data.yaml").write_text(
            f"path: {data}\ntrain: {folder}/train.txt\nval: {folder}/val.txt\n"
            "names:\n  0: potato\n",
            encoding="utf-8",
        )
    print("PREPARE_PASS", output, flush=True)


def load_fold(index: int) -> Fold:
    require(0 <= index < 5, "Expected fold index 0..4")
    row = document((DEST / f"fold-{index}/manifest.json").read_bytes())
    fold = Fold(
        integer(row["index"]),
        strings(row["predict"]),
        strings(row["train"]),
        strings(row["validation"]),
        {i: text(g) for i, g in mapping(row["family_by_image"]).items()},
    )
    assert_isolation(fold)
    return fold


def private_dataset(fold: Fold, directory: Path, source: Path = DEST / "data") -> Path:
    """Give a training process exclusive dataset-cache paths, preserving bytes."""
    data = new_output(directory / "dataset")
    for name in ("images", "labels"):
        _ = shutil.copytree(source / name, data / name)
    for name, images in (("train", fold.train), ("val", fold.validation)):
        _ = (data / f"{name}.txt").write_text(
            "".join(f"{data}/images/{i}.jpg\n" for i in images), encoding="utf-8"
        )
    yaml = data / "data.yaml"
    _ = yaml.write_text(
        f"path: {data}\ntrain: {data}/train.txt\nval: {data}/val.txt\n"
        "names:\n  0: potato\n",
        encoding="utf-8",
    )
    return yaml


def views(size: tuple[int, int], tiled: bool) -> tuple[tuple[int, int, int, int], ...]:
    width, height = size
    full = ((0, 0, width, height),)
    if not tiled:
        return full
    return full + tuple(
        (x, y, right, bottom)
        for y, bottom in ((0, height // 2), (height // 2, height))
        for x, right in ((0, width // 2), (width // 2, width))
    )


def original_box(box: Xyxy, crop: tuple[int, int, int, int]) -> Xyxy | None:
    """Offset view pixels; reject zero-area model boxes left by border clipping."""
    x, y, right, bottom = crop
    a, b, c, d = box
    require(0 <= a <= c <= right - x and 0 <= b <= d <= bottom - y, "View bounds")
    return (a + x, b + y, c + x, d + y) if a < c and b < d else None
