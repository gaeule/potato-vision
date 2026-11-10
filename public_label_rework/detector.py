"""Offline YOLO11s fold training and original-frame OOF proposals, never acceptance."""

import os
import shutil
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal, assert_never

from PIL import Image

from public_label_rework.common import ROOT, sha256, write_json
from public_label_rework.detector_data import (
    DEST,
    SOURCE,
    Detection,
    Oof,
    inventory,
    load_fold,
    oof_json,
    original_box,
    prepare,
    private_dataset,
    views,
)
from retraining_data.schema import (
    DataError,
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
from retraining_experiment.common import runtime
from scripts.run_baseline import environment

FONT: Final = Path("/home/sc/.config/Ultralytics/Arial.ttf")


def read_oof(data: bytes) -> Oof:
    """Parse the serialized proposal boundary before joining/auditing."""
    row = document(data)
    predictions: list[Detection] = []
    for value in array(row["predictions"]):
        box = mapping(value)
        coords = tuple(number(n) for n in array(box["xyxy"]))
        crop = tuple(integer(n) for n in array(box["crop_xyxy"]))
        require(len(coords) == len(crop) == 4, "Expected four coordinates")
        predictions.append(
            Detection(
                (coords[0], coords[1], coords[2], coords[3]),
                number(box["confidence"]),
                text(box["view"]),
                (crop[0], crop[1], crop[2], crop[3]),
                text(box["label"]),
                boolean(box["tta"]),
            )
        )
    return Oof(
        text(row["image_id"]),
        integer(row["fold"]),
        text(row["family"]),
        text(row["model_sha256"]),
        integer(row["width"]),
        integer(row["height"]),
        tuple(predictions),
    )


def network_guard(event: str, _args: tuple[Json, ...]) -> None:
    """Fail closed before any outbound socket/DNS operation, including imports."""
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        require(False, f"NETWORK_BLOCKED {event}")


@contextmanager
def offline(output: Path) -> Generator[None]:
    require(FONT.is_file(), f"Required local font missing: {FONT}")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    sys.addaudithook(network_guard)
    with runtime(output):
        font = Path(os.environ["YOLO_CONFIG_DIR"]) / "Arial.ttf"
        _ = shutil.copyfile(FONT, font)
        require(sha256(font) == sha256(FONT), "Font copy changed")
        print("OFFLINE_GUARD_ACTIVE local_font=" + sha256(font), flush=True)
        yield
    print("OFFLINE_GUARD_PASS", flush=True)


def train(index: int) -> None:
    fold = load_fold(index)
    folder = DEST / f"fold-{index}"
    require(not (folder / "train").exists(), "Refuse existing training output")
    weights = ROOT / "yolo11s.pt"
    before = sha256(weights)
    with offline(folder):
        import torch
        from ultralytics import YOLO

        require(torch.cuda.is_available(), "CUDA device 0 required")
        dataset = private_dataset(fold, Path(os.environ["YOLO_CONFIG_DIR"]).parent)
        model = YOLO(str(weights))
        require(len(model.names) == 80, "Expected COCO YOLO11s checkpoint")
        write_json(folder / "environment.json", environment())
        preset = document((ROOT / "baseline/config/train.json").read_bytes())
        require(text(preset.pop("model")) == "yolo11s.pt", "Baseline model drift")
        preset.update({"amp": False, "degrees": 0, "shear": 0, "perspective": 0})
        write_json(folder / "requested_args.json", preset)
        _ = model.train(
            **preset,
            data=str(dataset),
            project=str(folder),
            name="train",
            exist_ok=False,
        )
    write_json(
        folder / "checkpoints.json",
        {
            "best": sha256(folder / "train/weights/best.pt"),
            "last": sha256(folder / "train/weights/last.pt"),
            "original": before,
            "manifest_sha256": sha256(folder / "manifest.json"),
        },
    )
    print(f"TRAIN_PASS fold={index}", flush=True)


def predict(index: int) -> None:
    fold = load_fold(index)
    folder = DEST / f"fold-{index}"
    output = folder / "oof.jsonl"
    weight = folder / "train/weights/best.pt"
    digest = sha256(weight)
    completed = (
        tuple(read_oof(line) for line in output.read_bytes().splitlines())
        if output.exists()
        else ()
    )
    require(
        tuple(r.image_id for r in completed) == fold.predict[: len(completed)]
        and all(
            r.model_sha256 == digest
            and r.fold == index
            and r.family == fold.family_by_image[r.image_id]
            for r in completed
        )
        and len(completed) < len(fold.predict),
        "Invalid or completed OOF prefix",
    )
    rows = {r.image_id: r for r in inventory()}
    with offline(folder):
        from torch import Tensor
        from ultralytics import YOLO
        from ultralytics.engine.results import Results

        model = YOLO(str(weight))
        require(model.names == {0: "potato"}, "Wrong detector class")
        with output.open("a" if completed else "x", encoding="utf-8") as stream:
            for image_id in fold.predict[len(completed) :]:
                row = rows[image_id]
                detections: list[Detection] = []
                with Image.open(SOURCE / (image_id + ".jpg")) as source:
                    image = source.convert("RGB")
                for n, crop in enumerate(
                    views(image.size, row.dense or row.box_count >= 11)
                ):
                    result = next(
                        iter(
                            model.predict(
                                image.crop(crop),
                                conf=0.05,
                                augment=True,
                                imgsz=640,
                                device=0,
                                save=False,
                                verbose=False,
                                max_det=300,
                            )
                        )
                    )
                    match result:
                        case Results():
                            boxes = result.boxes
                            if boxes is None:
                                raise DataError("Detection result lacks boxes")
                        case Tensor():
                            raise DataError("Expected detection Results, not Tensor")
                        case unreachable:
                            assert_never(unreachable)
                    for box in boxes:
                        a, b, c, d = map(float, box.xyxy[0].tolist())
                        xyxy = original_box((a, b, c, d), crop)
                        if xyxy is None:
                            print(
                                "DEGENERATE_MODEL_BOX",
                                {"image_id": image_id, "view": n, "xyxy": [a, b, c, d]},
                                flush=True,
                            )
                            continue
                        detection = Detection(
                            xyxy=xyxy,
                            confidence=float(box.conf[0]),
                            view="full" if n == 0 else f"tile-{n}",
                            crop_xyxy=crop,
                        )
                        detections.append(detection)
                oof = Oof(
                    image_id=image_id,
                    fold=index,
                    family=fold.family_by_image[image_id],
                    model_sha256=digest,
                    width=row.width,
                    height=row.height,
                    predictions=tuple(detections),
                )
                _ = stream.write(oof_json(oof) + "\n")
                stream.flush()
    print(f"PREDICT_PASS fold={index} rows={len(fold.predict)}", flush=True)


def finalize() -> None:
    """Join complete folds; the actual-file audit verifies source/model provenance."""
    records: list[Oof] = []
    for index in range(5):
        fold = load_fold(index)
        folder = DEST / f"fold-{index}"
        predicted = [
            read_oof(line) for line in (folder / "oof.jsonl").read_bytes().splitlines()
        ]
        require(tuple(r.image_id for r in predicted) == fold.predict, "Fold coverage")
        records.extend(predicted)
    require(len(records) == len({r.image_id for r in records}) == 2060, "OOF coverage")
    require(
        {r.image_id for r in records} == {r.image_id for r in inventory()},
        "OOF inventory coverage differs",
    )
    with (DEST / "oof.jsonl").open("x", encoding="utf-8") as stream:
        for row in sorted(records, key=lambda r: r.image_id):
            _ = stream.write(oof_json(row) + "\n")
    print("FINALIZE_PASS rows=2060 weights=5", flush=True)


def main() -> None:
    """Explicit unit commands allow an external flock to release between units."""
    commands: dict[str, Literal["prepare", "train", "predict", "finalize"]] = {
        "prepare": "prepare",
        "train": "train",
        "predict": "predict",
        "finalize": "finalize",
    }
    require(len(sys.argv) >= 2 and sys.argv[1] in commands, "Expected detector command")
    match commands[sys.argv[1]]:
        case "prepare":
            prepare()
        case "train":
            train(int(sys.argv[2]))
        case "predict":
            predict(int(sys.argv[2]))
        case "finalize":
            finalize()
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    main()
