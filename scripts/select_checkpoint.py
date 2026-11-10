"""Select the deployed checkpoint using ONLY validation F1 at fixed confidence."""

import shutil
from pathlib import Path
from typing import TypedDict

from .baseline_common import BASE, PipelineError, output_directory, sha256, write_json

type Box = tuple[float, float, float, float]


class Counts(TypedDict):
    """One-to-one detection matching at IoU 0.5 and confidence 0.25."""

    images: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def read_boxes(path: Path, predictions: bool) -> list[tuple[float, Box]]:
    """Read validated class-zero YOLO rows; missing predictions mean no detections."""
    if predictions and not path.exists():
        return []
    rows: list[tuple[float, Box]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(part) for part in line.split()]
        if len(values) != (6 if predictions else 5) or values[0] != 0:
            raise PipelineError(f"Unexpected YOLO row in {path}: {line}")
        _, cx, cy, width, height = values[:5]
        score = values[5] if predictions else 1.0
        rows.append((score, (cx - width / 2, cy - height / 2,
                             cx + width / 2, cy + height / 2)))
    return sorted(rows, key=lambda row: row[0], reverse=True)


def iou(first: Box, second: Box) -> float:
    """Compute intersection over union in normalized image coordinates."""
    area = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    area *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    union = (first[2] - first[0]) * (first[3] - first[1])
    union += (second[2] - second[0]) * (second[3] - second[1]) - area
    return area / union


def score_predictions(labels: Path, predictions: Path) -> Counts:
    """Match confidence-ranked predictions to unused ground truth boxes once."""
    tp = fp = fn = 0
    images = sorted(labels.glob("*.txt"))
    if not images:
        raise PipelineError(f"No ground truth labels in {labels}")
    for label in images:
        truth = [box for _, box in read_boxes(label, False)]
        for confidence, box in read_boxes(predictions / label.name, True):
            if confidence < 0.25:
                continue
            overlaps = [iou(box, target) for target in truth]
            if overlaps and max(overlaps) >= 0.5:
                tp += 1
                truth.pop(overlaps.index(max(overlaps)))
            else:
                fp += 1
        fn += len(truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"images": len(images), "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def select() -> None:
    """Compare native best and last checkpoints on validation, never holdouts."""
    from ultralytics import YOLO

    output = output_directory(BASE / "selection", regenerate=False)
    candidates: dict[str, Counts] = {}
    for candidate in ("best", "last"):
        checkpoint = BASE / f"runs/train/weights/{candidate}.pt"
        model = YOLO(str(checkpoint))
        model.predict(source=str(BASE / "internal/images/val"), imgsz=640,
                      conf=0.25, iou=0.7, device=0, save=True,
                      save_txt=True, save_conf=True, verbose=False,
                      project=str(output), name=candidate, exist_ok=True)
        candidates[candidate] = score_predictions(
            BASE / "internal/labels/val", output / candidate / "labels",
        )
    selected = max(candidates, key=lambda name: candidates[name]["f1"])
    source = BASE / f"runs/train/weights/{selected}.pt"
    target = BASE / "runs/train/weights/selected.pt"
    if target.exists():
        raise PipelineError(f"Refusing to overwrite selected checkpoint: {target}")
    shutil.copyfile(source, target)
    write_json(output / "selection.json", {
        "criterion": "validation_F1_at_conf_0.25_IoU_0.5",
        "nms_iou": 0.7, "selected_candidate": selected,
        "checkpoint_sha256": sha256(target), "candidates": candidates,
        "selection_data": "internal/val_only",
        "protocol_note": "Operational selection added after initial exploratory evaluation exposed excessive false positives. Only validation scores determine this selection; external results are exploratory, not a pristine final acceptance test.",
    })
    print(f"SELECTED {selected}: {candidates[selected]}")


if __name__ == "__main__":
    select()
