"""Operational metrics and detailed detections, with no external label caches."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from stone_experiment.common import (
    BASE,
    CONFIDENCE,
    DATASET,
    MATCH_IOU,
    NMS_IOU,
    OUTPUTS,
    Options,
    PipelineError,
    environment,
    new_output,
    runtime,
    sha256,
    write_json,
)
from stone_experiment.metrics import (
    CONFUSION_ORDER,
    Audit,
    Detection,
    Summary,
    match_frame,
    summarize,
)
from stone_experiment.predict import check_classes, predict
from stone_experiment.prepare import verify_dataset
from stone_experiment.schema import Json, boxes

if TYPE_CHECKING:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results


@dataclass(frozen=True, slots=True)
class Example:
    """One fixed image and its pixel-coordinate truth."""

    image: Path
    truth: tuple[Detection, ...]


def internal_examples(dataset: Path, split: str) -> tuple[Example, ...]:
    """Read the frozen dataset snapshot and check its complete artifact hashes."""
    if split not in ("val", "test"):
        raise PipelineError("Operational evaluation accepts val or test only")
    frames = verify_dataset(dataset)
    return tuple(
        Example(dataset / "images" / split / frame.filename,
                tuple(Detection(class_id, box) for class_id, group in enumerate((frame.boxes, frame.stones))
                      for box in group))
        for frame in frames if frame.split == split
    )


def external_examples() -> tuple[Example, ...]:
    """Read external potato-only labels directly, never invoking YOLO val/cache."""
    images = sorted((BASE / "external/images/test").glob("*.png"))
    if not images:
        raise PipelineError("No prepared external potato test PNGs found")
    examples: list[Example] = []
    for image in images:
        with Image.open(image) as decoded:
            width, height = decoded.size
        truth: list[Detection] = []
        label = BASE / "external/labels/test" / image.with_suffix(".txt").name
        for line in label.read_text(encoding="utf-8").splitlines():
            values = [float(part) for part in line.split()]
            if len(values) != 5 or values[0] != 0:
                raise PipelineError(f"Expected external potato-only YOLO label: {label}")
            _, cx, cy, w, h = values
            raw = ((cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height)
            tolerance = max(width, height) * 1e-9
            if (raw[0] < -tolerance or raw[1] < -tolerance or
                    raw[2] > width + tolerance or raw[3] > height + tolerance):
                raise PipelineError(f"External box outside image: {label}")
            # Baseline YOLO labels were rounded to nine normalized decimals.
            # Recover exact image edges without accepting materially invalid GT.
            pixel: list[Json] = [max(0.0, raw[0]), max(0.0, raw[1]),
                                 min(float(width), raw[2]), min(float(height), raw[3])]
            truth.append(Detection(0, boxes([pixel], width, height)[0]))
        examples.append(Example(image, tuple(truth)))
    return tuple(examples)


def detections(result: "Results", *, potato_only: bool) -> tuple[Detection, ...]:
    """Convert library arrays into typed, finite pixel-space detections."""
    if result.boxes is None:
        raise PipelineError("Detection result lacks boxes")
    output: list[Detection] = []
    for index in range(len(result.boxes)):
        class_id = int(result.boxes.cls[index].item())
        if potato_only and class_id != 0:
            continue
        coordinates = tuple(float(result.boxes.xyxy[index, j].item()) for j in range(4))
        x1, y1, x2, y2 = coordinates
        output.append(Detection(class_id, (x1, y1, x2, y2), float(result.boxes.conf[index].item())))
    return tuple(output)


def score(model: "YOLO", examples: tuple[Example, ...], output: Path, *, potato_only: bool = False) -> Summary:
    """Persist every input, detection, and one-to-one match for visual audit."""
    audits: list[Audit] = []
    output.mkdir(parents=True, exist_ok=False)
    for example in examples:
        result = predict(model, str(example.image), CONFIDENCE)
        audit = match_frame(example.truth, detections(result, potato_only=potato_only))
        audits.append(audit)
        write_json(output / f"{example.image.stem}.json", {
            "image": str(example.image), "image_sha256": sha256(example.image),
            **asdict(audit),
        })
    summary = summarize(audits, potato_only=potato_only)
    write_json(output / "summary.json", summary)
    return summary


def evaluate(weights: Path, dataset: Path, split: str, output: Path) -> None:
    """Evaluate one checkpoint at the fixed operational threshold."""
    if not weights.is_file():
        raise PipelineError(f"Existing checkpoint required: {weights}")
    external = split == "external"
    examples = external_examples() if external else internal_examples(dataset, split)
    output = new_output(output)
    runtime()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    check_classes(model)
    summary = score(model, examples, output / "frames", potato_only=external)
    write_json(output / "evaluation.json", {
        "split": split, "weights": str(weights.resolve()), "weights_sha256": sha256(weights),
        "confidence": CONFIDENCE, "nms_iou": NMS_IOU, "matching_iou": MATCH_IOU,
        "matching": "confidence_ranked_class_agnostic_one_to_one",
        "confusion_order": CONFUSION_ORDER, "count_unit": "frame_observations_not_unique_objects",
        "annotation_snapshot_sha256": None if external else sha256(dataset / "annotations.json"),
        "dataset_manifest_sha256": sha256((BASE / "external" if external else dataset) / "manifest.json"),
        "external_policy": "class0_predictions_only; stone_precision_recall_not_scored" if external else None,
        "environment": environment(), "summary": summary,
    })


def main() -> None:
    """Parse evaluation options; external data is never used for selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--weights", type=Path, default=OUTPUTS / "selection/selected.pt")
    _ = parser.add_argument("--dataset", type=Path, default=DATASET)
    _ = parser.add_argument("--split", choices=("val", "test", "external"), required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=Options())
    evaluate(args.weights, args.dataset, args.split, args.output)


if __name__ == "__main__":
    main()
