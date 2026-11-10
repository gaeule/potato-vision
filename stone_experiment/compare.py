"""Compare selected new and baseline models on identical frozen annotations."""

import argparse
from pathlib import Path

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
from stone_experiment.evaluate import (
    Example,
    external_examples,
    internal_examples,
    score,
)
from stone_experiment.metrics import CONFUSION_ORDER, Summary
from stone_experiment.predict import check_classes
from stone_experiment.schema import document


def compare(weights: Path, baseline: Path, dataset: Path, output: Path, *, external: bool = False) -> None:
    """No fitting, threshold tuning, YOLO val, or writes to baseline caches."""
    if not weights.is_file() or not baseline.is_file():
        raise PipelineError("Both selected checkpoints must already exist")
    examples: dict[str, tuple[Example, ...]] = {
        split: internal_examples(dataset, split) for split in ("val", "test")
    }
    if external:
        examples["external"] = external_examples()
    protocol = OUTPUTS.parent / "review/protocol.json"
    if protocol.is_file() and document(protocol)["baseline_checkpoint_sha256"] != sha256(baseline):
        raise PipelineError("Baseline checkpoint differs from the frozen protocol")
    output = new_output(output)
    runtime()
    from ultralytics import YOLO

    results: dict[str, dict[str, Summary]] = {}
    for name, checkpoint in (("baseline", baseline), ("new", weights)):
        model = YOLO(str(checkpoint))
        check_classes(model, two_classes=name == "new")
        results[name] = {
            split: score(model, subset, output / name / split, potato_only=split == "external")
            for split, subset in examples.items()
        }
        del model
    write_json(output / "comparison.json", {
        "checkpoints": {
            "new": {"path": str(weights.resolve()), "sha256": sha256(weights)},
            "baseline": {"path": str(baseline.resolve()), "sha256": sha256(baseline)},
        },
        "annotation_snapshot_sha256": sha256(dataset / "annotations.json"),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "protocol_sha256": sha256(protocol) if protocol.is_file() else None,
        "confidence": CONFIDENCE, "nms_iou": NMS_IOU, "matching_iou": MATCH_IOU,
        "matching": "confidence_ranked_class_agnostic_one_to_one",
        "confusion_order": CONFUSION_ORDER,
        "count_unit": "frame_observations_not_unique_objects",
        "external_policy": "filter_predictions_class0; stone_precision_recall_not_scored",
        "environment": environment(), "models": results,
        "new_minus_baseline": {
            split: {
                "macro_f1": results["new"][split]["macro_f1"] - results["baseline"][split]["macro_f1"],
                "stone_as_potato": results["new"][split]["stone_as_potato"] - results["baseline"][split]["stone_as_potato"],
                "potato_as_stone": results["new"][split]["potato_as_stone"] - results["baseline"][split]["potato_as_stone"],
            }
            for split in examples
        },
    })


def main() -> None:
    """Compare identical val/test inputs; optional external truth is potato-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--weights", type=Path, default=OUTPUTS / "selection/selected.pt")
    _ = parser.add_argument("--baseline", type=Path, default=BASE / "runs/train/weights/selected.pt")
    _ = parser.add_argument("--dataset", type=Path, default=DATASET)
    _ = parser.add_argument("--output", type=Path, default=OUTPUTS / "comparison")
    _ = parser.add_argument("--external", action="store_true")
    args = parser.parse_args(namespace=Options())
    compare(args.weights, args.baseline, args.dataset, args.output, external=args.external)


if __name__ == "__main__":
    main()
