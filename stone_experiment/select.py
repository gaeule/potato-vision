"""Select best versus last on validation only at confidence 0.45."""

import argparse
import shutil
from pathlib import Path

from stone_experiment.common import (
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
from stone_experiment.evaluate import internal_examples, score
from stone_experiment.metrics import Summary
from stone_experiment.predict import check_classes


def select(training: Path, dataset: Path, output: Path) -> None:
    """Use fresh model instances; break ties by potato recall then native best."""
    if not all((training / "weights" / f"{name}.pt").is_file() for name in ("best", "last")):
        raise PipelineError("Both native best.pt and last.pt must already exist")
    examples = internal_examples(dataset, "val")
    output = new_output(output)
    runtime()
    from ultralytics import YOLO

    scores: dict[str, Summary] = {}
    hashes: dict[str, str] = {}
    for name in ("best", "last"):
        checkpoint = training / "weights" / f"{name}.pt"
        hashes[name] = sha256(checkpoint)
        model = YOLO(str(checkpoint))
        check_classes(model, two_classes=True)
        scores[name] = score(model, examples, output / name)
        del model
    selected = max(scores, key=lambda name: (
        scores[name]["macro_f1"], scores[name]["per_class"]["potato"]["recall"], name == "best",
    ))
    _ = shutil.copyfile(training / "weights" / f"{selected}.pt", output / "selected.pt")
    write_json(output / "selection.json", {
        "selected_candidate": selected, "selected_sha256": sha256(output / "selected.pt"),
        "criterion": "max_val_macro_f1_then_potato_recall_then_native_best",
        "selection_data": "val_only", "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU, "matching_iou": MATCH_IOU,
        "matching": "confidence_ranked_class_agnostic_one_to_one",
        "candidate_hashes": hashes, "candidates": scores,
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "annotation_snapshot_sha256": sha256(dataset / "annotations.json"),
        "environment": environment(),
    })


def main() -> None:
    """Parse candidate locations without any holdout-based tuning options."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--training", type=Path, default=OUTPUTS / "train")
    _ = parser.add_argument("--dataset", type=Path, default=DATASET)
    _ = parser.add_argument("--output", type=Path, default=OUTPUTS / "selection")
    args = parser.parse_args(namespace=Options())
    select(args.training, args.dataset, args.output)


if __name__ == "__main__":
    main()
