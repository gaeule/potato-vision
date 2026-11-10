"""Predeclared best/last selector using only the eleven validation images."""

import shutil
from pathlib import Path

from retraining_data.schema import require
from retraining_experiment.common import (
    DATASET,
    OUTPUTS,
    PROTOCOL,
    new_output,
    record_context,
    runtime,
    sha256,
    write_json,
)
from retraining_experiment.data import examples, verified
from retraining_experiment.evaluate import dataset_receipt, score
from stone_experiment.metrics import Summary
from stone_experiment.predict import check_classes
from stone_experiment.schema import document, text


def winner(scores: dict[str, Summary]) -> str:
    """Maximize validation macro F1, then potato recall, then native best."""
    require(set(scores) == {"best", "last"}, "Expected native best and last candidates")
    return max(
        scores,
        key=lambda name: (
            scores[name]["macro_f1"],
            scores[name]["per_class"]["potato"]["recall"],
            name == "best",
        ),
    )


def select(
    training: Path = OUTPUTS / "train",
    dataset: Path = DATASET,
    output: Path = OUTPUTS / "selection",
) -> None:
    """Freshly reopen each native checkpoint; no trainer-retained model is scored."""
    training, dataset = training.resolve(), dataset.resolve()
    checkpoints = tuple(
        training / "weights" / f"{name}.pt" for name in ("best", "last")
    )
    require(
        all(p.is_file() for p in checkpoints), "Native best.pt and last.pt must exist"
    )
    recorded = document(training / "checkpoints.json")
    require(
        all(sha256(p) == text(recorded[p.stem]) for p in checkpoints),
        "Training checkpoint hash differs",
    )
    training_inputs = document(training / "data-provenance.json")
    require(
        text(training_inputs["original_snapshot_sha256"])
        == sha256(dataset / "annotations.json"),
        "Training and selection datasets differ",
    )
    snapshot = verified(dataset)
    inputs = examples(dataset, snapshot, "val")
    require(len(inputs) == 11, "Selection requires val11 only")
    output = new_output(output)
    dataset_receipt(dataset, snapshot, output)
    with runtime(output):
        from ultralytics import YOLO

        record_context(output, checkpoints)
        scores: dict[str, Summary] = {}
        for name, checkpoint in zip(("best", "last"), checkpoints, strict=True):
            model = YOLO(str(checkpoint))
            check_classes(model, two_classes=True)
            scores[name] = score(model, inputs, output / name)
            del model
        selected = winner(scores)
        source = training / "weights" / f"{selected}.pt"
        _ = shutil.copyfile(source, output / "selected.pt")
        write_json(
            output / "selection.json",
            {
                **PROTOCOL,
                "selected_candidate": selected,
                "selected_sha256": sha256(output / "selected.pt"),
                "candidate_hashes": {p.stem: sha256(p) for p in checkpoints},
                "criterion": "max_val_macro_f1_then_potato_recall_then_native_best",
                "selection_data": "val11_only",
                "candidates": scores,
                "provenance_sha256": sha256(output / "provenance.json"),
                "environment_sha256": sha256(output / "environment.json"),
                "dataset_receipt_sha256": sha256(output / "dataset.json"),
            },
        )


if __name__ == "__main__":
    from retraining_experiment.cli import arguments

    args = arguments("select")
    select(args.training, args.dataset, args.output)
