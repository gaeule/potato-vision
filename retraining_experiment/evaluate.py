"""Fixed postprocessing and shared operational matching for new/old comparison."""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from postprocess_review.common import infer
from retraining_data.schema import Snapshot, require
from retraining_experiment.common import (
    BASELINE,
    DATASET,
    OUTPUTS,
    PARAMETERS,
    PROTOCOL,
    ROOT,
    SELECTED,
    check_frozen,
    check_selected,
    new_output,
    record_context,
    runtime,
    sha256,
    write_json,
)
from retraining_experiment.data import examples, verified
from stone_experiment.evaluate import Example, detections, external_examples
from stone_experiment.metrics import Audit, Detection, Summary, match_frame, summarize
from stone_experiment.predict import check_classes
from stone_experiment.schema import document, text

if TYPE_CHECKING:
    from ultralytics import YOLO


def audit_frame(
    truth: tuple[Detection, ...],
    predictions: tuple[Detection, ...],
    *,
    potato_only: bool,
) -> Audit:
    """Filter before matching so stone predictions cannot consume potato truth."""
    if potato_only:
        truth = tuple(d for d in truth if d.class_id == 0)
        predictions = tuple(d for d in predictions if d.class_id == 0)
    return match_frame(truth, predictions)


def score(
    model: "YOLO",
    inputs: tuple[Example, ...],
    output: Path,
    *,
    potato_only: bool = False,
) -> Summary:
    """Use exactly the same inference contract for selection and final comparison."""
    output.mkdir(parents=True, exist_ok=False)
    audits: list[Audit] = []
    for example in inputs:
        result = infer(model, str(example.image), PARAMETERS)
        raw = detections(result, potato_only=False)
        audit = audit_frame(example.truth, raw, potato_only=potato_only)
        audits.append(audit)
        write_json(
            output / f"{example.image.stem}.json",
            {
                "image": str(example.image),
                "image_sha256": sha256(example.image),
                "raw_detections": [asdict(d) for d in raw],
                **asdict(audit),
            },
        )
    summary = summarize(audits, potato_only=potato_only)
    write_json(output / "summary.json", summary)
    return summary


def dataset_receipt(dataset: Path, snapshot: Snapshot, output: Path) -> None:
    """Attach original 111-frame data membership and content hashes."""
    write_json(
        output / "dataset.json",
        {
            "dataset": str(dataset),
            "snapshot": asdict(snapshot),
            "snapshot_sha256": sha256(dataset / "annotations.json"),
            "manifest_sha256": sha256(dataset / "manifest.json"),
            "data_yaml_sha256": sha256(dataset / "data.yaml"),
        },
    )


def compare(
    weights: Path = SELECTED,
    baseline: Path = BASELINE,
    dataset: Path = DATASET,
    output: Path = OUTPUTS / "comparison",
) -> None:
    """Report val/test/external without any parameter or checkpoint search."""
    weights, baseline, dataset = (
        weights.resolve(),
        baseline.resolve(),
        dataset.resolve(),
    )
    require(
        weights.is_file() and baseline.is_file(),
        "Both comparison checkpoints are required",
    )
    check_frozen(baseline, "baseline_weights_sha256")
    check_selected(weights)
    snapshot = verified(dataset)
    frozen = document(ROOT / "retraining_experiment/review/protected_before.json")
    external_prefixes = (
        "baseline/external/images/test/",
        "baseline/external/labels/test/",
    )
    frozen_external = {
        path: text(value)
        for path, value in frozen.items()
        if path.startswith(external_prefixes)
        or path == "baseline/external/manifest.json"
    }
    actual_external = {
        str(p.relative_to(ROOT))
        for pattern in ("images/test/*.png", "labels/test/*.txt")
        for p in (ROOT / "baseline/external").glob(pattern)
    } | {"baseline/external/manifest.json"}
    require(
        actual_external == set(frozen_external)
        and all(
            sha256(ROOT / path) == value for path, value in frozen_external.items()
        ),
        "External evaluation inputs differ from pre-training snapshot",
    )
    external = external_examples()
    require(len(external) == 166, "Expected fixed 166-image external potato-only set")
    inputs = {
        "val": examples(dataset, snapshot, "val"),
        "test": examples(dataset, snapshot, "test"),
        "external": external,
    }
    output = new_output(output)
    dataset_receipt(dataset, snapshot, output)
    with runtime(output):
        from ultralytics import YOLO

        record_context(output, (weights, baseline))
        scores: dict[str, dict[str, Summary]] = {}
        for name, checkpoint in (
            ("old_two_class", baseline),
            ("new_selected", weights),
        ):
            model = YOLO(str(checkpoint))
            check_classes(model, two_classes=True)
            scores[name] = {
                split: score(
                    model, rows, output / name / split, potato_only=split == "external"
                )
                for split, rows in inputs.items()
            }
            del model
        write_json(
            output / "comparison.json",
            {
                **PROTOCOL,
                "scores": scores,
                "confusion_order": ["potato", "stone", "background"],
                "external_policy": "potato_only_filter_before_matching_no_stone_metrics",
                "external_manifest_sha256": sha256(
                    ROOT / "baseline/external/manifest.json"
                ),
                "external_inputs": [
                    {
                        "image": str(e.image),
                        "image_sha256": sha256(e.image),
                        "label_sha256": sha256(
                            ROOT
                            / "baseline/external/labels/test"
                            / e.image.with_suffix(".txt").name
                        ),
                        "truth": [asdict(d) for d in e.truth],
                    }
                    for e in external
                ],
            },
        )


if __name__ == "__main__":
    from retraining_experiment.cli import arguments

    args = arguments("evaluate")
    compare(args.weights, args.baseline, args.dataset, args.output)
