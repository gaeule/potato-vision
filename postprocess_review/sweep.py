"""Compare NMS on validation before looking at test or training diagnostics."""

import csv
import json
from dataclasses import asdict
from typing import Final

from PIL import Image

from postprocess_review.common import (
    BASE,
    DATASET,
    OUTPUT,
    ROOT,
    WEIGHTS,
    Postprocess,
    infer,
    load_model,
    sha256,
    write_json,
)
from stone_experiment.evaluate import detections, internal_examples
from stone_experiment.metrics import Audit, Detection, Summary, match_frame, summarize
from stone_experiment.prepare import verify_dataset

GRID: Final = (0.3, 0.4, 0.5, 0.6, 0.7)
DIAGNOSTICS: Final = (20, 70, 78, 80, 100, 130, 134)


def main() -> None:
    """Freeze a validation-only choice, then report exploratory checks."""
    output = OUTPUT / "nms"
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    if tuple(plan["nms_iou_candidates"]) != GRID or plan["confidence"] != 0.45:
        raise ValueError("Protocol differs from this analysis")
    frames = verify_dataset(DATASET)
    by_second = {frame.second: frame for frame in frames}
    model = load_model()
    results: dict[str, dict[float, Summary]] = {}
    golden = json.loads((ROOT / "stone_experiment/outputs/comparison/comparison.json").read_text())
    for split in ("val", "test"):
        examples = internal_examples(DATASET, split)
        results[split] = {}
        for threshold in GRID:
            folder = output / split / f"iou_{threshold:.2f}"
            folder.mkdir(parents=True)
            audits: list[Audit] = []
            for example in examples:
                prediction = infer(model, str(example.image), Postprocess(0.45, threshold))
                audit = match_frame(example.truth, detections(prediction, potato_only=False))
                audits.append(audit)
                write_json(folder / f"{example.image.stem}.json", {
                    "image": str(example.image), **asdict(audit),
                })
                second = int(example.image.stem.split("_")[1])
                if second in DIAGNOSTICS:
                    Image.fromarray(prediction.plot()[..., ::-1]).save(folder / f"{example.image.stem}.jpg")
            results[split][threshold] = summarize(audits)
            print("SCORED", split, threshold, results[split][threshold], flush=True)
        if results[split][0.7] != golden["models"]["new"][split]:
            raise ValueError(f"Baseline NMS=0.7 replay differs for {split}")
        if split == "val":
            selected = max(GRID, key=lambda t: (
                results["val"][t]["macro_f1"],
                results["val"][t]["per_class"]["potato"]["recall"],
                -abs(t - 0.5), t,
            ))
            best = results["val"][selected]
            equivalent = [
                t for t in GRID if
                (results["val"][t]["macro_f1"], results["val"][t]["per_class"]["potato"]["recall"]) ==
                (best["macro_f1"], best["per_class"]["potato"]["recall"])
            ]
            write_json(output / "selection.json", {
                "selected_nms_iou": selected, "validation_equivalent_candidates": equivalent,
                "confidence": 0.45, "criterion": plan["selection"],
                "selection_data": "validation_only_before_test_and_diagnostics",
                "checkpoint_sha256": sha256(WEIGHTS),
                "protocol_sha256": sha256(BASE / "protocol.json"),
                "baseline_configuration_changed": False,
            })
            print("NMS_SELECTED", selected, "EQUIVALENT", equivalent, flush=True)
    # These examples are diagnostic evidence, not selection data.
    for second in (20, 70, 78, 80):
        frame = by_second[second]
        truth = tuple(Detection(class_id, box)
                      for class_id, group in enumerate((frame.boxes, frame.stones)) for box in group)
        for threshold in GRID:
            folder = output / "train_diagnostics" / f"iou_{threshold:.2f}"
            folder.mkdir(parents=True, exist_ok=True)
            prediction = infer(model, str(DATASET / "images/train" / frame.filename),
                               Postprocess(0.45, threshold))
            audit = match_frame(truth, detections(prediction, potato_only=False))
            write_json(folder / f"{frame.filename[:-4]}.json", {
                "second": second, "summary": summarize([audit]), **asdict(audit),
            })
            Image.fromarray(prediction.plot()[..., ::-1]).save(folder / f"{frame.filename[:-4]}.jpg")
    write_json(output / "summary.json", {
        "confidence": 0.45, "matching_iou": 0.5, "agnostic_nms": False,
        "checkpoint_sha256": sha256(WEIGHTS), "results": results,
        "note": "All candidates rerun actual model NMS. No retraining or production setting changes.",
    })
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "nms_iou", "class", "tp", "fp", "fn", "precision", "recall", "f1"])
        for split, candidates in results.items():
            for threshold, summary in candidates.items():
                for name, counts in summary["per_class"].items():
                    writer.writerow([split, threshold, name, *[counts[k] for k in
                                     ("tp", "fp", "fn", "precision", "recall", "f1")]])
    print("NMS_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
