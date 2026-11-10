"""Small deterministic CPU-only proofs; no model, inference, or GPU jobs."""

import argparse
import ast
import subprocess
import sys
from uuid import uuid4

from stone_experiment.common import (
    ANNOTATIONS,
    DATASET,
    EXPERIMENT,
    OUTPUTS,
    Options,
    PipelineError,
    new_output,
    write_json,
)
from stone_experiment.evaluate import external_examples
from stone_experiment.metrics import Detection, match_frame, summarize
from stone_experiment.prepare import verified_frames


def require(condition: bool, name: str) -> None:
    """Fail a machine-value proof with its named condition."""
    if not condition:
        raise PipelineError(f"Smoke proof failed: {name}")


def main() -> None:
    """Check isolated behavior without touching chief-owned data or old assets."""
    output = new_output(OUTPUTS / f"smoke_{uuid4().hex}")
    # Given: each CLI has its own output default but shares a typed namespace.
    for expected in (DATASET, OUTPUTS / "train", OUTPUTS / "selection"):
        parser = argparse.ArgumentParser()
        _ = parser.add_argument("--output", default=expected)
        # When: no explicit output override is supplied.
        parsed = parser.parse_args([], namespace=Options())
        # Then: namespace annotations must not shadow the command's defaults.
        require(parsed.output == expected, "command_specific_output_default")
    # Given: two true classes plus an unmatched truth and background prediction.
    truth = (
        Detection(1, (0, 0, 10, 10)),
        Detection(0, (20, 0, 30, 10)),
        Detection(0, (40, 0, 50, 10)),
    )
    predictions = (
        Detection(0, (0, 0, 10, 10), 0.9),
        Detection(1, (20, 0, 30, 10), 0.8),
        Detection(0, (70, 0, 80, 10), 0.7),
    )
    # When: matching ignores class identity and consumes each box only once.
    audit = match_frame(truth, predictions)
    summary = summarize([audit])
    # Then: wrong classes cost both FP and FN; unmatched boxes stay separate.
    require(summary["stone_as_potato"] == summary["potato_as_stone"] == 1, "cross_class_confusion")
    require(summary["per_class"]["potato"]["fp"] == summary["per_class"]["potato"]["fn"] == 2,
            "wrong_class_and_unmatched_accounting")
    require(summary["background_fp"]["potato"] == summary["missed_truth"]["potato"] == 1,
            "misses_distinguished_from_confusion")
    require(summary["macro_f1"] == 0, "macro_includes_both_classes")
    # Given: duplicate predictions ranked opposite to input order.
    duplicate = (Detection(0, (0, 0, 10, 10), 0.5), Detection(0, (0, 0, 10, 10), 0.9))
    # When: scoring one potato truth.
    one = match_frame((Detection(0, (0, 0, 10, 10)),), duplicate)
    # Then: high confidence wins once; the duplicate is background.
    require(one.matches[0].prediction == 1 and one.background_predictions == (0,), "ranked_one_to_one")
    require(summarize([one])["per_class"]["potato"]["tp"] == 1, "duplicate_not_second_tp")
    empty = match_frame((), ())
    require(summarize([empty])["macro_f1"] == 0, "empty_frames")
    require(set(summarize([one], potato_only=True)["per_class"]) == {"potato"}, "potato_only_metrics")
    frames = verified_frames(ANNOTATIONS)
    counts = {
        split: {
            "frames": sum(frame.split == split for frame in frames),
            "potato": sum(len(frame.boxes) for frame in frames if frame.split == split),
            "stone": sum(len(frame.stones) for frame in frames if frame.split == split),
        }
        for split in ("train", "val", "test")
    }
    external = external_examples()
    require(all(box.class_id == 0 for example in external for box in example.truth), "external_truth_only_potato")
    try:
        _ = new_output(output)
    except FileExistsError:
        pass
    else:
        raise PipelineError("Existing output directory was not refused")
    try:
        _ = new_output(EXPERIMENT / "forbidden_smoke_output")
    except PipelineError:
        pass
    else:
        raise PipelineError("Output outside outputs/ was not refused")
    modules = ("video", "prepare", "train", "select", "evaluate", "compare")
    for module in modules:
        _ = subprocess.run([sys.executable, "-B", "-m", f"stone_experiment.{module}", "--help"],
                           check=True, capture_output=True, text=True)
    for path in EXPERIMENT.glob("*.py"):
        _ = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    write_json(output / "proof.json", {
        "status": "passed", "gpu_jobs": 0, "annotation_counts": counts,
        "external_frames": len(external), "confusion_proof": summary,
        "cli_modules": modules, "syntax": "passed", "existing_output_refused": True,
        "outside_outputs_refused": True,
        "command_specific_defaults": True,
    })
    print(f"CPU smoke passed: {output / 'proof.json'}")


if __name__ == "__main__":
    main()
