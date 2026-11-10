"""Deterministic CPU proofs; no model load, GPU inference or canonical run outputs."""

import importlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from postprocess_review.common import Postprocess
from retraining_data.schema import DataError, document, mapping, require
from retraining_experiment.cli import Command, arguments
from retraining_experiment.common import (
    DATASET,
    OUTPUTS,
    PARAMETERS,
    ROOT,
    check_selected,
    new_output,
    sha256,
    write_json,
)
from retraining_experiment.data import examples, label_truth, private_copy, verified
from retraining_experiment.evaluate import audit_frame, score
from retraining_experiment.select import winner
from retraining_experiment.train import config
from retraining_experiment.video_smoke import vfr_proof
from stone_experiment.evaluate import Example
from stone_experiment.metrics import Detection, Summary, summarize


def rejected[T](action: Callable[[], T], error: type[Exception]) -> None:
    """Require the stated boundary to reject the actual operation."""
    try:
        _ = action()
    except error:
        return
    raise DataError(f"Expected {error.__name__}")


def copy_proof(root: Path) -> None:
    """Prove actual v2 split membership, source immutability and copy provenance."""
    # Given: a real verified 111-image source and complete before inventory.
    snapshot = verified(DATASET)
    before = {
        str(p.relative_to(DATASET)): sha256(p)
        for p in DATASET.rglob("*")
        if p.is_file()
    }
    output = new_output(root / "copy")
    # When: the same function used by train prepares the only loader input.
    yaml = private_copy(DATASET, snapshot, output)
    # Then: physical membership excludes test, all bytes/provenance agree.
    private = yaml.parent
    require(
        len(tuple((private / "images/train").glob("*.png"))) == 88, "Private train88"
    )
    require(len(tuple((private / "images/val").glob("*.png"))) == 11, "Private val11")
    require(
        not (private / "images/test").exists()
        and not (private / "labels/test").exists(),
        "Test excluded from physical copy",
    )
    require(
        "test:" not in yaml.read_text() and str(DATASET) not in yaml.read_text(),
        "Loader YAML cannot reference original/test paths",
    )
    receipt = document((output / "data-provenance.json").read_bytes())
    recorded = mapping(receipt["private_copy_hashes"])
    require(len(recorded) == 198, "99 image/label pairs")
    for relative, expected in recorded.items():
        copied, source = private / relative, DATASET / relative
        require(
            sha256(copied) == expected == sha256(source), "Copy bytes/hash provenance"
        )
        require(
            not copied.is_symlink() and not copied.samefile(source),
            "Copy inode isolation",
        )
    require(
        receipt["original_snapshot_sha256"] == sha256(DATASET / "annotations.json"),
        "Original111 snapshot hash provenance",
    )
    # Given/When: a simulated loader label-cache write in the private directory.
    _ = (private / "labels/train.cache").write_bytes(b"private-cache-proof")
    # Then: original membership and all content remain byte-for-byte identical.
    after = {
        str(p.relative_to(DATASET)): sha256(p)
        for p in DATASET.rglob("*")
        if p.is_file()
    }
    require(before == after, "Source must not acquire any cache or changed bytes")
    require(len(examples(DATASET, snapshot, "val")) == 11, "Validated val Examples")
    require(len(examples(DATASET, snapshot, "test")) == 12, "Validated test Examples")
    # Given/When/Then: changing a recorded source hash must reject before copying bytes.
    first = next(r for r in snapshot.frames if r.split == "train")
    changed = replace(
        snapshot,
        frames=tuple(
            replace(r, image_sha256="bad") if r == first else r for r in snapshot.frames
        ),
    )
    rejected(
        partial(private_copy, DATASET, changed, new_output(root / "corrupt")), DataError
    )


def metrics_proof() -> None:
    """Prove confidence ranking and potato-only filtering before one-to-one match."""
    # Given: overlapping stone outranks a correct potato prediction.
    truth = (Detection(0, (0.0, 0.0, 10.0, 10.0)),)
    predictions = (Detection(1, truth[0].xyxy, 0.99), Detection(0, truth[0].xyxy, 0.8))
    # When: external scoring filters before matching.
    external = summarize(
        [audit_frame(truth, predictions, potato_only=True)], potato_only=True
    )
    # Then: the potato still matches and stone metrics do not exist.
    require(set(external["per_class"]) == {"potato"}, "No external stone metrics")
    require(
        external["per_class"]["potato"]["tp"] == 1
        and external["per_class"]["potato"]["fp"] == 0,
        "Filter precedes matching",
    )
    # Given/When: the same detections under internal class-agnostic matching.
    internal = summarize([audit_frame(truth, predictions, potato_only=False)])
    # Then: high-confidence cross-class match consumes truth, not the lower potato.
    require(
        internal["potato_as_stone"] == 1 and internal["per_class"]["potato"]["tp"] == 0,
        "Existing class-agnostic ranked semantics",
    )
    # Given: exact ties, then recall ties, then different macro F1.
    tied = summarize([])
    # When/Then: choose native best independent of dictionary insertion order.
    require(winner({"last": tied, "best": tied}) == "best", "Native best tie break")
    recall: Summary = {
        **tied,
        "per_class": {
            "potato": {**tied["per_class"]["potato"], "recall": 0.5},
            "stone": tied["per_class"]["stone"],
        },
    }
    require(winner({"best": tied, "last": recall}) == "last", "Potato recall second")
    require(
        winner({"best": {**tied, "macro_f1": 0.1}, "last": recall}) == "best",
        "Macro F1 first",
    )


def wiring_proof(root: Path) -> None:
    """Exercise actual scoring wiring through only the GPU inference boundary."""
    # Given: the fixed preset and one real CPU image example.
    require(PARAMETERS == Postprocess(0.45, 0.5), "Predeclared inference parameters")
    preset = config()
    require(preset.optimizer == "SGD" and preset.seed == 42, "Unchanged train preset")
    image = next((DATASET / "images/val").glob("*.png"))
    example = Example(image, ())
    # When: real scoring runs with the GPU-only inference/conversion seams replaced.
    with (
        patch("retraining_experiment.evaluate.infer") as prediction,
        patch("retraining_experiment.evaluate.detections", return_value=()),
    ):
        summary = score(Mock(), (example,), root / "score")
    # Then: score must pass precisely the fixed parameters, not old NMS defaults.
    prediction.assert_called_once()
    require(prediction.call_args.args[2] is PARAMETERS, "Score parameter wiring")
    require(summary["frames"] == 1, "Actual CPU metric/audit aggregation")


def boundary_proof(root: Path) -> None:
    """Prove new-only outputs and nine-decimal-only label edge tolerance."""
    # Given/When/Then: duplicate and escaped output directories are refused.
    used = new_output(root / "collision")
    rejected(partial(new_output, used), FileExistsError)
    rejected(partial(new_output, ROOT / "baseline/new"), DataError)
    escaped = root / "escape"
    escaped.symlink_to(ROOT / "baseline", target_is_directory=True)
    rejected(partial(new_output, escaped / "new"), DataError)
    # Given: normalized corners miss the exact image edge by nine-decimal rounding.
    label = root / "rounding.txt"
    _ = label.write_text("0 0.333333333 0.5 0.666666667 1\n")
    # When/Then: clipping recovers an exact edge, not a materially invalid box.
    require(label_truth(label, 1280, 720)[0].xyxy[0] == 0, "Rounding edge recovery")
    _ = label.write_text("0 0.3 0.5 0.8 1\n")
    rejected(partial(label_truth, label, 1280, 720), DataError)
    _ = label.write_text("0 nan 0.5 0.8 1\n")
    rejected(partial(label_truth, label, 1280, 720), DataError)


def cli_defaults_proof() -> None:
    """Each command owns its default output; explicit output still overrides it."""
    cases: tuple[tuple[Command, str], ...] = (
        ("train", "train"),
        ("select", "selection"),
        ("evaluate", "comparison"),
        ("video", "video"),
    )
    for command, directory in cases:
        # Given: no explicit destination for this command.
        with patch.object(sys, "argv", ["command"]):
            # When: the real parser fills its namespace.
            parsed = arguments(command)
        # Then: one command must not inherit another command's output.
        require(parsed.output == OUTPUTS / directory, f"{command} output default")
        with patch.object(
            sys, "argv", ["command", "--output", str(OUTPUTS / "custom")]
        ):
            overridden = arguments(command)
        require(overridden.output == OUTPUTS / "custom", "Explicit output respected")


def checkpoint_proof(root: Path) -> None:
    """A changed selected checkpoint is rejected before model construction."""
    # Given: an actual file and its validation-selection receipt.
    selected = new_output(root / "checkpoint") / "selected.pt"
    _ = selected.write_bytes(b"selected-checkpoint")
    write_json(
        selected.parent / "selection.json",
        {"selection_data": "val11_only", "selected_sha256": sha256(selected)},
    )
    check_selected(selected)
    # When: checkpoint bytes change while its recorded selection hash stays fixed.
    _ = selected.write_bytes(b"changed-checkpoint")
    # Then: the inference gate refuses it without needing a model or GPU.
    rejected(partial(check_selected, selected), DataError)


def imports_proof() -> None:
    """Import every CLI in a fresh process; no Ultralytics/torch or filesystem writes."""
    # Given: source module names, with no model/checkpoint construction.
    modules = ("common", "data", "train", "select", "evaluate", "video", "cli")
    # When: every public module is imported locally and in a fresh Python process.
    for name in modules:
        _ = importlib.import_module(f"retraining_experiment.{name}")
    code = (
        "import importlib,sys; "
        f"[importlib.import_module('retraining_experiment.'+n) for n in {modules!r}]; "
        "assert 'ultralytics' not in sys.modules; assert 'torch' not in sys.modules"
    )
    _ = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Then: all command help surfaces exit cleanly without starting GPU work.
    for module in ("train", "select", "evaluate", "video"):
        _ = subprocess.run(
            [sys.executable, "-B", "-m", f"retraining_experiment.{module}", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )


def main() -> None:
    """Use only a disposable namespaced CPU-test subtree under new outputs."""
    cli_defaults_proof()
    require(OUTPUTS.resolve() == OUTPUTS, "Refuse symlink outputs root")
    OUTPUTS.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="cpu-smoke-", dir=OUTPUTS) as disposable:
        root = Path(disposable)
        copy_proof(root)
        metrics_proof()
        wiring_proof(root)
        boundary_proof(root)
        imports_proof()
        vfr_proof(root)
        checkpoint_proof(root)
    require(
        "ultralytics" not in sys.modules and "torch" not in sys.modules,
        "Smoke must not import GPU frameworks",
    )
    print(
        "PASS: copy/provenance, isolation, fixed wiring, selector, potato-only metrics, CLI imports"
    )


if __name__ == "__main__":
    main()
