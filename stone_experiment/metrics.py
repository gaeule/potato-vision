"""Confidence-ranked, class-agnostic one-to-one operational matching."""

from dataclasses import dataclass
from typing import Final, TypedDict

from stone_experiment.common import MATCH_IOU, NAMES, PipelineError
from stone_experiment.schema import Box


@dataclass(frozen=True, slots=True)
class Detection:
    """A pixel-space box and its class; truth has confidence 1."""

    class_id: int
    xyxy: Box
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.class_id not in (0, 1) or not 0 <= self.confidence <= 1:
            raise PipelineError(f"Invalid detection class/confidence: {self}")


@dataclass(frozen=True, slots=True)
class Match:
    """Indices refer to original truth/prediction arrays, not sorted copies."""

    truth: int
    prediction: int
    iou: float


@dataclass(frozen=True, slots=True)
class Audit:
    """Full auditable matching, including missed truth and background boxes."""

    truth: tuple[Detection, ...]
    predictions: tuple[Detection, ...]
    matches: tuple[Match, ...]
    missed_truth: tuple[int, ...]
    background_predictions: tuple[int, ...]


class Counts(TypedDict):
    """Per-class operational statistics, including cross-class errors."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class Summary(TypedDict):
    """Aggregate counts represent observations, never distinct physical objects."""

    frames: int
    per_class: dict[str, Counts]
    macro_f1: float
    stone_as_potato: int
    potato_as_stone: int
    missed_truth: dict[str, int]
    background_fp: dict[str, int]
    confusion_truth_rows_prediction_columns: list[list[int]]


CONFUSION_ORDER: Final = ("potato", "stone", "background")


def iou(first: Box, second: Box) -> float:
    """Compute pixel-coordinate intersection over union."""
    area = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    area *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    union = (first[2] - first[0]) * (first[3] - first[1])
    union += (second[2] - second[0]) * (second[3] - second[1]) - area
    return area / union if union > 0 else 0.0


def match_frame(truth: tuple[Detection, ...], predictions: tuple[Detection, ...]) -> Audit:
    """Greedily match confidence-ranked predictions to highest-IoU unused GT."""
    remaining = set(range(len(truth)))
    matched: list[Match] = []
    background: list[int] = []
    for pred_index in sorted(range(len(predictions)), key=lambda i: (-predictions[i].confidence, i)):
        best = max(sorted(remaining), key=lambda i: iou(truth[i].xyxy, predictions[pred_index].xyxy),
                   default=None)
        overlap = iou(truth[best].xyxy, predictions[pred_index].xyxy) if best is not None else 0.0
        if best is not None and overlap >= MATCH_IOU:
            matched.append(Match(best, pred_index, overlap))
            remaining.remove(best)
        else:
            background.append(pred_index)
    return Audit(truth, predictions, tuple(matched), tuple(sorted(remaining)), tuple(background))


def summarize(audits: list[Audit], *, potato_only: bool = False) -> Summary:
    """Charge every wrong-class match as predicted-class FP and truth-class FN."""
    confusion = [[0] * 3 for _ in range(3)]
    for audit in audits:
        for pair in audit.matches:
            confusion[audit.truth[pair.truth].class_id][audit.predictions[pair.prediction].class_id] += 1
        for index in audit.missed_truth:
            confusion[audit.truth[index].class_id][2] += 1
        for index in audit.background_predictions:
            confusion[2][audit.predictions[index].class_id] += 1
    per_class: dict[str, Counts] = {}
    scored = NAMES[:1] if potato_only else NAMES
    for class_id, name in enumerate(scored):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(3) if row != class_id)
        fn = sum(confusion[class_id][column] for column in range(3) if column != class_id)
        per_class[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    return {
        "frames": len(audits), "per_class": per_class,
        "macro_f1": sum(count["f1"] for count in per_class.values()) / len(scored),
        "stone_as_potato": confusion[1][0], "potato_as_stone": confusion[0][1],
        "missed_truth": {name: confusion[i][2] for i, name in enumerate(scored)},
        "background_fp": {name: confusion[2][i] for i, name in enumerate(scored)},
        "confusion_truth_rows_prediction_columns": confusion,
    }
