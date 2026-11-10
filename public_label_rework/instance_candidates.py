"""Versioned candidate provenance and full-negative validation, without inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from public_label_rework import instance_roster as r
from public_label_rework import snap as s
from retraining_data.schema import (
    Json,
    JsonMap,
    array,
    boolean,
    document,
    mapping,
    number,
    require,
    text,
)

POLICY: Final = "sam-roster-nearest3-v2"
REVISION: Final = "nearest3-v2"


def component(raw: s.Mask, positive: r.Point) -> s.Mask:
    """Select only the genuine eight-connected component hit by the positive."""
    _, labels, _, _ = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=8
    )
    data = np.asarray(labels, dtype=np.int32)
    label = int(data[int(positive[1]), int(positive[0])])
    return np.equal(data, label) if label else np.zeros_like(raw)


def negatives(target: r.Target, image: r.RosterImage) -> tuple[r.Point, ...]:
    others = {
        t.point
        for t in image.targets
        if t.target_id != target.target_id
        and t.status == "identifiable"
        and t.point is not None
    }
    return tuple(sorted(set(target.negative_points) | others))


def wire_negatives(
    positive: r.Point | None, full: tuple[r.Point, ...]
) -> tuple[r.Point, ...]:
    """Uniform maximum of three; all distance ties break by x then y."""
    if positive is None:
        return ()
    return tuple(
        sorted(
            full,
            key=lambda q: (
                (q[0] - positive[0]) ** 2 + (q[1] - positive[1]) ** 2,
                q[0],
                q[1],
            ),
        )[:3]
    )


def prompt_checks(
    selected: s.Mask, wire: tuple[r.Point, ...], full: tuple[r.Point, ...]
) -> tuple[bool, bool, bool]:
    """Point inclusion and both exclusion checks; full-set acceptance is mandatory."""
    hit = bool(np.any(selected))
    return (
        hit,
        hit and all(not selected[int(y), int(x)] for x, y in wire),
        hit and all(not selected[int(y), int(x)] for x, y in full),
    )


@dataclass(frozen=True, slots=True)
class Candidate:
    image_id: str
    target_id: str
    image_sha256: str
    box: s.Box | None
    positive: r.Point | None
    negative: tuple[r.Point, ...]
    raw_masks: tuple[tuple[str, str], ...]
    quality_scores: tuple[float, ...]
    reason: str
    elapsed_seconds: float
    initial_status: str
    initial_reason: str
    producer: str = "sam_point_group"
    algorithm_revision: str = REVISION
    model_called: bool = False
    wire_negative: tuple[r.Point, ...] = ()
    wire_points: tuple[r.Point, ...] = ()
    wire_labels: tuple[int, ...] = ()
    positive_in_component: bool | None = None
    wire_consistent: bool | None = None
    full_consistent: bool | None = None

    def json(self) -> JsonMap:
        return document(json.dumps(asdict(self)).encode())


def save_mask(path: Path, raw: s.Mask) -> tuple[str, str]:
    with path.open("xb") as stream:
        np.save(stream, raw, allow_pickle=False)
    return path.name, s.digest(path)


def parse_box(value: Json) -> s.Box:
    values = array(value)
    require(len(values) == 4, "four_box_coordinates_required")
    box = number(values[0]), number(values[1]), number(values[2]), number(values[3])
    require(s.area(box) > 0, "positive_candidate_area_required")
    return box


def verify_candidates(
    records: Sequence[Json], roster: tuple[r.RosterImage, ...], root: Path
) -> None:
    """Refuse v1 reinterpretation, wire changes, lost negatives and unbacked boxes."""
    r.membership(records, roster)
    lookup = {(i.image_id, t.target_id): (i, t) for i in roster for t in i.targets}
    for value in records:
        row = mapping(value)
        image, target = lookup[text(row["image_id"]), text(row["target_id"])]
        require(row["producer"] == "sam_point_group", "unknown_candidate_producer")
        require(
            row.get("algorithm_revision") == REVISION, "candidate_revision_mismatch"
        )
        expected: Json = None if target.point is None else list(target.point)
        require(
            row["positive"] == expected
            and row["initial_status"] == target.status
            and row["initial_reason"] == target.reason,
            "candidate_roster_tamper",
        )
        full = tuple(r.point(v) for v in array(row["negative"]))
        require(full == negatives(target, image), "negative_provenance_mismatch")
        called = boolean(row["model_called"])
        require(called or not array(row["raw_masks"]), "uncalled_raw_mask")
        require(target.point is not None or not called, "null_target_model_call")
        wire = wire_negatives(target.point, full) if called else ()
        require(
            tuple(r.point(v) for v in array(row["wire_negative"])) == wire,
            "wire_provenance_mismatch",
        )
        points = () if not called or target.point is None else (target.point,) + wire
        labels = () if not called else (1,) + (0,) * len(wire)
        require(
            row["wire_points"] == [list(p) for p in points]
            and row["wire_labels"] == list(labels),
            "wire_group_mismatch",
        )
        raw: list[s.Mask] = []
        for entry in array(row["raw_masks"]):
            pair = array(entry)
            require(len(pair) == 2, "mask_path_hash_required")
            filename = text(pair[0])
            require(Path(filename).name == filename, "unsafe_mask_path")
            path = root / image.image_id / filename
            require(s.digest(path) == text(pair[1]), "candidate_mask_tamper")
            loaded = np.load(path, allow_pickle=False)
            require(
                loaded.dtype == np.bool_
                and loaded.shape == (image.height, image.width),
                "raw_mask_contract",
            )
            raw.append(np.asarray(loaded, dtype=np.bool_))
        require(len(array(row["quality_scores"])) == len(raw), "mask_score_membership")
        _ = tuple(number(v) for v in array(row["quality_scores"]))
        flags: tuple[bool | None, bool | None, bool | None] = (None, None, None)
        if len(raw) == 1 and target.point is not None:
            selected = component(raw[0], target.point)
            flags = prompt_checks(selected, wire, full)
        require(
            (
                row["positive_in_component"],
                row["wire_consistent"],
                row["full_consistent"],
            )
            == flags,
            "prompt_check_tamper",
        )
        if row["box"] is not None:
            if target.point is None:
                raise s.SnapError("null_target_with_candidate")
            require(len(raw) == 1, "unbacked_candidate")
            require(flags[2] is True, "candidate_prompt_inconsistent")
            require(
                parse_box(row["box"]) == s.extent(component(raw[0], target.point)),
                "candidate_box_tamper",
            )
        else:
            require(text(row["reason"]) != "tool_candidate", "missing_candidate_box")
