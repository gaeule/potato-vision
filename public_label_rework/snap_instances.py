"""Nearest-three SAM delivery; full negative provenance and unchanged default snap."""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from public_label_rework import instance_roster as r
from public_label_rework import snap as s
from public_label_rework.instance_candidates import (
    POLICY,
    REVISION,
    Candidate,
    component,
    negatives,
    parse_box,
    prompt_checks,
    save_mask,
    verify_candidates,
    wire_negatives,
)
from public_label_rework.snap_prompts import Guidance, refinement_result
from retraining_data.schema import Json, array, document, mapping, require, text

__all__ = (
    "POLICY",
    "REVISION",
    "Candidate",
    "component",
    "negatives",
    "parse_box",
    "prompt_checks",
    "save_mask",
    "verify_candidates",
    "wire_negatives",
)

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Results


def generate(
    image: s.Image, roster: r.RosterImage, output: Path
) -> tuple[Candidate, ...]:
    """One locked image unit; limit delivery only, never full-set validation."""
    records: list[Candidate] = []
    bounds = s.content_box(image)
    for target in roster.targets:
        started = time.monotonic()
        neg = negatives(target, roster)
        box: s.Box | None = None
        masks: tuple[tuple[str, str], ...] = ()
        scores: tuple[float, ...] = ()
        reason = "null_body_point"
        p = target.point
        wire: tuple[r.Point, ...] = ()
        points: tuple[r.Point, ...] = ()
        labels: tuple[int, ...] = ()
        called = False
        flags: tuple[bool | None, bool | None, bool | None] = (None, None, None)
        if p is not None:
            reason = "point_outside_content"
            if all(
                bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3]
                for x, y in (p,) + neg
            ):
                reason = "contradictory_roster_points"
                if p not in neg:
                    wire = wire_negatives(p, neg)
                    points, labels = (p,) + wire, (1,) + (0,) * len(wire)
                    predictions = s.model().predict(
                        image,
                        points=[points],
                        labels=[labels],
                        device=0,
                        conf=0.0,
                        verbose=False,
                        save=False,
                    )
                    called = True
                    raw = s.masks_from(predictions)
                    scores = r.qualities(predictions)
                    require(len(scores) == len(raw), "sam_mask_score_count_mismatch")
                    require(
                        all(m.shape == image.shape[:2] for m in raw), "sam_raster_shape"
                    )
                    masks = tuple(
                        save_mask(output / f"{target.target_id}-point-{i}.npy", m)
                        for i, m in enumerate(raw)
                    )
                    reason = "point_mask_count_mismatch"
                    if len(raw) == 1:
                        selected = component(raw[0], p)
                        flags = prompt_checks(selected, wire, neg)
                        reason = "positive_not_in_raw_component"
                        if flags[0]:
                            reason = "negative_in_raw_component"
                            if flags[2]:
                                box, reason = s.extent(selected), "tool_candidate"
        records.append(
            Candidate(
                roster.image_id,
                target.target_id,
                roster.image_sha256,
                box,
                p,
                neg,
                masks,
                scores,
                reason,
                time.monotonic() - started,
                target.status,
                target.reason,
                model_called=called,
                wire_negative=wire,
                wire_points=points,
                wire_labels=labels,
                positive_in_component=flags[0],
                wire_consistent=flags[1],
                full_consistent=flags[2],
            )
        )
    return tuple(records)


@contextmanager
def capture_masks() -> Generator[list[s.Mask]]:
    """Forward real toolkit outputs unchanged while retaining default API masks."""
    captured: list[s.Mask] = []
    original = s.masks_from

    def capture(predictions: Iterable[Results | Tensor]) -> tuple[s.Mask, ...]:
        masks = original(predictions)
        captured.extend(masks)
        return masks

    s.masks_from = capture
    try:
        yield captured
    finally:
        s.masks_from = original


def evaluate(image: s.Image, records: list[Json], output: Path) -> list[Json]:
    """Score every frozen record once; absence contributes a failed receipt."""
    results: list[Json] = []
    for value in records:
        row = mapping(value)
        receipt: Json = None
        raw_record: Json = None
        consistent = False
        started = time.monotonic()
        if row["box"] is not None:
            box = parse_box(row["box"])
            with capture_masks() as captured:
                result = s.snap(image, box)
            require(len(captured) == 1, "default_api_raw_mask_count")
            raw = captured[0]
            name, digest = save_mask(
                output / f"{text(row['target_id'])}-default.npy", raw
            )
            raw_record = [name, digest]
            positive = r.point(row["positive"])
            guidance = Guidance(
                box, (positive,), tuple(r.point(v) for v in array(row["negative"]))
            )
            check = refinement_result(
                raw, guidance, s.prompt(box, s.content_box(image))
            )
            require(check.result == result, "default_receipt_capture_mismatch")
            consistent = check.point_consistent
            receipt = document(json.dumps(asdict(result)).encode())
        results.append(
            {
                "image_id": row["image_id"],
                "target_id": row["target_id"],
                "image_sha256": row["image_sha256"],
                "candidate": row,
                "default_result": receipt,
                "default_raw_mask": raw_record,
                "prompt_consistent": consistent,
                "elapsed_seconds": time.monotonic() - started,
                "semantic_single_body": None,
                "independent_review_required": True,
            }
        )
    return results
