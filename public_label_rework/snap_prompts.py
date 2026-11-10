"""Explicit single-object SAM refinement; geometry is never semantic approval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from public_label_rework import snap as s

type Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Guidance:
    """Original box and observed body/background clicks in image coordinates."""

    original: s.Box
    positive: tuple[Point, ...]
    negative: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class Refinement:
    """Unedited SAM mask and geometry receipt, pending single-body visual review."""

    guidance: Guidance
    padded: s.Box
    raw_mask: s.Mask | None
    result: s.SnapResult
    point_consistent: bool
    review_required: Literal[True] = field(default=True, init=False)

    @property
    def candidate(self) -> s.SnapResult | None:
        """Expose only a geometrically valid, click-consistent review candidate."""
        return self.result if self.result.tight and self.point_consistent else None


def refinement_result(raw: s.Mask, guidance: Guidance, padded: s.Box) -> Refinement:
    """Keep the geometry receipt, but withhold masks contradicting given clicks."""
    count, label_data, stat_data, _ = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=8
    )
    labels = np.asarray(label_data, dtype=np.int32)
    stats = np.asarray(stat_data, dtype=np.int32)
    index = 1 + int(np.argmax(stats[1:count, cv2.CC_STAT_AREA])) if count > 1 else 0
    consistent = (
        index > 0
        and all(labels[int(y), int(x)] == index for x, y in guidance.positive)
        and all(labels[int(y), int(x)] != index for x, y in guidance.negative)
    )
    result = s.mask_result(raw, guidance.original, padded)
    return Refinement(guidance, padded, raw, result, bool(consistent))


def refine(image: s.Image, guidance: Guidance) -> Refinement:
    """Send one grouped point set plus the usual padded box; never shrink masks."""
    bounds = s.content_box(image)
    original = s.clip(guidance.original, bounds)
    padded = s.prompt(original, bounds)
    points = guidance.positive + guidance.negative
    reason = ""
    if not np.isfinite(guidance.original).all() or not s.area(original):
        reason = "empty_or_degenerate_prompt"
    elif not guidance.positive or not guidance.negative:
        reason = "positive_and_negative_points_required"
    elif not all(
        bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3] for x, y in points
    ):
        reason = "point_outside_content"
    elif not all(
        padded[0] <= x < padded[2] and padded[1] <= y < padded[3]
        for x, y in guidance.positive
    ):
        reason = "positive_point_outside_prompt"
    elif set(guidance.positive).intersection(guidance.negative):
        reason = "contradictory_point_labels"
    if reason:
        return Refinement(
            guidance, padded, None, s.SnapResult(original, False, reason, 0.0), False
        )
    predictions = s.model().predict(
        image,
        bboxes=[padded],
        points=[points],
        labels=[[1] * len(guidance.positive) + [0] * len(guidance.negative)],
        device=0,
        conf=0.0,
        verbose=False,
        save=False,
    )
    masks = s.masks_from(predictions)
    if len(masks) != 1:
        raise s.SnapError("prompt_mask_count_mismatch")
    clipped_guidance = Guidance(original, guidance.positive, guidance.negative)
    return refinement_result(masks[0], clipped_guidance, padded)
