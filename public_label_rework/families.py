"""Bounded dihedral hashing and conservative consistency versus fold families."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from public_label_rework.common import OUTPUTS
from public_label_rework.labelio import Xyxy
from retraining_data.schema import JsonMap, require

POPCOUNT: Final = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
type BytesArray = NDArray[np.uint8]
type DistanceArray = NDArray[np.uint16]


@dataclass(frozen=True, slots=True)
class Feature:
    image_id: str
    source: str
    box_count: int
    hashes: BytesArray
    appearance: BytesArray


@dataclass(frozen=True, slots=True)
class FamilyReport:
    """Stable downstream schema: family IDs map to sorted image stems."""

    strict: dict[str, list[str]]
    loose: dict[str, list[str]]
    image_to_strict: dict[str, str]
    image_to_loose: dict[str, str]
    sources: dict[str, JsonMap]
    splits: list[JsonMap]
    examples: list[list[str]]
    threshold_method: str = "Per source nearest compatible Hamming Q25 floor capped12 strict; Q50 floor capped20 loose; 257=incompatible/self; strict minimum1, loose>strict. Caps avoid observed threshold24 chains."
    strict_rule: str = "D4 average16 Hamming, pairwise count delta<=1, RGB32 content MAE<=8, complete linkage, cap6, nearest-first split"
    loose_rule: str = "D4 average16 Hamming, edge count delta<=1, connected components, unsplit for folds"
    schema_version: int = 1


def picture_geometry(
    image: Image.Image,
) -> tuple[tuple[int, ...], tuple[int, ...], float | None]:
    """Near-black bars and linear wedge estimates, not semantic crop decisions."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    black = gray <= 24
    rows = tuple(int(i) for i in np.flatnonzero(black.mean(axis=1) >= 0.995))
    cols = tuple(int(i) for i in np.flatnonzero(black.mean(axis=0) >= 0.995))
    angles: list[float] = []
    for k in range(4):
        mask = np.rot90(black, k)
        boundary = np.argmax(~mask, axis=0)
        valid = (boundary >= 3) & (boundary < mask.shape[0] * 0.2) & (~mask).any(axis=0)
        for side in (
            np.arange(mask.shape[1]) < mask.shape[1] * 0.4,
            np.arange(mask.shape[1]) > mask.shape[1] * 0.6,
        ):
            xs = np.flatnonzero(valid & side)
            if len(xs) < max(16, mask.shape[1] // 10):
                continue
            ys = boundary[xs]
            slope, intercept = np.polyfit(xs, ys, 1)
            residual = float(np.mean((ys - (slope * xs + intercept)) ** 2))
            angle = math.degrees(math.atan(float(slope)))
            if 0.5 <= abs(angle) <= 15 and residual <= 2.0 and np.ptp(ys) >= 4:
                angles.append(angle)
    return rows, cols, round(float(np.median(angles)), 3) if len(angles) >= 2 else None


def variant_hashes(image: Image.Image) -> BytesArray:
    """Eight D4 variants of a 16x16 bilinear average hash, packed into 32 bytes."""
    gray = np.asarray(
        image.convert("L").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32
    )
    variants = [
        v.ravel() > v.mean()
        for k in range(4)
        for v in (np.rot90(gray, k), np.fliplr(np.rot90(gray, k)))
    ]
    return np.packbits(np.stack(variants), axis=1)


def appearance(image: Image.Image, rects: tuple[Xyxy, ...]) -> BytesArray:
    """D4 RGB content thumbnails veto similar silhouettes of different tubers."""
    if rects:
        region = (
            min(b[0] for b in rects),
            min(b[1] for b in rects),
            max(b[2] for b in rects),
            max(b[3] for b in rects),
        )
        image = image.crop(region)
    rgb = np.asarray(
        image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8
    )
    return np.stack(
        [v for k in range(4) for v in (np.rot90(rgb, k), np.fliplr(np.rot90(rgb, k)))]
    )


def distances(features: list[Feature]) -> DistanceArray:
    """At most 32 query images per vectorized batch; incompatible counts use 257."""
    n = len(features)
    hashes = np.stack([f.hashes for f in features])
    counts = np.array([f.box_count for f in features], dtype=np.int32)
    result = np.full((n, n), 257, dtype=np.uint16)
    for start in range(0, n, 32):
        xor = np.bitwise_xor(
            hashes[start : start + 32, :, None, :], hashes[None, :, 0, :]
        )
        hamming = POPCOUNT[xor].sum(axis=3).min(axis=1)
        compatible = np.abs(counts[start : start + 32, None] - counts[None, :]) <= 1
        result[start : start + 32] = np.where(compatible, hamming, 257)
    np.fill_diagonal(result, 257)
    return result


def components(distance: DistanceArray, threshold: int) -> list[list[int]]:
    """Deterministic connected components for deliberately conservative fold isolation."""
    seen: set[int] = set()
    result: list[list[int]] = []
    for first in range(len(distance)):
        if first in seen:
            continue
        seen.add(first)
        pending = [first]
        group: list[int] = []
        while pending:
            item = pending.pop()
            group.append(item)
            for candidate in np.flatnonzero(distance[item] <= threshold):
                index = int(candidate)
                if index not in seen:
                    seen.add(index)
                    pending.append(index)
        result.append(sorted(group))
    return result


def strict_groups(
    features: list[Feature], distance: DistanceArray, threshold: int
) -> list[list[int]]:
    """Split chains by complete linkage, RGB agreement and a hard six-member cap."""
    groups: list[list[int]] = []
    for component in components(distance, threshold):
        remaining = set(component)
        while remaining:
            seed = min(remaining)
            group = [seed]
            remaining.remove(seed)
            for candidate in sorted(
                remaining, key=lambda i: (int(distance[seed, i]), features[i].image_id)
            ):
                if len(group) == 6:
                    break
                compatible = all(
                    distance[candidate, member] <= threshold for member in group
                )
                if not compatible:
                    continue
                # Count gate above also applies to every pair, not only the seed.
                for member in group:
                    delta = (
                        features[candidate].appearance.astype(np.int16)
                        - features[member].appearance[0]
                    )
                    if float(np.abs(delta).mean(axis=(1, 2, 3)).min()) > 8:
                        break
                else:
                    group.append(candidate)
                    remaining.remove(candidate)
            groups.append(sorted(group))
    return groups


def histogram(values: Iterable[int]) -> JsonMap:
    return {str(k): v for k, v in sorted(Counter(values).items())}


def family_id(kind: str, members: list[str]) -> str:
    return f"{kind}-{hashlib.sha256(chr(10).join(members).encode()).hexdigest()[:16]}"


def build_families(features: list[Feature]) -> FamilyReport:
    """Thresholds from observed source nearest-distance histograms, never gold."""
    strict: dict[str, list[str]] = {}
    loose: dict[str, list[str]] = {}
    strict_by_image: dict[str, str] = {}
    loose_by_image: dict[str, str] = {}
    reports: dict[str, JsonMap] = {}
    splits: list[JsonMap] = []
    for source in sorted({f.source for f in features}):
        subset = sorted(
            (f for f in features if f.source == source), key=lambda f: f.image_id
        )
        distance = distances(subset)
        near = distance.min(axis=1)
        observed = near[near <= 256]
        strict_threshold = (
            min(12, max(1, int(np.quantile(observed, 0.25)))) if len(observed) else 1
        )
        loose_threshold = (
            min(20, max(strict_threshold + 1, int(np.quantile(observed, 0.5))))
            if len(observed)
            else 2
        )
        strict_components = components(distance, strict_threshold)
        strict_parts = strict_groups(subset, distance, strict_threshold)
        loose_parts = components(distance, loose_threshold)
        for kind, parts, groups, lookup in (
            ("strict", strict_parts, strict, strict_by_image),
            ("loose", loose_parts, loose, loose_by_image),
        ):
            for part in parts:
                members = sorted(subset[i].image_id for i in part)
                identity = family_id(kind, members)
                require(identity not in groups, "Family ID collision")
                groups[identity] = members
                for member in members:
                    lookup[member] = identity
        for component in strict_components:
            children = [part for part in strict_parts if part[0] in component]
            if len(children) > 1:
                splits.append(
                    {
                        "source": source,
                        "raw_members": [subset[i].image_id for i in component],
                        "oversized": len(component) > 6,
                        "children": [
                            [subset[i].image_id for i in part] for part in children
                        ],
                        "reason": "deterministic nearest-first; complete-link, RGB content MAE<=8, cap6",
                    }
                )
        in_two_three = sum(len(part) for part in strict_parts if 2 <= len(part) <= 3)
        readme = (OUTPUTS / f"source/sources/{source}/README.roboflow.txt").read_text(
            encoding="utf-8"
        )
        reports[source] = {
            "images": len(subset),
            "strict_threshold": strict_threshold,
            "loose_threshold": loose_threshold,
            "nearest_hamming_histogram": histogram(map(int, near)),
            "pair_hamming_histogram": histogram(
                map(int, distance[np.triu_indices(len(subset), 1)])
            ),
            "strict_size_histogram": histogram(map(len, strict_parts)),
            "loose_size_histogram": histogram(map(len, loose_parts)),
            "strict_2_3_images": in_two_three,
            "strict_2_3_share": in_two_three / len(subset),
            "readme_versions": 3 if "3 versions" in readme else 1,
            "expectation_note": "Export-wide versions, not a guaranteed retained-subset recall denominator; hash/RGB gates under-group.",
            "largest_loose": max(map(len, loose_parts)),
            "largest_loose_share": max(map(len, loose_parts)) / len(subset),
            "raw_strict_largest": max(map(len, strict_components)),
            "oversized_components_split": sum(len(c) > 6 for c in strict_components),
        }
    require(
        len(strict_by_image) == len(loose_by_image) == len(features), "Unassigned image"
    )
    require(max(map(len, strict.values())) <= 6, "Strict family cap")
    multi = sorted((v for v in strict.values() if len(v) >= 2), key=str)
    require(len(multi) >= 6, "Need six actual strict families for visual QA")
    examples = random.Random(42).sample(multi, 6)
    return FamilyReport(
        strict, loose, strict_by_image, loose_by_image, reports, splits, examples
    )
