"""CPU-only authorization, identity and immutable per-instance input boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

import numpy as np

from public_label_rework import snap as s
from retraining_data.schema import (
    Json,
    JsonMap,
    array,
    boolean,
    document,
    integer,
    mapping,
    number,
    require,
    text,
)

ROOT: Final = s.ROOT.parent
RELEASE: Final = ROOT / ".omo/ulw-execute/task-4-roster-approval.json"
SPEC: Final = ROOT / ".omo/ulw-execute/task-4-instance-validation.json"
type Point = tuple[float, float]

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Results


def qualities(predictions: Iterable[Results | Tensor]) -> tuple[float, ...]:
    """Narrow actual toolkit results after model initialization, preserving scores."""
    from torch import Tensor
    from ultralytics.engine.results import Results

    scores: list[float] = []
    for prediction in predictions:
        match prediction:
            case Results(boxes=boxes):
                if boxes is not None:
                    scores.extend(float(v) for v in np.asarray(boxes.cpu().conf))
            case Tensor():
                raise s.SnapError("unexpected_tensor_prediction")
            case unreachable:
                assert_never(unreachable)
    return tuple(scores)


def encoded(value: Json) -> bytes:
    """Deterministic exact-byte format; forbid nonfinite floats."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def decoded(data: bytes, decoder: Callable[[bytes], Json] = json.loads) -> Json:
    """Keep JSON library typing at one explicit boundary before field parsing."""
    return decoder(data)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, value: Json) -> str:
    """Exclusive creation, never overwrite an existing evidence artifact."""
    data = encoded(value)
    with path.open("xb") as stream:
        _ = stream.write(data)
    return sha(data)


def frozen(path: Path) -> JsonMap:
    """Check the independent digest sidecar before parsing frozen inputs."""
    data = path.read_bytes()
    require(
        sha(data) == path.with_suffix(".sha256").read_text().strip(),
        f"frozen_input_tamper:{path}",
    )
    return document(data)


def seal(path: Path, value: Json) -> str:
    """Refuse either preexisting member of a frozen file/digest pair."""
    require(
        not path.exists() and not path.with_suffix(".sha256").exists(),
        f"existing_output:{path}",
    )
    digest = write_new(path, value)
    with path.with_suffix(".sha256").open("x") as stream:
        _ = stream.write(digest + "\n")
    return digest


def local(path: str, root: Path = ROOT) -> Path:
    result = (root / path).resolve()
    require(
        not Path(path).is_absolute() and result.is_relative_to(root),
        "workspace_relative_path_required",
    )
    return result


def point(value: Json) -> Point:
    values = array(value)
    require(len(values) == 2, "two_point_coordinates_required")
    return number(values[0]), number(values[1])


@dataclass(frozen=True, slots=True)
class Target:
    """Roster provenance is retained, including uncertain and null targets."""

    target_id: str
    point: Point | None
    negative_points: tuple[Point, ...]
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class RosterImage:
    image_id: str
    image_sha256: str
    width: int
    height: int
    background_potatoes: bool
    targets: tuple[Target, ...]
    context_notes: tuple[str, ...]


def images(values: Json) -> tuple[RosterImage, ...]:
    """Parse either released roster representation into one typed boundary."""
    match values:
        case dict():
            rows = array(values["images"])
        case list():
            rows = values
        case _:
            require(False, "roster_array_or_wrapper_required")
            rows = []
    result: list[RosterImage] = []
    for value in rows:
        row = mapping(value)
        targets: list[Target] = []
        width, height = integer(row["width"]), integer(row["height"])
        require(width > 0 and height > 0, "positive_image_dimensions_required")
        for raw in array(row["targets"]):
            t = mapping(raw)
            p = None if t["point"] is None else point(t["point"])
            negatives = tuple(point(v) for v in array(t["negative_points"]))
            require(
                all(
                    0 <= x < width and 0 <= y < height
                    for x, y in negatives + (() if p is None else (p,))
                ),
                "roster_point_outside_image",
            )
            tid = text(t["target_id"])
            require(
                bool(tid) and Path(tid).name == tid and "\\" not in tid,
                "unsafe_target_id",
            )
            status = text(t["status"])
            require(status in {"identifiable", "uncertain"}, "unknown_target_status")
            targets.append(Target(tid, p, negatives, status, text(t["reason"])))
        require(
            len({t.target_id for t in targets}) == len(targets),
            "duplicate_target_membership",
        )
        iid = text(row["image_id"])
        require(
            bool(iid) and Path(iid).name == iid and "\\" not in iid, "unsafe_image_id"
        )
        result.append(
            RosterImage(
                iid,
                text(row["image_sha256"]),
                width,
                height,
                boolean(row["background_potatoes"]),
                tuple(targets),
                tuple(text(v) for v in array(row.get("context_notes", []))),
            )
        )
    require(len({r.image_id for r in result}) == len(result), "duplicate_image")
    return tuple(result)


def counts(rows: tuple[RosterImage, ...]) -> JsonMap:
    targets = [t for r in rows for t in r.targets]
    return {
        "targets": len(targets),
        "uncertain": sum(t.status == "uncertain" for t in targets),
        "null_points": sum(t.point is None for t in targets),
    }


@dataclass(frozen=True, slots=True)
class Approved:
    """Verified bytes, not a draft roster or a pilot/full confirmation."""

    release_sha256: str
    original_input_hash: str
    sources: tuple[tuple[str, str], ...]
    images: tuple[RosterImage, ...]


def membership(records: Sequence[Json], roster: tuple[RosterImage, ...]) -> None:
    """Enforce one and only one input record for every frozen target."""
    expected = {
        (r.image_id, t.target_id): r.image_sha256 for r in roster for t in r.targets
    }
    seen: set[tuple[str, str]] = set()
    for raw in records:
        row = mapping(raw)
        key = text(row["image_id"]), text(row["target_id"])
        require(key in expected and key not in seen, "duplicate_or_unknown_candidate")
        require(row["image_sha256"] == expected[key], "wrong_image_association")
        seen.add(key)
    require(seen == set(expected), "missing_target_membership")
