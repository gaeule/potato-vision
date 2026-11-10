"""Deterministic CPU behavior proofs using actual assignment and source times."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from retraining_data.schema import DataError, require
from retraining_experiment.video import TIME_BASE
from stone_experiment.metrics import Detection
from tracking_experiment.cache import OUTPUTS
from tracking_experiment.cli import arguments
from tracking_experiment.model import Frame, Settings, center
from tracking_experiment.smoke_video import integration_proof
from tracking_experiment.tracker import Tracker


def rejected[T](action: Callable[[], T], error: type[Exception] = DataError) -> None:
    try:
        _ = action()
    except error:
        return
    raise DataError(f"Expected {error.__name__}")


def frame(pts: int, centers: tuple[tuple[float, float], ...] = ()) -> Frame:
    return Frame(
        pts // 3750,
        pts,
        TIME_BASE,
        3750,
        tuple(Detection(0, (x - 20, y - 20, x + 20, y + 20), 0.9) for x, y in centers),
    )


def directed_once_proof() -> None:
    # Given: one potato crosses upward, reverses, then crosses again.
    tracker = Tracker(Settings())
    positions = (390, 370, 359, 354, 351, 370, 390, 340)
    # When: all actual observations are replayed.
    outputs = [
        tracker.update(frame(i * 9000, ((500, y),))) for i, y in enumerate(positions)
    ]
    events = [event for _, crossing in outputs for event in crossing]
    # Then: hysteresis confirms at frame four, exactly once for this ID.
    require(len(events) == 1 and events[0].pts == 36000, "Once after hysteresis")
    require(events[0].track_id == 1 and events[0].crossing_x == 500, "Crossing ID/x")
    require(0.1 < events[0].crossing_second_estimate < 0.2, "Interpolated crossing")
    require(all(o.track_id == 1 for obs, _ in outputs for o in obs), "Stable ID")


def band_and_birth_proof() -> None:
    # Given: separate tracks born downstream or in the band, and upstream jitter.
    tracks = [Tracker(Settings()) for _ in range(3)]
    sequences = ((350, 340, 330), (365, 359, 350), (375, 359, 361, 354, 359))
    # When: none has a confirmed upstream-to-downstream traversal.
    events = [
        event
        for tracker, ys in zip(tracks, sequences, strict=True)
        for i, y in enumerate(ys)
        for event in tracker.update(frame(i * 9000, ((500, y),)))[1]
    ]
    # Then: births and jitter produce no event.
    require(not events, "Downstream/band birth and jitter must not count")


def finite_line_proof() -> None:
    # Given: segments whose interpolated x is inside, on, or outside endpoints.
    cases = ((210, 230, 1), (200, 210, 0), (990, 1010, 1), (1001, 1010, 0))
    # When: each potato traverses the full hysteresis band.
    actual: list[int] = []
    for x0, x1, _ in cases:
        tracker = Tracker(Settings())
        _ = tracker.update(frame(0, ((x0, 380),)))
        actual.append(len(tracker.update(frame(9000, ((x1, 340),)))[1]))
    # Then: finite segment interpolation, including inclusive endpoints, decides.
    require(actual == [expected for _, _, expected in cases], "Finite-line geometry")


def direction_and_class_proof() -> None:
    # Given: downward configuration with a potato and a stone.
    tracker = Tracker(Settings(direction="down"))
    first = frame(0, ((500, 330),))
    stone = Detection(1, (580, 310, 620, 350), 0.9)
    _ = tracker.update(replace(first, detections=(*first.detections, stone)))
    # When: the potato moves downward across the line.
    observations, events = tracker.update(frame(9000, ((500, 390),)))
    # Then: only the potato has an ID and a downward event.
    require(len(observations) == len(events) == tracker.created == 1, "Potato only")
    require(events[0].direction == "down", "Configured downward direction")


def dropout_proof() -> None:
    # Given: one established upstream ID.
    tracker = Tracker(Settings())
    _ = tracker.update(frame(0, ((500, 400),)))
    _ = tracker.update(frame(9000, ((500, 380),)))
    # When: an empty frame precedes a crossing observation exactly 0.5 seconds later.
    absent, predicted_events = tracker.update(frame(36000))
    observed, events = tracker.update(frame(54000, ((500, 280),)))
    # Then: gaps never fabricate observations/events, and inclusive retention keeps ID.
    require(not absent and not predicted_events, "No predicted-box crossings")
    require(observed[0].track_id == 1 and len(events) == 1, "Time-based retention")
    require(events[0].observation_gap_seconds == 0.5, "Exact source gap evidence")


def expired_proof() -> None:
    # Given: a track that has been absent for longer than max_gap.
    tracker = Tracker(Settings())
    _ = tracker.update(frame(0, ((500, 400),)))
    _ = tracker.update(frame(9000, ((500, 380),)))
    # When: a downstream observation returns one source tick beyond retention.
    observations, events = tracker.update(frame(54001, ((500, 280),)))
    # Then: it receives a new unarmed ID rather than bridging a long dropout.
    require(observations[0].track_id == 2 and not events, "Expired ID is not reused")


def motion_proof() -> None:
    # Given: two objects approach, coincide, then pass one another.
    tracker = Tracker(Settings())
    for pts, xs in ((0, (400, 600)), (9000, (450, 550)), (18000, (500, 500))):
        _ = tracker.update(frame(pts, tuple((x, 400) for x in xs)))
    # When: final detections are ordered opposite to their established velocities.
    observations, _ = tracker.update(frame(27000, ((450, 400), (550, 400))))
    # Then: motion, not input order or nearest last center, preserves identities.
    require(
        [(o.track_id, center(o.xyxy)[0]) for o in observations] == [(1, 550), (2, 450)],
        "Motion-based one-to-one identities",
    )


def global_assignment_proof() -> None:
    # Given: the closest choice for ID 1 would steal ID 2's only valid match.
    tracker = Tracker(Settings())
    _ = tracker.update(frame(0, ((400, 400), (480, 400))))
    # When: global assignment considers both tracks and explicit unmatched dummies.
    observations, _ = tracker.update(frame(9000, ((330, 400), (410, 400))))
    # Then: it retains two valid matches rather than greedy fragmentation.
    require(
        [(o.track_id, center(o.xyxy)[0]) for o in observations] == [(1, 330), (2, 410)],
        "Global valid assignment",
    )


def deterministic_proof() -> None:
    # Given: ambiguous equal-cost observations in reversed input orders.
    first, second = Tracker(Settings()), Tracker(Settings())
    detections = frame(0, ((450, 400), (550, 400))).detections
    _ = first.update(replace(frame(0), detections=detections))
    _ = second.update(replace(frame(0), detections=tuple(reversed(detections))))
    # When: the same tied boxes are presented in opposite orders again.
    boxes = frame(9000, ((500, 390), (500, 410))).detections
    a = first.update(replace(frame(9000), detections=boxes))
    b = second.update(replace(frame(9000), detections=tuple(reversed(boxes))))
    # Then: identical canonical assignment inputs yield identical IDs and states.
    require(a == b, "Input-order-independent assignment ties")


def confirmation_proof() -> None:
    # Given: a crossing before the configured third hit.
    tracker = Tracker(Settings(min_hits=3))
    _ = tracker.update(frame(0, ((500, 390),)))
    second, premature = tracker.update(frame(9000, ((500, 340),)))
    # When: another actual downstream observation confirms the candidate.
    third, events = tracker.update(frame(18000, ((500, 320),)))
    # Then: only the third observation emits a confirmed event.
    require(not second[0].confirmed and not premature, "Tentative crossing deferred")
    require(third[0].confirmed and len(events) == 1, "Hit threshold confirmation")


def boundary_proof() -> None:
    # Given: a configured tracker and invalid boundary inputs.
    tracker = Tracker(Settings())
    _ = tracker.update(frame(9000))
    # When/Then: time reversal, nonfinite settings and invalid ranges are rejected.
    rejected(lambda: tracker.update(frame(9000)))
    rejected(lambda: Settings(max_gap=0))
    rejected(lambda: Settings(line_y=float("nan")))
    rejected(lambda: Settings(x_min=1000, x_max=220))
    rejected(lambda: Settings(min_hits=1))
    options = arguments(
        [
            "--line-y",
            "300",
            "--x-min",
            "100",
            "--x-max",
            "900",
            "--direction",
            "down",
            "--max-gap",
            "0.7",
        ]
    )
    require(
        options.settings
        == Settings(
            line_y=300,
            x_min=100,
            x_max=900,
            direction="down",
            max_gap=0.7,
        ),
        "CLI overrides reach the tracking settings",
    )


def main() -> None:
    for proof in (
        directed_once_proof,
        band_and_birth_proof,
        finite_line_proof,
        direction_and_class_proof,
        dropout_proof,
        expired_proof,
        motion_proof,
        global_assignment_proof,
        deterministic_proof,
        confirmation_proof,
        boundary_proof,
    ):
        proof()
        print(f"PASS {proof.__name__}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cpu-smoke-", dir=OUTPUTS) as directory:
        integration_proof(Path(directory))
    print("PASS real CLI, cache rejection, new-only outputs and VFR encode/decode")


if __name__ == "__main__":
    main()
