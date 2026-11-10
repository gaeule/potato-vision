# Cached potato tracking

This package replays the existing `retraining_experiment/outputs/video/counts.json`.
It never loads weights, runs inference, retrains, installs packages, or imports
Ultralytics at runtime. The checkpoint and existing annotated MP4 are opened only
to verify their SHA-256 hashes against the detection cache.

## Run

Run from the workspace root in a fresh process:

```bash
cd /mnt/c/Users/USER/Desktop/협력/potato
PY=/home/sc/.cache/potato-baseline/.venv/bin/python
$PY -B -m tracking_experiment.video \
  --line-y 360 --x-min 220 --x-max 1000 --direction up --hysteresis 8 \
  --output tracking_experiment/outputs/video
```

The chief owns the whole-source run, visual verification and manual comparison.
Every output path must be new and strictly below `tracking_experiment/outputs`;
existing paths and symlinks are refused. A failed run may leave partial artifacts:
use a fresh output path rather than overwriting it. `review/**` is chief-owned.

Use `--help` for all options. `--source`, `--cache`, `--weights`, and
`--cached-video` must describe the same cached run. Defaults use the original
1280x720 source, selected v2 checkpoint, and v2 full-video artifacts.
`--line-y`, `--x-min`, `--x-max`, and `--direction {up,down}` configure the finite
horizontal line. Coordinates are original-source pixels, not resized preview
coordinates. Defaults come from the chief's source-video review, not manual counts.

## Semantics

Only cached class 0 (potato) boxes enter tracking; cached stone counts remain
separate on-screen observations. Detection confidence 0.45 and NMS IoU 0.5 stay
unchanged. Empty source frames participate in the replay.

IDs are assigned in canonical box order. Tracks are associated globally one to
one using installed SciPy's Hungarian assignment, predicted-center distance plus
predicted-box IoU, and unmatched dummy columns. Track order is ascending ID and
box order is xyxy then descending confidence, fixing assignment tie inputs.
Boxes with more than 4x or less than 1/4x the last area cannot match.
The center prediction uses actual elapsed source PTS and smoothed observed
velocity; the second observation initializes velocity. Default
`--max-distance 100` is the maximum predicted-center residual in pixels.
Default `--max-gap 0.5` retains an unmatched ID for at most 0.5 source seconds
(inclusive), independent of the variable frame rate. No predicted box is drawn
or counted during a detection dropout.

A track must be observed at least `--min-hits 2` times to be confirmed.
To count upward it must first be observed with its center at or below y=368,
then cross the center line inside x=220..1000 inclusive and reach y<=352.
Downward reverses that rule. A track born downstream or inside the hysteresis
band is not immediately armed. Reversing back before confirmation clears the
crossing candidate. Finite-line membership uses the interpolated x at y=360,
not merely the endpoint x. A short dropout may bridge a crossing using the
segment between actual observations. A gap longer than the retention limit
starts a new ID. Each ID emits at most one event, even after recrossing.

`events.json` records the confirming source frame's exact PTS/time base and an
explicitly estimated crossing second, plus the two actual observation PTS that
bracket the center crossing and their gap. Use `second` for frame/overlay
alignment, or `crossing_second_estimate` for estimated line-crossing intervals;
choose the same timestamp convention when comparing manual intervals.
Confirmation can occur later than the center crossing because of hysteresis.

## Outputs and verification

- `annotated.mp4`: original source pixels with observed potato IDs (`?` means
  tentative), visible finite line, hysteresis boundaries, direction arrow,
  separate on-screen potato/stone observations and cumulative directed events.
- `tracks.json`: every source frame, original PTS/time base/duration, observed
  IDs/boxes/confidence/confirmation state, retained unobserved IDs and cumulative
  count. Retained IDs have no invented box observations.
- `events.json`: once-per-ID finite-line crossing evidence.
- `summary.json`: configuration, counts, input/cache/output hashes, code hashes,
  dependency versions and successful full-decode timing verification.

The loader verifies source/checkpoint/cached-video hashes, detector protocol,
frame order, integer PTS/durations, matching 90 kHz clocks, finite in-bounds
boxes, confidence/classes and per-frame class totals. Rendering pairs every
cache frame with every decoded original source frame and rejects timing
mismatches. H.264 encoding reuses the existing `bf=0` and packet-duration
helpers. The completed output is decoded again to check every PTS, duration
(including the final interval), frame count, codec and no-audio condition.

## CPU checks

```bash
PY=/home/sc/.cache/potato-baseline/.venv/bin/python
$PY -B -m tracking_experiment.smoke
/home/sc/.cache/potato-baseline/.venv/bin/ruff check --no-cache tracking_experiment
/home/sc/.cache/potato-baseline/.venv/bin/ruff format --check --no-cache tracking_experiment
/home/sc/.cache/potato-baseline/.venv/bin/basedpyright \
  --pythonpath "$PY" tracking_experiment
```

Smoke tests use deterministic source times, actual SciPy assignment, temporary
owned outputs and real CPU H.264 encode/decode. They do not run inference or
the real full-source tracking job. No pytest installation is needed.
The existing venv's basedpyright CLI reports 0 errors and 30 warnings, returning
exit code 1 because SciPy has no installed type stubs and PyAV's stream/packet
types are incomplete. No warning suppression or dependency installation is used.
The editor LSP is unavailable; use the CLI above.

These are track IDs, not guaranteed physical identities. Occlusion, touching
potatoes, velocity ambiguity, detector duplicates and dropouts can split or
switch IDs and change event counts. Cache-only replay cannot recover detections
below the existing confidence threshold. Confirmation rejects single-frame
tracks, but a persistent false positive can still cross and count. The chief's
independent manual source intervals `[30,40)`, `[75,85)`, `[124,137)` are for
comparison, not parameter tuning.
