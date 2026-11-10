# Potato / stone experiment

The experiment has run. See `RESULTS.md` for Korean results and limitations.
Both completed videos are under `outputs/baseline_video_conf045/` and
`outputs/new_video_conf045/`. The new selected checkpoint is
`outputs/selection/selected.pt`; the previous baseline was not replaced.

Run from the repository root using the existing environment:

```bash
cd /mnt/c/Users/USER/Desktop/협력/potato
export PYTHONDONTWRITEBYTECODE=1
PY=/home/sc/.cache/potato-baseline/.venv/bin/python
```

## Full baseline video first

```bash
$PY -B -m stone_experiment.video \
  --weights baseline/runs/train/weights/selected.pt \
  --confidence 0.45 \
  --output stone_experiment/outputs/baseline_video_conf045
```

The standalone command accepts the single-class baseline or a two-class model
(class 0 potato, class 1 stone). It processes every decoded source frame,
encodes H.264 with no audio, and sets **both** the output stream and encoder
codec time base to the source's `1/90000`. It preserves source frame PTS and
duration rather than quantizing onto an average-rate grid, which previously
caused duplicate DTS. `counts.json` records per-frame class counts, exact PTS,
time bases, seconds, source/checkpoint/output hashes, and environment versions.
Counts are visible detections in each frame, not unique physical objects.

Every command refuses an existing output directory. All outputs, runtime
settings, new dataset caches, and predictions belong under
`stone_experiment/outputs/`. Use a different new output path for a rerun;
there is no overwrite/regenerate flag. No command imports the original ZIP.
Baseline frames, labels, checkpoints, and evaluation directories are read-only.
Mutable library settings are isolated under `outputs/runtime/process-<pid>/`
to prevent concurrent commands from racing on the same settings JSON.
The video counter is placed over the bottom foreground guard, preserving
the visibility of top-edge partial detections and mistakes.

## Chief-owned annotations

Do not prepare or train from the train-only `stone_labels.json`. Chief merges
all reviewed splits into `stone_experiment/annotations.json`:

```json
{
  "source": "KakaoTalk_261020_161759.mp4",
  "width": 1280,
  "height": 720,
  "provenance": "agent_visual_review_not_human_expert",
  "frames": [
    {"second": 16, "split": "train", "boxes": [[500,354,588,446]], "stones": []}
  ]
}
```

The example illustrates the schema only; the actual file must contain all 63
original frames. Potato boxes and splits must exactly equal
`baseline/annotations.json`: train 40 frames / 222 boxes, val 11 / 60, test
12 / 33. Ranges stay train 0..84, val 92..112, test 120..147. Both classes
must occur in train and val. Current chief review has 16 train, 8 val, and
7 test stone **observations**, not that many distinct stones.

Preparation validates source and annotation hashes and provenance against
`baseline/internal/manifest.json`. The old manifest lacks PNG hashes:
therefore each `baseline/frames` PNG must hash-identically to its verified
`baseline/internal/images/<split>` copy. The new manifest records those hashes,
timestamps, counts, label hashes, YAML hash, and a frozen annotation snapshot.
All later commands verify that snapshot and the dataset files before use.

## Prepare, train, select

Only run preparation after chief has merged all splits:

```bash
$PY -B -m stone_experiment.prepare
$PY -B -m stone_experiment.train
$PY -B -m stone_experiment.select
```

Defaults are `outputs/dataset`, `outputs/train`, `outputs/selection`.
Use `--dataset`, `--training`, or `--output` to choose other new experiment
paths. Training requires the unchanged `config/train.json` preset and compares
it with `baseline/config/train.json`. Initialization is the root, fresh
COCO-pretrained `yolo11s.pt`, never the old single-class checkpoint.
The preset remains SGD lr0=0.01, seed 42, imgsz 640, batch 8, epochs 60,
patience 15, workers 0, device 0, with all other baseline parameters unchanged.
Training records initialization/config/dataset/annotation hashes and runtime
environment. AMP's auxiliary downloads, if needed, go to new runtime weights.

Selection opens a fresh model for each native `best.pt` and `last.pt`.
It uses **val only**, confidence 0.45, NMS IoU 0.7, and matching IoU 0.5.
Choose maximum potato/stone macro F1, tie higher potato recall, then native best.
The copied `outputs/selection/selected.pt` and `selection.json` record the
criterion, candidate hashes/counts, and validation per-frame audits.
No test or external result influences selection.

## Evaluate and compare

```bash
$PY -B -m stone_experiment.evaluate --split val \
  --output stone_experiment/outputs/evaluation_val
$PY -B -m stone_experiment.evaluate --split test \
  --output stone_experiment/outputs/evaluation_test
$PY -B -m stone_experiment.evaluate --split external \
  --output stone_experiment/outputs/evaluation_external
$PY -B -m stone_experiment.compare --external
$PY -B -m stone_experiment.video \
  --weights stone_experiment/outputs/selection/selected.pt \
  --confidence 0.45 --output stone_experiment/outputs/new_video_conf045
```

`compare` scores the new selected model and
`baseline/runs/train/weights/selected.pt` on the **exact same new val/test
annotation snapshot**, both at confidence 0.45. Baseline checkpoint identity
is checked against chief's frozen `review/protocol.json` when present.
This uses prediction only: no Ultralytics validation caches or old evaluation
outputs are touched. `--external` optionally adds the prepared external set.

Matching sorts predictions by descending confidence (ties by original index).
Each prediction matches the highest-IoU unused truth regardless of class,
provided IoU >= 0.5 (ties by truth index). A wrong-class match consumes both
boxes: GT stone predicted potato increments `stone_as_potato`, potato FP,
and stone FN; the reverse increments `potato_as_stone`, stone FP, and potato
FN. Unmatched truth is a miss; unmatched prediction is background FP.
Per-class TP/FP/FN, precision, recall, F1, and two-class macro F1 include
these cross-class errors. The confusion matrix rows are truth and columns
prediction, ordered potato, stone, background. An absent class has F1 0.

Detailed frame JSONs contain image identity/hash, truth and prediction xyxy,
class IDs, confidences, matched indices and IoUs, missed truth indices,
and background prediction indices. Aggregate JSONs contain raw counts as well
as rates and hashes, so chief can audit observations visually.

External labels contain **only potato truth**. External evaluation filters
predictions to class 0 **before matching** and scores only potato. Stone
precision/recall are not scored, missing stone truth is not treated as a
background error, and external macro F1 (if displayed) is potato-only.
The existing nine-decimal normalized labels can reconstruct image edges a
fraction of a micro-pixel outside the image; only this rounding tolerance
(at most `max(width,height) * 1e-9` pixels) is clipped in memory.
Operational fixed-threshold metrics are central; no AP tuning is performed.

Chief's independent 0.45 reference found zero stone-as-potato matches on both
val and test. Do not assume a confusion reduction is possible or claim one
without the new comparison observing it. Assess explicit stone detection
and its impact on potato precision/recall as separate outcomes.

## Interpretation and CPU verification

Labels are uncertain AI visual review, not human expert or physical
confirmation. The long gray-blue object around val seconds 110/112 is among
the uncertain visual labels. The footage is a single scene, temporally
correlated, and holdouts have already been viewed. Results are exploratory,
not evidence of generalization to another capture session. There is no soil
class yet, no calibrated mm-size claim, tracking, or unique-object counting.

```bash
$PY -B -m stone_experiment.smoke
/home/sc/.cache/potato-baseline/.venv/bin/ruff check --no-cache stone_experiment
/home/sc/.cache/potato-baseline/.venv/bin/basedpyright \
  --pythonpath /home/sc/.cache/potato-baseline/.venv/bin/python stone_experiment
```

The smoke proof is CPU-only: matching/counting, annotation parsing, output
collision/refusal, CLI imports, and syntax. It never loads a model, prepares
training data, runs inference, trains, or renders the source video.
It also verifies that the shared externally initialized argparse namespace
does not override each command's own output default.
Final syntax, Ruff, CPU smoke, dataset checks and independent 300-case metric
oracle passed. Basedpyright reports zero errors and 16 unsuppressed strict
warnings, exit code 1; this diagnostic is not claimed clean.
