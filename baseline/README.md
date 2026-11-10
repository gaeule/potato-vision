# Independent potato YOLO11s baseline

Run from the repository root. Nothing below starts training until the explicit
`run_baseline train` command. The source MP4 and ZIP are read-only inputs.
The ZIP is **never a training source**.

## Environment

Prepared interpreter: `/home/sc/.cache/potato-baseline/.venv/bin/python`
(Python 3.12). Exact initial package pins are in `baseline/requirements.txt`.
No install is needed in the prepared environment. For a fresh environment:

```sh
UV=/tmp/potato-tooling/bin/uv
$UV venv --python 3.12 /home/sc/.cache/potato-baseline/.venv
$UV pip install --python /home/sc/.cache/potato-baseline/.venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'
$UV pip install --python /home/sc/.cache/potato-baseline/.venv/bin/python \
  -r baseline/requirements.txt
```

Use the following variable for every command below:

```sh
PY=/home/sc/.cache/potato-baseline/.venv/bin/python
```

CUDA device 0 is required for training, evaluation, and inference. Framework
versions are checked before these commands run. The first training command
downloads the official `yolo11s.pt` pretrained weights if absent; internet access
is needed then. Preserve those weights and the generated environment records.
Fixed seeds/deterministic mode improve repeatability but do not promise identical
floating-point results across GPU, driver, or dependency versions. Requirements
pin direct packages, not every transitive dependency.

## Source extraction, without annotations

```sh
$PY -m scripts.extract_frames --seconds 0 6 10 12 --output baseline/future_frames
```

This sequentially decodes the source; it does not seek to approximate positions.
For each requested integer second it selects the first decoded timestamp greater
than or equal to that second and saves `frame_000012.png`, for example.
`extraction_manifest.json` records actual PTS, time base, dimensions, and source
SHA256. Existing files are refused. `--regenerate` permits overwriting only frames
listed in this extractor's manifest with the same source hash. Existing chief
frames in `baseline/frames` without that manifest are deliberately protected.
An incomplete extraction reports an error and leaves partial output for inspection.

## Internal annotations and dataset

The chief supplies `baseline/annotations.json`, never modified by these scripts:

```json
{
  "source": "KakaoTalk_261020_161759.mp4",
  "width": 1280,
  "height": 720,
  "provenance": "agent_visual_review_not_human_expert",
  "frames": [
    {"second": 12, "split": "train", "boxes": [[100, 100, 200, 200]]},
    {"second": 92, "split": "val", "boxes": []},
    {"second": 120, "split": "test", "boxes": []}
  ]
}
```

Coordinates are pixel xyxy. All boxes become class `0: potato`; empty arrays
produce empty label files. At most 70 unique seconds are accepted. Every split
must be represented. Allowed seconds: train 0..84, val 92..112, test 120..147,
leaving eight-second boundary gaps. The initial chief plan is 40/11/12 frames:
train `[0,6,10] + list(range(12,85,2))`, val `range(92,113,2)`, test
`list(range(120,139,2)) + [142,146]`.

```sh
$PY -m scripts.prepare_data internal
```

Outputs: `baseline/internal/data.yaml`, `images/{train,val,test}`,
`labels/{train,val,test}`, `decoded/`, and `manifest.json`. Preparation decodes
the source again into its own artifacts to guarantee timestamps and pixel
correspondence; it never overwrites the chief's annotation-reference frames.
The manifest hashes both source and annotations and records actual frame times,
splits, and counts. Rebuilding requires explicit `--regenerate`; only
pipeline-marked output directories are eligible. No command deletes originals.

## External evaluation-only ZIP import

```sh
$PY -m scripts.prepare_data external
```

The implementation smoke check has already imported the supplied archive in
this workspace. To explicitly recreate those owned artifacts, run:

```sh
$PY -m scripts.prepare_data external --regenerate
```

Outputs: `baseline/external/data.yaml`, `images/test`, `labels/test`, and
`manifest.json`. The importer validates flat unique filenames, exact PNG/JSON
pairs, image dimensions/imagePath, finite nondegenerate rectangles, and class
`potato`. Two opposite points or four axis-aligned corners are accepted.
Empty supplied labels are preserved. The one known x1 near -0.1353767 in
`extra2_t000151000ms_f002265.json` is clamped to zero and logged in the manifest
and command's JSON output. Other out-of-bounds boxes fail.

External YAML includes a syntactically legal `train: images/test` because the
loader requires a train key; it is **evaluation-only**, not permission to train
on it. The supplied ZIP annotations are not expert gold. The training command
has no external dataset argument and never opens the archive.

## Train, then evaluate without holdout tuning

```sh
$PY -m scripts.run_baseline train
$PY -m scripts.select_checkpoint
$PY -m scripts.run_baseline evaluate --dataset val
$PY -m scripts.run_baseline evaluate --dataset test
$PY -m scripts.run_baseline evaluate --dataset external
```

`baseline/config/train.json` pins YOLO11s, 640 pixels, seed 42, deterministic
mode, 60 epochs, patience 15, batch 8, workers 0, device 0, and explicit SGD
settings. Native best.pt uses internal validation AP. For operational deployment,
`select_checkpoint` compares native best and last on internal validation ONLY
at fixed confidence 0.25 and matching IoU 0.5, then saves the highest-F1 model as
`baseline/runs/train/weights/selected.pt`. No threshold is fitted on holdouts.
Ultralytics keeps its `args.yaml`, `results.csv`, and plots there; the pipeline
also writes `environment.json`, `input.json`, and `checkpoint.json`.

Evaluation always reopens that same selected checkpoint. No threshold or model
selection uses internal test or external scores. Each command writes
`baseline/evaluations/{val,test,external}/metrics.json`, plots, and
`predictions/` images/labels. Metrics include precision, recall, mAP50, mAP50-95,
timings, environment versions, and hashes of checkpoint and dataset manifest.
AP uses confidence 0.001 to retain the precision-recall curve; **visualizations**
use fixed confidence 0.25 and NMS IoU 0.7. These are not tuned on holdouts.
Library precision/recall summarize each dataset at its best-F1 operating point;
they are NOT precision/recall at the displayed confidence 0.25.
`fixed_confidence_metrics.json` records actual displayed-condition TP/FP/FN,
precision, recall, and F1 using confidence-ranked one-to-one IoU>=0.5 matching.
Existing training/evaluation/inference output directories are refused; move
them aside explicitly before a separate run.

## Full-source H.264 visualization

```sh
$PY -m scripts.run_baseline video
```

Outputs: `baseline/inference/annotated.mp4` and `counts.json`. Every source frame
is processed at confidence 0.25, NMS IoU 0.7, and 640 pixels. PyAV/libx264 encodes
H.264/yuv420p MP4; codec and stream explicitly use source time_base=1/90000
to preserve variable-frame-rate timestamps without average-rate rounding; no OpenCV writer is used.
No audio stream is added (the supplied source has none). The overlay says
`ON SCREEN`, meaning visible detections in that frame, **not unique count**.
JSON records each frame timestamp/count plus source/checkpoint hashes.

## Verification and limitations

```sh
$PY -m compileall -q scripts
$PY -m ruff check scripts
```

This is a single-video baseline with correlated frames despite temporal gaps.
The chief's `baseline/review/duplicate_audit.json` reports no decoded-RGB exact
matches between the 63 selected source frames and 166 external frames. Four
nearest positive pairs were visually different in placement/framing. This does
not establish different capture sessions or different physical potatoes.
Internal annotations are an agent's visual review, not expert annotations;
external supplied labels may also be imperfect. There is one potato class,
no stone class, no physical-size estimation, no tracking, and no unique-potato
counting. Detection counts can duplicate an object across consecutive frames.
Do not present these holdout metrics as evidence of broad production accuracy.
Training completed at epoch 36 (early stopping). Native AP-best is epoch 21;
fixed-confidence validation F1 selected the epoch-36 last checkpoint.
Initial exploratory predictions revealed excessive false positives, prompting
this operational selection. Only validation determines the choice, but the
external set has been viewed: treat its score as exploratory, not pristine
acceptance evidence. Initial outputs remain in evaluations_initial_ap and
inference_initial_ap. See RESULTS.md for final observed results.
Implementation checks exercised synthetic-video sequential frame selection,
YOLO conversion including empty labels, overwrite refusal, explicit
regeneration, and actual libx264 H.264/no-audio encode/decode. The real ZIP
import contained 166 pairs, 192 boxes, 87 empty labels, and one correction.
Training, selected-checkpoint evaluation, and full-source inference have run.
Syntax/lint and real-data checks passed. Basedpyright reports zero errors but
146 strict warnings (dynamic JSON/argparse and third-party API typing among
others), exit code 1; this check is NOT claimed clean and no warnings are
suppressed. Regressions verify retained frame ownership, 63 labels surviving
a failed regeneration, and duplicate/background false-positive matching.
