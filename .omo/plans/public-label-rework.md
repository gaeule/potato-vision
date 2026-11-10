# public-label-rework - Work Plan

## TL;DR (For humans)

**Who this is for and what changes for them:** The data provider who trains a potato size-and-count model gets the 2,060 public photos back with boxes that hug each potato, exactly one box per potato, and a clear list of photos set aside, so he can drop the folder straight into his training without the problems he flagged last time.

**What you'll get:** A corrected copy of the public labels in the same format and file names, a list of set-aside photos with reasons, a separate folder of photos that still need a person's look, a change log with before/after pictures, measured quality numbers, and a Korean guide including license credits.

**Why this approach:** Most of the bad boxes come from how the public images were rotated when they were built, so boxes are redrawn by measuring each potato's outline automatically rather than by hand; AI reviewers only judge what they see, and one fixed program applies their decisions so the result is repeatable. A 100-photo pilot is shown to you first, and work stops until you confirm.

**What it will NOT do:** It does not touch the original ZIP, the provider's own field photos, or any image pixels; it deletes nothing; it never claims a person checked the labels.

**Effort:** XL
**Risk:** Medium - quality depends on how well AI reviewers handle dense piles; the pilot measures this before the full run.
**Decisions to sanity-check:** all 2,060 photos are kept and corrected one by one; blurry background potatoes are left unboxed but marked; the pass/fail reference answers are made by AI reviewers, not people.

Your next move: approve the plan, then start execution separately. Full execution detail follows below.

---

> TL;DR (machine): XL, medium risk; corrected images/public X-AnyLabeling labels + exclusions + needs_human + change log + metrics + Korean README/ZIP, pilot-gated.

## Scope
### Affected user and ideal state
**Affected user:** (U1) the data provider 손원섭, who trains a potato size/quantity detector on `images/harvest` + `images/public` and opens folders in X-AnyLabeling v4.2.0; today the public labels are loose, shifted, duplicated and incomplete, and he already rejected another team's attempt for exactly those reasons; after, he drops a corrected `images/public` folder plus exclusion and needs-human lists into his pipeline. (U2) his training pipeline, a program that reads X-AnyLabeling JSON (`shapes[].label`, 4-point `rectangle`, `imagePath`, `imageWidth/Height`). (U3) our team (the user), who must be able to explain and trust every change. (U4) the reviewer agents that judge rendered proposals; they need renders dense enough to judge.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | Every in-focus, identifiable potato in a kept image has exactly one box | Provider feedback: duplicates and misses degrade training |
| IS-2 | Every delivered box is tight to the visible potato outline (SAM-snapped), including the tail of the distribution, not only the median | Provider feedback: loose boxes inflate size estimates |
| IS-3 | Images whose potato boundaries cannot be determined are excluded with a reason; images still holding an unresolved or untight box go to a separate needs-human set; no kept image is partially labelled | Provider allows removal; partial labels teach potatoes as background |
| IS-4 | Output keeps the same schema (only `flags` and `shape.description` additions), filenames, folder layout and byte-identical images; the README states GUI opening was not tested | Drop-in replacement for U1 and U2 |
| IS-5 | `final-dataset.zip` and every pre-existing workspace file stay untouched; a per-box change log and before/after renders for every changed image exist | Revertible and explainable for U3 |
| IS-6 | Provenance is honest: AI visual review, `checked:false`, `flags.human_reviewed:false`, no human-gold claim, per-image status recorded | Project norm in `label_review/protocol.json` |
| IS-7 | The harvest subset is untouched (read in memory only for a convention report) | Provider states harvest labelling is complete |
| IS-8 | Augmented copies of the same scene receive consistent decisions, every mismatch carries an allowed explanation code | Inconsistent labels across flips/rotations confuse training |
| IS-9 | Delivery happens only after validators pass, held-out gold thresholds pass, and the user confirmed the pilot | Avoid a second rejection |
| IS-10 | The public box convention (visible extent, sprouts excluded, background rule) is compared with the harvest labels and any deviation is documented | U1 trains on both subsets together |
| IS-11 | The delivery credits each CC BY 4.0 source, states that labels were modified, and flags watermarked stock photos and stretch/letterbox preprocessing | Provider compliance and correct use of pixel sizes |
| GAP-1 | Dense-pile images label only a few of many potatoes (e.g. `public_original01_0069`, `public_original06_0161/0162/0150`, `public_original07_0296/0300/0114/0024`) | Breaks IS-1 |
| GAP-2 | Boxes oversized/shifted: original02 (1,343 images) used Fit-black-edges + 90deg + random +-8deg rotation (rotated envelopes, median ~14%, max ~28% larger), and also has loose unrotated (`_0077`) and shifted (`_0177`) boxes | Breaks IS-2 |
| GAP-3 | 40 public images contain overlapping box pairs (34 IoU>=0.5, 44 containment>=80%) | Breaks IS-1 |
| GAP-4 | Some composite images remain (provider: 'a few left') | Breaks IS-3 |
| GAP-5 | No per-image status, change log or provenance exists | Breaks IS-5, IS-6 |
| GAP-6 | Augmented copies are unidentified (READMEs: 3 versions per source image for original01/02/03/07; a naive hash grouping chained 548 images) | Breaks IS-8 |
| GAP-7 | No tightness measurement, reference set or honeypot set exists | Blocks proving IS-2 and IS-9 |
| GAP-8 | No delivery package, README, exclusion or needs-human list exists | Breaks IS-4, IS-9 |
| GAP-9 | Harvest box convention unknown; existing public boxes include sprouts (`public_original02_0144`, `public_original06_0058`); background piles behind subjects unlabelled (`public_original06_0018`) | Breaks IS-10, IS-1 |
| GAP-10 | No license/modification notice; watermarked stock photos (`public_original02_0440` shutterstock footer, `public_original01_0065` Alamy bar) unflagged | Breaks IS-11 |

### Must have
- All work in a new package folder `public_label_rework/` at the workspace root `/mnt/c/Users/USER/Desktop/협력/potato`; all outputs in new-only subfolders of `public_label_rework/outputs/` (refuse existing paths, pattern `retraining_experiment/common.py:new_output`).
- Single class `potato`. Box policy: visible extent only; occluded potatoes boxed by visible part when the outline is identifiable; edge-clipped potatoes boxed by visible extent; sprouts excluded (tuber body only); separate cut pieces boxed separately; other produce in source 01 left unlabelled.
- D1: label every in-focus potato whose outline is identifiable; an image where an in-focus potato's boundary cannot be determined is excluded whole (all-or-nothing).
- D7 background rule: out-of-focus background potatoes and potatoes smaller than 16 px on the long side are context, not boxed; the image is kept and gets `flags.background_potatoes: true`; counts reported in README.
- D2: all 2,060 images are kept and corrected individually; augmentation families are used only for consistency checks and fold grouping; exclusion only for IS-3 reasons.
- D3: SAM2.1-b (`sam2.1_b.pt`) downloaded once through installed Ultralytics into `public_label_rework/weights/`, SHA-256 recorded.
- D4: pilot of 100 images end to end, then STOP for explicit user confirmation before any other image is processed.
- D5: gold set of 100 images built by two independent AI labelers plus an AI adjudicator, SAM-snapped, `shape.description` contains `agent_gold_not_human`, built before pipeline outputs are inspected; gold labels never shown to pipeline agents. Roles: G-pilot (50) may drive at most one recorded rubric iteration; G-full (50) is the untouched delivery gate.
- Geometry is produced by tools (detector, SAM); review agents only judge rendered proposals; any agent-authored point or coordinate is snapped by SAM before use. Tightness comes only from SAM, never from detector agreement.
- Output JSON: top-level keys and order exactly as the source document; top-level `checked` stays `false`; additions only inside top-level `flags` (`human_reviewed:false`, `ai_reviewed:true`, `background_potatoes:bool`, `rework_status:"kept"`) and inside each shape (`flags.tight:true`, `description` = provenance like `src=sam;batch=B017`); `score` kept exactly as in source (absent in public) and never added.
- Delivery layout: `images/public/` (kept only), `needs_human/` (JPG + JSON, outside images/public), `exclusions.csv`; kept + excluded + needs_human = 2060.
- Deterministic apply: same inputs produce byte-identical outputs.
- CC BY 4.0: `sources/` copied unchanged and README section "라이선스 및 출처" lists each source name, URL, license and a modification notice.
- Korean CSVs written as `utf-8-sig`; ZIP entry names ASCII.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No edit, move or re-zip of `final-dataset.zip`; no extraction of `images/harvest` to disk (in-memory read for the convention report only); harvest overlap candidates `harvest_04_004`, `harvest_05_020`, `harvest_05_036` only reported.
- No change to image pixels, sizes or filenames; letterbox bars stay, boxes are clipped to content.
- No file deletion; excluded and needs-human images are listed/separated, not removed from the source copy.
- No new classes, no stone labels in public, no new top-level JSON keys.
- No `human_reviewed:true`, no claim of human-verified gold.
- No package installs and no network downloads except SAM2.1-b weights (training runs with `amp=False` so the AMP check downloads nothing).
- No edits to any pre-existing workspace folder (`baseline/`, `stone_experiment/`, `retraining_*`, `tracking_experiment/`, `label_review*/`, `data/`, `scripts/`, `postprocess_review/`, `runs/`, `weights/`).
- No message to the provider; the user delivers.
- No processing beyond the pilot until the user's confirmation file exists.
- No tuning on G-full; at most one recorded rubric iteration on G-pilot.
- No image with a `tight:false` box or a needs-human status inside `images/public`.
- No acceptance criterion phrased as "reported", "expected" or "red text"; every criterion names a threshold or a PASS/FAIL token.

## Verification strategy
> Zero human intervention - all verification is agent-executed. Todo 14 is the D4 scope gate (a user decision), not a QA criterion.
- Test decision: tests-after. Framework = repository-style executable CPU smoke modules (`python -B -m public_label_rework.smoke`), Given/When/Then comments, no pytest (not installed; see `tracking_experiment/smoke.py`, `retraining_experiment/smoke.py`). Plus Ruff (`/home/sc/.cache/potato-baseline/.venv/bin/ruff check --no-cache public_label_rework` and `ruff format --check --no-cache public_label_rework`) and basedpyright CLI (`/home/sc/.cache/potato-baseline/.venv/bin/basedpyright --pythonpath /home/sc/.cache/potato-baseline/.venv/bin/python public_label_rework`; report warnings, never suppress).
- Python: `PY=/home/sc/.cache/potato-baseline/.venv/bin/python`, run from the workspace root, fresh process per GPU command (Ultralytics runtime isolated per process as in `retraining_experiment/common.py:runtime`).
- Agent-executed QA on real outputs: validators, PASS/FAIL tokens in `metrics.json`, rendered images opened and inspected by the executing agent. Prose deliverables (Korean README, pilot report) are verified by an agent read against required sections plus machine checks of the numbers they quote against `metrics.json`; no sentence greps.
- Evidence: `.omo/evidence/public-label-rework/task-<N>.<ext>` for every todo (outside ulw-loop); inside ulw-loop use <attemptDir>/task-<N>-public-label-rework.<ext> (attemptDir = result.currentAttemptDir from await agentToolkit.status() in a JS eval cell after importing agentToolkit with await import(`${env("OMO_AGENT_TOOLKIT_SDK_ROOT")}/sdk.js`)).

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

- Wave 1 (foundation): 1 intake and protection, 5 rendering (fixture-based), 2 label IO (fixtures first; integration step after 1 finishes), 4 SAM snap (download + fixtures; integration step after 1).
- Wave 2 (references, models, core): 3 inventory, 9 decision schema and apply, 19 package builder start immediately; 7 gold set and list freezing and 8 bootstrap detector start after 3; 18 honeypot pools starts after 7 has frozen the lists.
- Wave 3 (pipeline): 6 triage pass, 10 proposals, 11 decision-pass orchestration, 12 validators and metrics.
- Wave 4 (pilot, sequential): 13 pilot run, 14 pilot STOP gate.
- Wave 5 (full run, after gate): 15 full run, 16 independent verification, 17 delivery package.
- Final: F1-F4.

PRE-GATE SCOPE: before the user's confirmation, only these steps may read all 2,060 images, and none of them produces a review decision or a label change: todo 1 (intake/hash), todo 2 (round trip), todo 3 (automated inventory), todo 8 (detector training and out-of-fold predictions). Reference sets are processed on their frozen lists only: todo 7 gold (100 images) and todo 18 honeypots (50 images). Triage (todo 6), proposals (todo 10), decision passes (todo 11) and apply (todo 9) run before the gate ONLY on the frozen pilot list `outputs/lists/pilot_list.json` (100 images, frozen in todo 7 with SHA-256). Every pipeline module takes an explicit image-list file and refuses images outside it. Honeypot images shown inside pilot batches are scored against their adjudicated answers only; those decisions are never applied, and apply (todo 9) refuses any non-pilot id before the gate.

STOP RULE: after todo 14 presents the pilot, execution halts. No Wave 5 todo may start until `public_label_rework/outputs/pilot/user_confirmation.json` exists in the `label_review_v2/user_confirmation.json` shape (date, basis, `statement_verbatim` non-empty). Triage, proposals, decisions and apply for the remaining 1,960 images happen only inside todo 15.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | - | 2 (integration), 3, 4 (integration), 7, 8, 13, 18 | 5, 2 (fixture part), 4 (fixture part) |
| 2 | 1 (integration step only) | 9, 10, 12, 19 | 1, 4, 5 |
| 3 | 1 | 6, 7, 8, 10, 12 | 9, 19 |
| 4 | 1 (integration step only) | 7, 10, 18 | 1, 2, 5 |
| 5 | - | 6, 7, 10, 11, 18 | 1, 2, 4 |
| 6 | 3, 5, 7, 18 | 11, 13, 15 | 10, 12 |
| 7 | 1, 3, 4, 5 | 6, 10, 12, 13, 18 | 8, 9, 19 |
| 8 | 1, 3 | 10 | 7, 9, 18, 19 |
| 9 | 2 | 11, 13, 19 | 3, 7, 8, 18 |
| 10 | 2, 3, 4, 5, 7, 8 | 11, 13, 15 | 6, 12 |
| 11 | 5, 9, 18 (code + fixture dry run); 6, 10 (real batches inside 13) | 13 | 6, 10, 12 |
| 12 | 2, 3, 7 | 13, 16 | 6, 10, 11 |
| 13 | 1, 6, 7, 9, 10, 11, 12, 19 | 14 | - |
| 14 | 13 | 15 | - |
| 15 | 6, 10, 11, 14 | 16 | - |
| 16 | 12, 15 | 17 | - |
| 17 | 16, 19 | F1-F4 | - |
| 18 | 1, 4, 5, 7 | 6, 11 | 8, 9, 19 |
| 19 | 2, 9 | 13, 17 | 3, 7, 8, 18 |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [x] 1. Intake public subset and protect existing files
  Recommended task executor category: quick - mechanical extraction and hashing in one module.
  What to do: create package `public_label_rework/` (`__init__.py`, `common.py` with ROOT, ZIP path, OUTPUTS, `new_output()` refusing existing paths, `sha256`, `write_json`). Module `intake.py`: before writing anything, hash every pre-existing workspace file (exclude `.omo/`, `.ruff_cache/`, `__pycache__/`, `public_label_rework/`) into `public_label_rework/outputs/protected_before.json`; then copy from `final-dataset.zip` exactly `images/public/*` (2,060 JPG + 2,060 JSON), `sources/**`, `manifest.json`, `사용안내.txt` into `public_label_rework/outputs/source/`; verify every image SHA-256 against `manifest.json` `records[].image_sha256`; verify 2,060 pairs and 9,498 boxes; check each `sources/original0*/README.roboflow.txt` + `data.yaml` contains a URL and a license line; write `intake.json` (counts per source, ZIP SHA-256, license lines per source). Module `protect.py verify` re-hashes and fails on any change.
  Must NOT do: extract `images/harvest`; modify the ZIP; write outside `public_label_rework/`.
  Closes: GAP-5 (provenance baseline), GAP-10 (license inputs); supports IS-5, IS-7, IS-11
  Parallelization: Wave 1 | Blocked by: - | Blocks: 2 (integration), 3, 4 (integration), 7, 8, 13, 18
  References: `final-dataset.zip` (`manifest.json` keys updated/summary/subsets/public_sources/checks/records; records[] = image, label, source_prefix, image_sha256); `scripts/baseline_common.py` (sha256, write_json); `retraining_experiment/common.py` (new_output); prototype `/tmp/potato_protect.py` (snapshot/verify).
  Acceptance criteria: `$PY -B -m public_label_rework.intake` exits 0; `intake.json` shows public images 2060, labels 2060, boxes 9498, per source original01:80 original02:1343 original03:96 original06:223 original07:318, hash mismatches 0, every source has url and license `CC BY 4.0`; `$PY -B -m public_label_rework.protect verify` prints `CHANGED []`.
  QA scenarios: happy - run intake, then count `outputs/source/images/public` = 4120 files; failure - rerun intake with the same output path exits nonzero; smoke fixture with one corrupted image is rejected with a hash mismatch. Evidence .omo/evidence/public-label-rework/task-1.txt
  Commit: N | workspace is not a git repository

- [x] 2. X-AnyLabeling label IO with byte-faithful round trip
  Recommended task executor category: unspecified-low - one parsing/writing module plus smoke tests.
  What to do: `labelio.py` parsing an X-AnyLabeling JSON into frozen dataclasses (top-level: version, flags, checked, shapes, imagePath, imageData, imageHeight, imageWidth, description; shape: label, points, group_id, description, difficult, shape_type, flags, attributes, and `score` only when present) preserving unknown keys and key order; stable box id = original index + SHA-256 of points rounded to 0.01; `to_xyxy`/`from_xyxy` (4 points clockwise from top-left, matching input); writer emitting the same key order, `ensure_ascii=False`, indent and trailing newline identical to source. Fixture tests first (Wave 1), then the integration round trip on all 2,060 public JSON once todo 1 has produced `outputs/source`.
  Must NOT do: drop unknown keys; reorder points; change imagePath; add `score`.
  Closes: supports IS-4
  Parallelization: Wave 1 | Blocked by: 1 (integration step only) | Blocks: 9, 10, 12, 19
  References: a public JSON from `outputs/source/images/public/` (after todo 1) or read in-memory from `final-dataset.zip` `images/public/public_original06_0001.json`; `retraining_data/native.py:117-155` (strict parse style); `retraining_data/schema.py` (require, DataError, document).
  Acceptance criteria: smoke round-trips all 2,060 public JSON byte-identically (2060/2060); unknown-key fixture preserved; 3-point shape fixture raises DataError.
  QA scenarios: happy - round trip report 2060 identical; failure - fixture with `shape_type: polygon` is rejected with DataError. Evidence .omo/evidence/public-label-rework/task-2.txt
  Commit: N | workspace is not a git repository

- [x] 3. Inventory, flags, augmentation families and harvest convention report
  Recommended task executor category: deep-low - hashing thresholds and family recall need judgment from data.
  What to do: `inventory.py` producing `outputs/inventory/inventory.jsonl` (per image: source, size, box count, letterbox rows/cols, rotation-wedge flag with estimated angle, out-of-bounds boxes, overlap candidates, tiny/huge boxes). Overlap formula pinned: IoU = inter/union; containment = inter/min(area_a, area_b); a pair is an overlap candidate when IoU >= 0.5 or containment >= 0.8 (same as `/tmp/potato-final/duplicate_check.py:overlap`). Augmentation families: 16x16 average hash over the 8 flip/rot90 variants; two thresholds chosen from the per-source Hamming histogram (never from gold): `strict` for consistency checks (component size <= 6, larger components split and logged) and `loose` for fold grouping (over-grouping harmless); edge also requires box counts within 1. Family recall: share of images in 2-3 member strict families per source vs the READMEs' 3-versions expectation. Conversion sanity: render 12 original06 images with boxes. Harvest convention report: read `images/harvest/*.json` and a 24-image sample in memory from the ZIP (no extraction): box area/aspect stats, whether sprouts are inside boxes, amodal vs visible; save renders to `outputs/inventory/harvest_convention/`. Source01 report: class names from `sources/original01/data.yaml` and images likely holding non-potato produce.
  Must NOT do: drop or move any image because of family membership; write harvest images outside the renders folder.
  Closes: GAP-3 (detection), GAP-6, GAP-9 (harvest convention), supports GAP-2 measurement
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 6, 8, 10, 12
  References: `/tmp/potato-final/duplicate_check.py`, `/tmp/potato-final/near_duplicates.py` (548-image chaining failure at threshold 24), `/tmp/potato-final/inspect_public.py` (per-source stats), `final-dataset.zip` `sources/original0*/README.roboflow.txt`; `stone_experiment/metrics.py:iou`.
  Acceptance criteria: inventory rows = 2060; overlap-candidate images = 40 (PASS token); no strict family larger than 6; family recall table per source written to `families.json`; `harvest_convention.json` contains box stats and a sprouts verdict (`included` / `excluded` / `none_seen`); conversion sanity sheet inspected (boxes on potatoes, PASS token written by the agent).
  QA scenarios: happy - open `families_examples.jpg` (6 random strict families) and confirm each is the same scene; failure - smoke fixture of two different scenes with equal hashes but box counts 1 vs 5 is not grouped. Evidence .omo/evidence/public-label-rework/task-3.txt
  Commit: N | workspace is not a git repository

- [ ] 4. SAM2.1-b download and box snapping
  Human-informed amendment (2026-11-02, user `ㅇㅇ 그렇게 진행하자`): allow a separate input version using the50actualjudgments and exactlytwoKEEP effective-negative overrides(C022/C028),preserving original375CASE/points/fullnegativepayload/oldruns/90percent. Bounded32case diagnostic,max50modelcalls;no full375/pilot approval. Contract: `.omo/ulw-execute/task-4-human-guidance-v2-policy.json`.
  - [x] Prepare human-guidance-v2 inputs and bounded runner.
  - [x] Independently verify new inputs and runner before release. Evidence: `.omo/evidence/public-label-rework/human-guidance-v2-gate.md`;32case-only parent release issued.
  - [x] Execute bounded diagnostic and independently inspect actual masks. Actual48calls;sample PASS12/FAIL17/UNRESOLVED3;mechanismFAIL andconservative335<338bound. Task4 remains unchecked;no full375/pilot expansion. Summary: `.omo/evidence/public-label-rework/human-guidance-v2-result-20261102.md`.
  - [x] User feedback (2026-11-02): all50actual judgments recorded and independently verified:14distinct/10same/3nonpotato/23unknown. Corrected impact confirmed:twoKEEP originalnegative conflicts(C022/C028),not10. Existing375/90contract unchanged;further human-informed input version requires a separate decision. Evidence: `.omo/evidence/public-label-rework/user-feedback-50-gate-repair-20261102.md`.
  - [x] User follow-up (2026-10-30): create and verify an offline Korean review packet for the50unresolvedtargets, report Windows folder/ZIP location and original-plan progress; no inference or label edits authorized by this request. Delivery: `public_label_rework/outputs/user_review_50_20261030/index.html` and sibling ZIP. Independent evidence: `.omo/evidence/public-label-rework/user-review-50-gate-link-fix.md`.
  Supplementary diagnostic outcome (2026-10-30): independently confirmed input accounting325ready/43abstain/7null cannot reach338of375. The approved diagnostic stopped before GPU calls per its capacity rule; CPU runner repairs independently confirmed. Task4 remains unchecked. Evidence: `.omo/evidence/public-label-rework/body-guidance-diagnostic-result-20261030.md`.
  Supplementary-input amendment (2026-10-30, user: `ㅇㅇ 그렇게 진행해`): authorize raw-image-only independent supplementary-point authorship/review and pre-inference freeze, followed by a bounded diagnostic under `.omo/ulw-execute/task-4-supplementary-protocol.json`. Preserve all375 targets, original points/statuses, null7 failures, old runs and90percent. This is explicitly a NEW input version, not identical-input comparison or full375/pilot authorization.
  Recommended task executor category: deep-low - mask-to-box guards need careful edge-case handling.
  Execution amendment (2026-10-29, explicit user decision): `검사 입력: 개체별 입력으로 평가 (Recommended)`. Keep the 90% threshold and the same original 20 images, but evaluate individual-potato candidate inputs rather than source boxes that enclose groups. The user separately retained the threshold: `SAM 기준: 원래 90% 기준 유지`. Preserve the original 40-box experiment and its failures as a separate baseline; do not relabel those results as passing.
  What to do: `snap.py`: download `sam2.1_b.pt` via Ultralytics `SAM('sam2.1_b.pt')` with cwd and weights dir `public_label_rework/weights/` (isolated runtime as `retraining_experiment/common.py:runtime`), record SHA-256 in `weights/weights.json`. `snap(image, xyxy) -> SnapResult(xyxy, tight, reason, mask_area_ratio)`: box prompt with 10% padding; largest mask component; tight bbox constrained inside the padded prompt; accept when mask-bbox IoU with input >= 0.5, mask area >= 30% of input box area and mask touches the padded border on at most 2 sides; clip to image content (exclude letterbox bars); else return clipped input with tight=false and reason. Point-prompt variant `snap_point(image, xy)` for agent-indicated misses. `auto_masks(image, region)` wrapper for SAM automatic masks (used by todo 7 and todo 11 follow-ups). Batch API on GPU.
  Must NOT do: install packages; place weights outside `public_label_rework/weights/`.
  Closes: GAP-7 (tightness tool), supports GAP-2
  Parallelization: Wave 1 | Blocked by: 1 (integration step only) | Blocks: 7, 10, 18
  References: Ultralytics 8.4.183 SAM API at `/home/sc/.cache/potato-baseline/.venv/lib/python3.12/site-packages/ultralytics/models/sam/`; `retraining_experiment/common.py:runtime`.
  Acceptance criteria (amended input contract): weights file exists with SHA-256 recorded; retain the exact same 20 original06 image IDs from the original experiment. Before the evaluation run, freeze and hash a per-instance target/candidate roster, with tool-produced candidate boxes, provenance, and an explicit new denominator. Include every identifiable in-focus tuber under the plan's box/background policy; record ambiguous or unproposed targets rather than silently dropping difficult cases. Evaluate every frozen target; unresolved targets count as failures. A success requires the unchanged geometry guards, tight=true, and independent visual confirmation of a single visible tuber body without plate/background/neighbor/sprout contamination. At least 90% of the frozen targets must succeed. Write mask-area/box-area distribution and all failures to `snap_stats.json`, report per-image counts and the original 40-box baseline separately, and never change the roster after observing evaluation results. A prompt on a letterbox-only area returns tight=false; on `public_original02_0177` (shifted box) the snapped box is allowed to extend beyond the input box within the padded prompt.
  QA scenarios: happy - render 12 before/after snaps to `snap_examples.jpg` and the agent confirms green boxes hug potato outlines (PASS token); failure - empty-area prompt returns tight=false with reason. Evidence .omo/evidence/public-label-rework/task-4.txt
  Commit: N | workspace is not a git repository

- [x] 5. Review rendering: triage sheets, proposal sheets, before/after
  Recommended task executor category: unspecified-low - PIL rendering utilities.
  What to do: `render.py`: (a) triage sheet of 6 images at >= 480 px each with current boxes, image id and automated flags; (b) proposal sheet per image: full image upscaled to 1280 px with numbered boxes color-coded (existing red, refined green, candidate yellow), an 8x8 grid with ids A1..H8, plus a 3x crop strip per box showing old vs refined, and optional 3x per-cell crops for requested cells; (c) before/after pair renderer. Deterministic font (`/home/sc/.config/Ultralytics/Arial.ttf`, fallback PIL default).
  Must NOT do: write renders anywhere except `outputs/**/renders/`.
  Closes: supports GAP-1, GAP-2, GAP-4 review (U4)
  Parallelization: Wave 1 | Blocked by: - | Blocks: 6, 7, 10, 11, 18
  References: `label_review/previews/`, `label_review/before_after/` (visual conventions); `tracking_experiment/video.py:overlay` (PIL drawing style).
  Acceptance criteria: smoke renders all three types for 3 fixture images; a box-id collision raises DataError; the agent reads every box number on a 21+-box fixture render (PASS token).
  QA scenarios: happy - proposal sheet for a 21+-box fixture is legible; failure - collision raises DataError. Evidence .omo/evidence/public-label-rework/task-5.txt
  Commit: N | workspace is not a git repository

- [ ] 6. Image-level triage pass
  Recommended task executor category: unspecified-high - orchestrates parallel reviewer agents with rubric and validation.
  What to do: `triage.py --list <image-list.json>` (refuses images outside the list). Before the gate it runs on `pilot_list.json` only; todo 15 runs it on `remaining_list.json`. It builds triage sheets (6 images each, grouped by source; about 17 for the pilot, about 327 for the remainder) and dispatches reviewer agents (8-16 parallel, fresh agent per batch of sheets) with a rubric: bucket ok / dense-pile / composite-suspect / off-domain / junk, plus independent flags watermark_stock and background_pile, one-line reason. Each batch contains triage honeypots from todo 18; any wrong honeypot or schema error fails the batch, which is re-issued to a fresh agent (max 2 retries, then images marked needs_human). Bucket fate: ok/dense-pile -> normal decision pass; composite-suspect -> decision pass with drop_image(composite_unclear) allowed; off-domain -> decision pass with delete(not_potato) or drop_image(junk); junk -> drop_image(junk) confirmed in todo 11; watermark_stock and background_pile -> flags only. Output `outputs/triage/triage.jsonl`.
  Must NOT do: exclude images at this stage.
  Closes: GAP-4 (identification), GAP-10 (watermark flag), supports GAP-1, GAP-9
  Parallelization: Wave 3 | Blocked by: 3, 5, 7 (pilot list), 18 | Blocks: 11, 13, 15
  References: draft `.omo/drafts/public-label-rework.md` (architect stage 1); `label_review/README.md`.
  Acceptance criteria: pre-gate run: triage.jsonl has exactly the 100 pilot rows (no other image ids); after todo 15 the merged triage covers 2060 rows; buckets and flags valid, every accepted batch has 100% honeypot accuracy, needs_human count written to `triage_summary.json`.
  QA scenarios: happy - bucket counts per source written and 10 random composite-suspect images inspected by the agent; failure - a batch output missing one image is rejected and re-issued. Evidence .omo/evidence/public-label-rework/task-6.txt
  Commit: N | workspace is not a git repository

- [ ] 7. Gold reference set (agent gold, not human)
  Recommended task executor category: unspecified-high - independent labelers plus adjudication protocol.
  What to do: `gold.py` selects 100 images with a fixed seed, stratified by source (original01:8, original02:40, original03:10, original06:20, original07:22) and box-count bands (1, 2-5, 6-10, 11+), disjoint from honeypots (todo 18); assign G-pilot (50) and G-full (50); write `outputs/gold/gold_list.json` and `gold_list.sha256`. In the same step freeze the pilot list `outputs/lists/pilot_list.json` (+ `.sha256`): the 50 G-pilot images + 50 non-gold, non-honeypot images stratified by source and inventory box-count band with at least 10 images having >= 11 boxes or an inventory dense flag (fixed seed); also freeze `outputs/lists/honeypot_list.json` (+ `.sha256`): 20 triage + 30 decision honeypot ids, disjoint from gold and pilot; the rest of the images form `outputs/lists/remaining_list.json` (1,960 = 2,060 - 100 pilot; honeypot and G-full images are part of it and are processed by the pipeline only in todo 15, their reference answers are never shown to pipeline agents). Labelers see the image WITHOUT existing boxes. Candidates = SAM automatic masks (todo 4 `auto_masks`, no detector) rendered as numbered candidates + agent accept/reject, plus agent point prompts on 3x per-cell crops for missed potatoes (snapped by `snap_point`). Two independent labelers per image, a third agent adjudicates; rubric includes D1, D7 and the box policy; exclusion allowed with reason. Every gold box records its candidate source in `shape.description` together with `agent_gold_not_human`. Labeler-pair F1 @0.5 < 0.8 tags the image `gold_uncertain` (excluded from metric denominators, count written).
  Must NOT do: show gold labels to any pipeline decision agent; use the bootstrap detector; tune thresholds on G-full.
  Closes: GAP-7
  Parallelization: Wave 2 | Blocked by: 1, 3, 4, 5 | Blocks: 6, 10, 12, 13
  References: draft D5, D7; `label_review/protocol.json` box_policy; todo 4 snap API; todo 3 inventory (box-count bands).
  Acceptance criteria: pilot_list (100) + remaining_list (1,960) + gold partition consistent and hashed; 100 gold JSON files; every box has a candidate-source tag and snap provenance; adjudication log; `gold_summary.json` with labeler F1 per image and `gold_uncertain` count; `gold_list.sha256` exists before todo 10 runs.
  QA scenarios: happy - render 10 gold images including `public_original06_0162` and `public_original06_0018` and confirm in-focus potatoes boxed tightly and background left unboxed with the flag; failure - a gold file with a box lacking snap provenance fails validation. Evidence .omo/evidence/public-label-rework/task-7.txt
  Commit: N | workspace is not a git repository

- [x] 8. Bootstrap detector with out-of-fold predictions
  Recommended task executor category: deep-low - training config and fold isolation.
  What to do: `detector.py`: train YOLO11s from COCO `yolo11s.pt` (workspace root, read only) on inventory-clean images (no rotation-wedge flag, no overlap candidate, no out-of-bounds box) with `degrees=0, shear=0, perspective=0, amp=False`, imgsz 640, seed 42, SGD and other settings as `baseline/config/train.json`; 5 folds grouped by `loose` augmentation families (no family split across folds); predict every one of the 2,060 images with the fold model that did not train on its family (conf 0.05 saved, TTA on), plus 2x2 tiled prediction for dense-pile-like images (box count >= 11 or inventory flag); save `outputs/detector/oof.jsonl` and fold manifests. Isolated runtime per process.
  Must NOT do: train on gold labels; download any asset during training; modify existing models.
  Closes: supports GAP-1, GAP-2, GAP-3
  Parallelization: Wave 2 | Blocked by: 1, 3 | Blocks: 10
  References: `retraining_experiment/train.py`, `retraining_experiment/common.py:runtime`, `baseline/config/train.json`, `retraining_experiment/outputs/train/args.yaml` (amp was true there; set false here).
  Acceptance criteria: 5 fold weights with SHA-256; assertion that no image's loose family appears in its predicting fold's train list passes; training logs contain no download line (PASS token); `amp: false` in each fold args.
  QA scenarios: happy - render OOF boxes on 12 images incl. 3 dense piles and inspect; failure - an injected family leak in a fixture fold list raises. Evidence .omo/evidence/public-label-rework/task-8.txt
  Commit: N | workspace is not a git repository

- [x] 9. Decision schema and deterministic apply
  Recommended task executor category: unspecified-high - correctness core with idempotency guarantees.
  What to do: `decisions.py` JSONL ops referencing stable box ids: keep(box_id, refined_xyxy, src=sam), replace(box_id, xyxy, src=sam|sam_point), delete(box_id, reason=dup|not_potato|background), add(xyxy, src=sam|sam_point, candidate_id), drop_image(reason=composite_unclear|boundary_unresolvable|junk), needs_human(reason), set_flag(background_potatoes|watermark_stock), overlap_reviewed(box_a, box_b). `apply.py` reads source JSON + all decision files, rejects conflicting or unknown-id ops, writes `outputs/<run>/labels/images/public/*.json` for kept images, `needs_human/` (JPG copy + JSON), `exclusions.csv` (image, reason, batch; utf-8-sig), `changes.jsonl` (before/after per box), `changes.csv` (Korean text templated by apply, utf-8-sig), `status.csv`; any image with a box lacking tight=true is routed to needs_human automatically; flags per Must have; byte-identical rerun.
  Must NOT do: accept coordinates without a snap receipt; touch images; add top-level keys.
  Closes: GAP-5, GAP-8 (lists), supports IS-3, IS-4, IS-6
  Parallelization: Wave 2 | Blocked by: 2 | Blocks: 11, 13, 19
  References: `label_review/changes.json`, `label_review/review.csv` (prior change log shapes); todo 2 labelio.
  Acceptance criteria: smoke - applying twice gives an identical SHA-256 tree; conflicting ops rejected; untight box routes image to needs_human; every kept JSON has top-level `checked:false`, `flags.human_reviewed:false`, `flags.ai_reviewed:true`.
  QA scenarios: happy - apply a fixture decision set and diff before/after; failure - add without snap receipt is rejected. Evidence .omo/evidence/public-label-rework/task-9.txt
  Commit: N | workspace is not a git repository

- [ ] 10. Proposal generation
  Recommended task executor category: deep-low - matching and status rules.
  What to do: `proposals.py --list <image-list.json>` (refuses images outside the list and refuses to start unless `outputs/gold/gold_list.sha256` exists; pre-gate on `pilot_list.json` only, todo 15 runs it on `remaining_list.json`): Hungarian match existing boxes vs OOF (IoU); status `agreed` (IoU >= 0.8), `disagreed` (0.5-0.8), `unmatched`; overlap candidates from the pinned rule; every existing box gets a SAM-snapped refinement (todo 4), fallback to the snapped OOF box when existing snap fails and IoU >= 0.5, else marked tight=false; miss candidates = unmatched OOF conf >= 0.25 (and tiled predictions for piles), each SAM-snapped; render proposal sheets (todo 5) and write `proposal.json` per image.
  Must NOT do: apply anything; call agreement "tight".
  Closes: supports GAP-1, GAP-2, GAP-3
  Parallelization: Wave 3 | Blocked by: 2, 3, 4, 5, 7, 8 | Blocks: 11, 13, 15
  References: draft architect stage 3; `stone_experiment/metrics.py`; `scipy.optimize.linear_sum_assignment` call site in `tracking_experiment/tracker.py`.
  Acceptance criteria: pre-gate run produces proposals for exactly the 100 pilot images (no other ids); median refined/existing area ratio for original02 pilot images < 1.0 (PASS token) written to `proposal_stats.json`; zero-detection images still get sheets with existing boxes; after todo 15 proposals exist for all 2060.
  QA scenarios: happy - inspect 10 proposal sheets across sources incl. `public_original02_0857`; failure - running without gold_list.sha256 exits nonzero. Evidence .omo/evidence/public-label-rework/task-10.txt
  Commit: N | workspace is not a git repository

- [ ] 11. Decision-pass orchestration
  Recommended task executor category: unspecified-high - parallel agent batches, rubric, honeypots, follow-up rounds.
  What to do: `review.py` batches of 12-20 images by source+bucket; rubric with 2 worked examples: per box keep/replace/delete, per candidate accept/reject, per overlap candidate pair `overlap_reviewed` decision, per image missing-cell ids + estimated in-focus potato count (background excluded per D7) or drop_image/needs_human with reason; decision honeypots (todo 18) 1-2 per batch, any wrong answer fails the batch (fresh agent re-run, max 2, then needs_human). Count rule: estimated count vs final box count within 10% (or within 1 for <= 10 potatoes) else follow-up round: SAM automatic masks + 0.05-conf tiled detector restricted to flagged cells, agent accept/reject, max 2 rounds, then drop_image(boundary_unresolvable) or needs_human. Family consistency check with allowed explanation codes `geometry_differs`, `crop_cuts_potato`, `composite_only_in_member`, `background_rule`, `occlusion_differs`; other mismatches re-reviewed. Resumable `batches.jsonl`. Build and dry-run on fixture batches in Wave 3; real batches run inside todos 13 and 15.
  Must NOT do: let agents edit JSON or see gold.
  Closes: GAP-1, GAP-3, GAP-4, GAP-6 (consistency)
  Parallelization: Wave 3 | Blocked by: 5, 9, 18 (real batches also need 6, 10) | Blocks: 13
  References: draft architect stage 4 and batching section.
  Acceptance criteria: fixture dry run produces valid decision JSONL; a decision referencing an unknown box id is rejected; a failed honeypot triggers a re-run (logged).
  QA scenarios: happy - one real batch of 12 images end to end after 6 and 10 exist; failure - unknown box id rejected. Evidence .omo/evidence/public-label-rework/task-11.txt
  Commit: N | workspace is not a git repository

- [ ] 12. Validators and metrics
  Recommended task executor category: unspecified-low - checks and reports.
  What to do: `validate.py` asserts: schema via todo 2; key set/order equal to source except `flags` and shape additions; 4-point rectangles in bounds; min size 4 px; no pair IoU >= 0.9; every overlap candidate pair has an `overlap_reviewed` record; every shape `flags.tight == true` in images/public; top-level `checked == false`; `flags.human_reviewed == false`; `flags.ai_reviewed == true`; image bytes equal `manifest.json` `image_sha256`; filenames equal; kept + excluded + needs_human = total. `metrics.py` vs gold (excluding `gold_uncertain`): P/R @0.5 and @0.75, median IoU, median and p90 area ratio, share of matched boxes with IoU < 0.7, per-image count MAE, duplicate rate; per source and population-weighted (by source share); emits `metrics.json` with PASS/FAIL tokens for: P >= 0.97 and R >= 0.97 @0.5, median IoU >= 0.85, median area ratio within 0.95-1.05, p90 area ratio <= 1.15, share IoU < 0.7 <= 5%, duplicates = 0.
  Must NOT do: alter outputs.
  Closes: GAP-7 measurement, supports IS-2, IS-4, IS-9
  Parallelization: Wave 3 | Blocked by: 2, 3, 7 | Blocks: 13, 16
  References: `stone_experiment/metrics.py` (match_frame), `/tmp/potato_compare_full.py` (report style).
  Acceptance criteria: validators on untouched source labels report the 40 overlap-candidate images and FAIL (no overlap records); metrics on gold vs itself give P=R=1, IoU=1 with all PASS tokens.
  QA scenarios: happy - validators pass on a fixture tree; failure - a box out of bounds, a changed image byte or a tight=false box in images/public fails. Evidence .omo/evidence/public-label-rework/task-12.txt
  Commit: N | workspace is not a git repository

- [ ] 13. Pilot run (100 images)
  Recommended task executor category: unspecified-high - runs the whole pipeline on a subset.
  What to do: use the frozen `pilot_list.json` from todo 7 (verify its SHA-256); run real triage-derived decision batches (todo 11) on pilot images only, apply, validators, metrics on G-pilot; at most one rubric iteration if a threshold FAILs, recorded in `rubric_changes.md`, then rerun and report both runs; record per-image agent calls, tokens when available, and wall time, and the projected full-run cost; build the pilot mini-package in the final delivery layout via todo 19 under `outputs/pilot/package/` plus `pilot_report.md` in Korean and `pilot_metrics.json`.
  Must NOT do: process any image outside the pilot list; use G-full.
  Closes: GAP-1..GAP-4 for pilot images; IS-9 evidence
  Parallelization: Wave 4 | Blocked by: 1, 6, 7, 9, 10, 11, 12, 19 | Blocks: 14
  References: todos 9-12, 19; draft D4.
  Acceptance criteria: validators PASS on the pilot tree; `pilot_metrics.json` contains every threshold with PASS/FAIL tokens; cost projection present (numbers, not prose); all changed pilot images rendered before/after.
  QA scenarios: happy - open 15 before/after images (5 dense, 5 original02, 5 others) and confirm tight, one per potato (agent PASS token per image); failure - any FAIL token is copied verbatim into the report headline. Evidence .omo/evidence/public-label-rework/task-13.md
  Commit: N | workspace is not a git repository

- [ ] 14. Pilot STOP gate
  Recommended task executor category: quick - present and record.
  What to do: present the pilot report summary (metrics with PASS/FAIL, examples, exclusions, needs-human count, projected full-run cost, option to reduce todo 16's retrain step) and package path to the user in Korean, then halt. On confirmation, write `outputs/pilot/user_confirmation.json` in the `label_review_v2/user_confirmation.json` shape with `statement_verbatim`; feedback becomes rubric changes in `rubric_changes.md` and the pilot is re-run before proceeding.
  Must NOT do: start Wave 5 without the confirmation file.
  Closes: IS-9
  Parallelization: Wave 4 | Blocked by: 13 | Blocks: 15
  References: draft D4; `label_review_v2/user_confirmation.json`.
  Acceptance criteria: confirmation file exists with non-empty `statement_verbatim` before any Wave 5 command; `full_run` exits nonzero without it.
  QA scenarios: happy - confirmation recorded; failure - `full_run` without the file exits nonzero. Evidence .omo/evidence/public-label-rework/task-14.txt
  Commit: N | workspace is not a git repository

- [ ] 15. Full run over remaining images
  Recommended task executor category: unspecified-high - batch orchestration at scale.
  What to do: `full_run.py` refuses without the confirmation file; on `remaining_list.json` runs triage (todo 6 module), proposals (todo 10 module) and decision batches (todo 11), reusing pilot decisions unless the rubric changed; then apply over all 2,060 and validators; resumable via `batches.jsonl`; renders before/after for every changed image under `outputs/full/renders/`.
  Must NOT do: re-decide pilot images unless the rubric changed in todo 14; read G-full labels.
  Closes: GAP-1..GAP-4 for all images
  Parallelization: Wave 5 | Blocked by: 6, 10, 11, 14 | Blocks: 16
  References: todos 6, 9-12, 14.
  Acceptance criteria: every image has one final status (kept/excluded/needs_human) summing to 2060; validators PASS on the full tree.
  QA scenarios: happy - status counts sum to 2060; failure - kill and resume mid-run without duplicated batches. Evidence .omo/evidence/public-label-rework/task-15.txt
  Commit: N | workspace is not a git repository

- [ ] 16. Independent verification and residual round
  Recommended task executor category: deep-low - adversarial verification.
  What to do: in this order. (a) Blind audit by a fresh agent without the change log on a random 10% of kept images plus every image with >= 5 changes (rubric: every in-focus potato boxed once, tight, no non-potato, background rule), per-image defect-free rate and per-box error rate; every defect becomes a remediation decision. (b) Retrain detector 5-fold on the current labels (reducible at the gate to OOF-disagreement listing only) and run one residual decision round on OOF disagreements. (c) Re-apply, re-validate, re-audit the remediated images; family consistency report with explanation codes; per-source box area/aspect distribution before/after; harvest 3 overlap candidates and convention comparison report-only. (d) Freeze the final tree: write `final_tree.sha256` (hash of every output file). (e) Only then evaluate G-full exactly once on the frozen tree (population-weighted and per source) and record `gfull_evaluations: 1`; G-pilot reported separately. A G-full FAIL blocks delivery: write status `blocked_gfull_fail` with the failing tokens and stop for the user; no correction loop runs against G-full. Writes `verification.json` and `verification.md`.
  Must NOT do: change labels except through todo 11 decision files; read G-full labels before step (e); evaluate G-full more than once.
  Closes: IS-9 proof, GAP-6 check, IS-10 comparison
  Parallelization: Wave 5 | Blocked by: 12, 15 | Blocks: 17
  References: todo 12; draft architect verification section.
  Acceptance criteria: blind audit per-image defect-free rate >= 95% after remediation; residual round applied and validators PASS before freezing; `final_tree.sha256` written before the G-full evaluation timestamp; `gfull_evaluations == 1`; all G-full PASS tokens, else status `blocked_gfull_fail` and todo 17 does not run; zero family mismatches without an allowed code.
  QA scenarios: happy - all G-full thresholds PASS on the frozen tree; failure - a G-full FAIL writes `blocked_gfull_fail`, todo 17's builder refuses to run on that status, and the user is asked how to proceed. Evidence .omo/evidence/public-label-rework/task-16.md
  Commit: N | workspace is not a git repository

- [ ] 17. Delivery package
  Recommended task executor category: writing - Korean README plus packaging run.
  What to do: run todo 19's builder on the final tree to `outputs/delivery/public_relabel_v1/`: `images/public/` (byte-identical JPG + corrected JSON for kept images), `needs_human/`, `exclusions.csv`, `changes.csv`, `before_after/` samples (all renders stay in outputs), `sources/` unchanged, `README.md` in Korean with sections: 작업 기준(박스 정책, 흐린 배경 규칙, 새싹 제외), 수량(출처별 kept/excluded/needs_human, 배경 감자 표시 수, 워터마크 수), 품질 지표(G-full, PASS/FAIL), 라이선스 및 출처(각 출처 이름·URL·CC BY 4.0, 라벨 수정 사실, 이미지는 무수정), 한계(AI 검수이며 사람 확정 아님, 늘림/레터박스 전처리로 픽셀 크기 비교 주의, X-AnyLabeling GUI 열기 미검증, harvest 규칙과의 차이), 사용 방법; ZIP with ASCII entry names + `archive_verification.json` (CRC test, SHA-256, reparse every JSON from the ZIP via todo 2); final `protect verify`.
  Must NOT do: include renders inside images/public; send the ZIP anywhere.
  Closes: GAP-8, GAP-10, IS-4, IS-5, IS-6, IS-7, IS-11
  Parallelization: Wave 5 | Blocked by: 16, 19 | Blocks: F1-F4
  References: `label_review/README.md`, `label_review/archive_verification.json`; todo 19.
  Acceptance criteria: ZIP CRC ok; every JSON reparses and passes validators; kept + excluded + needs_human = 2060; `protect verify` prints `CHANGED []`; an agent read confirms every required README section and the numbers quoted match `metrics.json` (PASS token).
  QA scenarios: happy - extract the ZIP to a temp dir, parse all labels with todo 2 and convert to YOLO txt with a throwaway script, 0 errors; failure - a renamed image in a fixture package is detected. Evidence .omo/evidence/public-label-rework/task-17.txt
  Commit: N | workspace is not a git repository

- [ ] 18. Honeypot pools
  Recommended task executor category: unspecified-high - adjudicated reference answers for batch quality control.
  What to do: `honeypots.py` builds answers for the frozen `outputs/lists/honeypot_list.json` from todo 7 (verify its SHA-256; 20 triage honeypots and 30 decision honeypots, disjoint from gold and pilot); answers built like gold (two independent agents + adjudicator, SAM-snapped where geometry is involved); store with SHA-256 in `outputs/honeypots/`; decision honeypots are rendered exactly like real proposal sheets so reviewers cannot tell them apart.
  Must NOT do: reuse gold images; reveal honeypot identity in renders or file names given to reviewers.
  Closes: GAP-7 (batch quality control)
  Parallelization: Wave 2 | Blocked by: 1, 4, 5, 7 | Blocks: 6, 11
  References: draft architect batching section; todo 7 protocol and frozen lists.
  Acceptance criteria: 20 + 30 honeypots with adjudicated answers and hashes; overlap with gold list = 0; renders contain no honeypot marker.
  QA scenarios: happy - list counts and disjointness check PASS; failure - a fixture honeypot also in the gold list is rejected. Evidence .omo/evidence/public-label-rework/task-18.txt
  Commit: N | workspace is not a git repository

- [x] 19. Package builder module
  Recommended task executor category: quick - assembles files from an applied tree.
  What to do: `package.py` building the delivery layout (images/public, needs_human, exclusions.csv, changes.csv, before_after samples, sources, README placeholder rendered from a template with numbers read from `metrics.json` and `status.csv`) for any applied tree, used by todo 13 (pilot mini-package) and todo 17 (final); ZIP writer with ASCII entry names and CRC/SHA-256 verification.
  Must NOT do: write prose numbers by hand; place renders inside images/public.
  Closes: GAP-8 (mechanism)
  Parallelization: Wave 2 | Blocked by: 2, 9 | Blocks: 13, 17
  References: `label_review/archive_verification.json`; todo 9 outputs.
  Acceptance criteria: smoke builds a package from a fixture tree; ZIP reparse 100%; README numbers equal `status.csv` counts.
  QA scenarios: happy - fixture package builds and verifies; failure - a fixture with a needs_human image inside images/public is refused. Evidence .omo/evidence/public-label-rework/task-19.txt
  Commit: N | workspace is not a git repository

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
  Recommended task executor category: unspecified-high - checklist audit against artifacts.
  Steps: for every Must have / Must NOT have bullet, record PASS/FAIL with evidence in `.omo/evidence/public-label-rework/F1.md`. Required checks: `$PY -B -m public_label_rework.protect verify` prints `CHANGED []`; validators (todo 12) PASS on the delivered tree; `find public_label_rework/outputs -iname 'harvest_*' -not -path '*harvest_convention*'` returns nothing; `pilot/user_confirmation.json` timestamp precedes the earliest batch timestamp of `remaining_list` images in `batches.jsonl`; triage/proposal/decision files created before that timestamp reference only `pilot_list.json` ids, gold ids or honeypot ids; `verification.json` has `gfull_evaluations == 1` and `final_tree.sha256` written before it; `weights/weights.json` lists only `sam2.1_b.pt` plus training outputs and fold args show `amp: false`; no top-level JSON key additions (validator key-equivalence PASS).
  Pass condition: every bullet PASS.
- [ ] F2. Code quality review
  Recommended task executor category: unspecified-high - code review with tooling.
  Steps: run `$PY -B -m public_label_rework.smoke` (exit 0), `/home/sc/.cache/potato-baseline/.venv/bin/ruff check --no-cache public_label_rework` (0 findings), `ruff format --check --no-cache public_label_rework` (clean), basedpyright CLI (0 errors; list warnings unsuppressed); grep `public_label_rework` for `type: ignore`, `cast(`, `except Exception` (0 hits); confirm every module that processes images accepts an explicit list file and refuses ids outside it (read code + one fixture run); confirm apply is deterministic by re-running apply on the final decisions into a temp dir and comparing SHA-256 of every file. Record in `.omo/evidence/public-label-rework/F2.md`.
  Pass condition: all commands meet their stated result and determinism check identical.
- [ ] F3. Real manual QA
  Recommended task executor category: unspecified-high - visual inspection of the delivered package.
  Steps: extract the delivery ZIP to a fresh `/tmp/plr-f3-<uuid>/`; with the todo 5 renderer, render from the extracted files (seed 2026) 30 kept images stratified by source (at least 6 original02, 5 with >= 11 boxes, 3 with `background_potatoes: true`), all images listed in `needs_human/` up to 10, and 10 rows of `exclusions.csv`; open every render with the image Read tool. Per kept image check: every in-focus potato has exactly one box; box edges touch the potato outline with no visible background margin and no clipped potato part; no box on a non-potato; out-of-focus or tiny background potatoes are unboxed only when the flag is true. Per needs_human/excluded image check that the stated reason is visibly true. Also confirm README numbers equal `status.csv` counts. Record per-image verdicts in `.omo/evidence/public-label-rework/F3.md` with the render paths.
  Pass condition: 0 defects on the 30 kept images; every needs_human/excluded reason confirmed; README numbers match.
- [ ] F4. Ideal-state fidelity
  Recommended task executor category: unspecified-high - one-to-one IS audit.
  Steps: for each row IS-1..IS-11 in Success criteria, open the listed evidence file, confirm the named PASS token or condition is present and produced from the final frozen tree (`final_tree.sha256` matches), and write a row verdict to `.omo/evidence/public-label-rework/F4.md`; any row without proof becomes a new `- [ ] N.` todo, never a note.
  Pass condition: 11/11 rows PASS.

## Commit strategy
The workspace is not a git repository (`git status` exits 128). No commits; every todo writes evidence under `.omo/evidence/public-label-rework/` and outputs under new-only `public_label_rework/outputs/` folders.

## Success criteria
> One row per IS row. The plan is complete only when every IS row has a delivering todo and a proving QA scenario; F4 checks the delivered behavior against these rows 1:1, and a shortfall becomes new `- [ ] N.` rows, never a note.
| IS | Delivering todo(s) | Proving QA scenario | Evidence |
| --- | --- | --- | --- |
| IS-1 | 3, 10, 11, 13, 15, 16 | 16 G-full P/R >= 0.97 @0.5 PASS, duplicates = 0 PASS, blind audit defect-free >= 95% | .omo/evidence/public-label-rework/task-16.md |
| IS-2 | 4, 10, 12, 16 | 16 G-full median IoU >= 0.85, area ratio 0.95-1.05, p90 <= 1.15, IoU<0.7 share <= 5% all PASS; 12 no tight=false box in images/public | .omo/evidence/public-label-rework/task-16.md |
| IS-3 | 6, 9, 11, 17 | 17 exclusions.csv and needs_human reasons valid; 16 blind audit finds no partially labelled kept image | .omo/evidence/public-label-rework/task-17.txt |
| IS-4 | 2, 9, 12, 17, 19 | 17 ZIP reparse + YOLO conversion 0 errors; 12 key-set equivalence and image hash equality PASS | .omo/evidence/public-label-rework/task-17.txt |
| IS-5 | 1, 9, 15, 17 | 17 protect verify CHANGED []; 15 before/after renders for every changed image | .omo/evidence/public-label-rework/task-17.txt |
| IS-6 | 9, 12, 17 | 12 flags assertions PASS on every output JSON; 17 README provenance section read PASS | .omo/evidence/public-label-rework/task-12.txt |
| IS-7 | 1, 3, 17 | 1 harvest never extracted; 17 protect verify CHANGED [] | .omo/evidence/public-label-rework/task-1.txt |
| IS-8 | 3, 11, 16 | 16 zero family mismatches without an allowed explanation code | .omo/evidence/public-label-rework/task-16.md |
| IS-9 | 12, 13, 14, 16 | 14 confirmation file exists; 16 all G-full PASS tokens | .omo/evidence/public-label-rework/task-14.txt |
| IS-10 | 3, 16, 17 | 3 harvest_convention.json verdict present; 17 README deviation section read PASS | .omo/evidence/public-label-rework/task-3.txt |
| IS-11 | 1, 6, 17 | 1 license lines per source PASS; 6 watermark flags; 17 README license section read PASS | .omo/evidence/public-label-rework/task-17.txt |
