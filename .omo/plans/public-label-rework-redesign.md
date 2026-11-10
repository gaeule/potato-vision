# public-label-rework-redesign - Work Plan

## TL;DR (For humans)

Replace further SAM prompt experiments with a reference-first, CPU contour feasibility test. Independent reviewers first decide what body boundary the original image supports. Only if that reference gate survives, try one frozen full-frame color segmentation and bounded protrusion-cut candidate policy on the same32 cases. A candidate is never a correct label merely because its numerical guards pass.

This plan delivers a defensible STOP or evidence supporting a separately authorized full qualification, not corrected2060 labels. The original delivery plan remains paused,7/23 complete. Keep90%,375 targets and338 required successes. No new model calls, installs, training, source edits or pilot processing are included.

Effort: Large. Risk: High; only one failure is affordable among 339 potentially eligible targets. This method is an untested hypothesis. Existing source-boundary uncertainty makes STOP_REFERENCE_UNAVAILABLE a substantial risk before algorithm work.

Status: planning authorized; execution not authorized. Review required: independent design review; native plan-reviewer approval unavailable under this session's prior-execution restriction. Do not label advisory review as native approval.

## Scope

### Affected user and ideal state

The user and dataset provider need one tight visible-body box per identifiable potato, without sprouts, neighbors, invented hidden boundaries or false success claims.

| ID | Ideal state / gap | Consequence |
|---|---|---|
| IS1 / GAP1 | Same375 denominator,338 successes required;36 fixed failures leave only one additional failure | Never make exclusions improve qualification |
| IS2 / GAP2 | Complete visible body, no sprouts/neighbors; current crop loses C022 body and appendage probes fail | Replace method-specific cropping, not quality requirements |
| IS3 / GAP3 | Independent references; current375 repeatedly exposed and gold depends on SAM | Raw-only reference authors, honest development-exposed designation |
| IS4 / GAP4 | Bounded, auditable decision; repeated tuning has not qualified | One32-case diagnostic; immediate mathematical/semantic stops |
| IS5 / GAP5 | Original labels, image bytes, completed work and pilot STOP preserved | New-only artifacts; no production integration in this plan |

### Immutable baseline

Workspace `/mnt/c/Users/USER/Desktop/협력/potato`; existing Python `/home/sc/.cache/potato-baseline/.venv/bin/python` (`$PY`). No Git repository or commits.

Original plan `.omo/plans/public-label-rework.md` remains the historical execution plan. Its Tasks1,2,3,5,8,9,19 remain complete. Task4 is not satisfied by this plan. Preserve all20 original images,375 CASE IDs, original points/statuses/full negatives, old outputs,339 effective-ready/36 non-success accounting, including null7. Keep only the already approved effective-negative exceptions for0014/T037 and0017/T048; do not add exceptions.

Reuse the exact32 targets in `public_label_rework/outputs/snap_instance_v4/human_guidance_v2/sample.json`, SHA256 `82843974d751c7a2750959e543e1167fab3ca12a9558cfd3787d4ca609220e34`. Of these25 are eligible and7 null. For every eligible target, use **only its human_v2_20261102_01 generate raw mask** and actual source image as the initialization input. All25 exist, including the two rejected candidates. No run_v2/human-v2 mixing or best-mask selection. Bind actual source, input, receipt and raw bytes before processing.

Evidence: `human-guidance-v2-result-20261102.md`, `human-guidance-v2-run-gate.md`, `human-guidance-v2-visual-review.md` under `.omo/evidence/public-label-rework/`.

### Proposed contract changes — require explicit execution approval

1. SAM becomes a cached proposal source, not the only geometry authority. New source is `cpu_body_contour_v1`; never impersonate `sam`/`sam_point`.
2. New candidates are not cropped to the old10% padded box. Retain image/content bounds and original pixels. Existing SAM code and receipts remain unchanged.
3. Old IoU>=0.5, area/input-box>=0.3 and padded-border<=2 are retained as historical diagnostics, **not acceptance guards for this new method**. Replace them with the point, extent, identity and independent body-quality conditions below. Removing a guard never converts an old failure into a success.
4. Allow raw-image boundary annotation for independent reference artifacts only. Production coordinates remain tool-generated. Reference authors never see candidate masks, old verdicts or each other's annotations before their initial decisions.
5. Explicitly permit frozen protrusion-removal candidates, not post-result erosion to satisfy guards. Preserve base mask, removed pixels and all candidates; no uniform opening, convex hull or fitted ellipse.
6. No adapter to current `decisions.py` is implemented here. Its accepted-source and area checks must not be weakened as a shortcut.

Execution approval must name this plan and accept these six changes. Merely approving the planning work or receiving an automatic old-plan continuation is not execution approval.

### Reference contract

For each eligible target, two independent authors view source/full context and immutable original identity point, but no algorithm output. They provide a source-space visible-body polygon/multipolygon, explicit body-versus-sprout notes and S/F/U observability. Disconnected visible parts are allowed; hidden body is not invented. Authoring geometry is reference-only and never copied into production candidates.

If either author reports uncertainty or the two annotations differ in identity, lobe coverage, appendage membership or visible extent, one independent adjudicator may select one existing annotation or return U using the source and both records; no third polygon or redrawing is allowed. No further rounds. A reference is usable only with an explicit resolved body interpretation; otherwise U. Rasterize without resampling source pixels and record author/adjudicator provenance; it is AI reference, never human ground truth. Do not impose a polygon-overlap threshold that could forgive a missing narrow body lobe.

Reference authors receive only source images, target IDs and original identity points: no supplementary points, human50 judgments, cached masks, prior verdicts or reference-gate artifacts. They evaluate visible body boundaries, not rewrite the fixed identity/disposition ledger.

Reference-only data are inaccessible to the candidate generator. Conflicts with frozen effective/supplementary point constraints are non-success, not permission to edit points. Null7 receive no polygon or invented point and remain fixed failures.

### Frozen candidate policy

One candidate-generation invocation per eligible target, zero neural-model calls:

1. Full-frame GrabCut mask initialization: cached generate raw foreground becomes probable foreground; remaining content becomes probable background. Existing positive/supplementary positives override probable labels as hard foreground; established non-content bars override probable labels as hard background. Effective-full/supplementary negatives do NOT alter these labels: they remain independent output constraints. If any pixel is both a required positive and a negative, fail with contradictory_labels before processing. Points outside content, missing foreground/background support or invalid source/raw binding fail without fallback. Do not force the image border or pixels outside an old box to background.
2. Immediately before EACH installed OpenCV mask-initialized GrabCut call, call setRNGSeed(20261102) and setNumThreads(1); use 5 iterations and record seed, threads and versions in its receipt. This is per target, not once per batch. Choose the resulting 8-connected foreground component containing the original positive. Missing positive or required supplementary positives means failure. Other components are recorded, not silently represented as complete body.
3. Emit at most three candidates: uncut base, and independently derived3%/6% neck-width variants. These are hypotheses, not semantic predictions.
4. For each cut variant, let L be the base mask's longer extent dimension. Enumerate pairs of concave boundary vertices (farthest points of convexity defects on the external CHAIN_APPROX_SIMPLE contour); exclude zero-length pairs and pairs farther apart than the variant's fraction of L. Compare unrounded Euclidean lengths; contour coordinates are source-pixel integers. Every pixel of the one-pixel LINE_8 centerline must belong to current foreground. Remove a straight cut band with cv2.line(thickness=3, lineType=LINE_8, shift=0); take the 8-connected component containing the positive. Never restore removed cut-band pixels. A valid proposal preserves every positive and removes exactly one connected protrusion beyond the cut band. The maximum Euclidean distance from a protrusion pixel center to the finite chord segment must be at least twice the chord length.
5. Rank valid removals by descending protrusion area, then lexicographic normalized endpoint coordinates. Apply the first, recompute, at most8 cuts per variant. Process widths independently from the same uncut base; L stays fixed. Preserve all cut coordinates and intermediate mask hashes. No valid cut means the variant equals base; deduplicate equal final masks, retaining provenance. These algorithm choices are frozen hypotheses, not optimized constants.
6. Candidate bbox is its actual foreground extent, source coordinates, upper bound exclusive. Do not alter it to fit old guards. A candidate whose foreground component contains any effective/supplementary negative is invalid with negative_in_component, retaining the discriminating output check rather than carving holes at negative points. OriginalP may fail only as disclosed by the two authorized exceptions; record originalP separately.

Degenerate contours with fewer than four vertices or fewer than three hull vertices, or the documented non-monotonic-hull convexityDefects failure, produce no cut proposals for that variant: return the unchanged base with defects_unavailable, not an automatic target failure. This is a frozen no-cut branch, not permission to catch unrelated errors or substitute algorithms. Other library errors STOP. Include one-pixel-spur and L<100 fixtures.

Implement exact rounding, pixel-line rasterization and tie behavior using installed library versions, covered by synthetic fixtures before freezing the implementation hash. No real diagnostic image may be used to tune constants or tie rules. Implementation defects may be fixed using synthetic fixtures before freezing; never weaken expected behavior to make tests pass. If the specified method cannot satisfy its fixtures, STOP; do not substitute another algorithm.

### Output and independent acceptance

`ContourProposal`: image/target/candidate IDs; source/input/config/code/raw hashes; stage masks and hashes; exact extent; originalP/effectiveP/supplementaryP; added/removed pixels; failure reason; execution receipt. Also record the cached raw positive-component extent and contained negative-point IDs as diagnostics only, never as a fourth selectable candidate.

`QualityCertificate`: candidate ID and mask hash; reference hash; reviewer IDs; identity_unique, visible_body_complete, sprout_free, neighbor_background_free; verdict S/F/U and concrete evidence.

Two independent candidate reviewers compare source, frozen reference and neutral candidate renders; one adjudication maximum per disagreement. They cannot edit coordinates or masks. Selection is first certified S in fixed order base,3%,6%; preserve verdicts for every generated candidate. Only exact extent + valid point constraints + unique physical target + independent quality S permits `tight=true`. Missing/U never passes. Check same-image candidates jointly for duplicate physical bodies; overlapping pixels alone neither prove nor disprove duplication.

## Verification strategy

Test-first deterministic fixtures, then exactly one real diagnostic. No timing sleeps. Reference creation precedes algorithm processing. Freeze configuration, raw mapping, sample, dependency versions and code before any real candidate output.

Required probes (all must become S):0014/T037;0002/T001;0005/T001;0015/T002;0019/T015;0020/T024;0008/T034;0019/T040. Image prefix is `public_original06_`.

The 12 prior-success controls are: 0001/T012, 0001/T060, 0003/T001, 0004/T001, 0006/T001, 0008/T010, 0017/T048, 0019/T022, 0020/T005, 0020/T010, 0020/T018, 0020/T019 (same image prefix). These differ from sample.json's four-member controls group. Freeze this exact list and the human-guidance-v2-visual-review.md hash in prepare. All 12 must remain S under the new reference. They are controls, not truth copied into references. Retain all 32 results, including null7. Necessity gate: at least 24/25 eligible successes, all eight required probes S, all 12 controls S, no integrity failures.

At reference or candidate review, a required/control target F/U stops the method. Two distinct non-successes among the25 eligible also stop immediately:36+2 failures imply maximum337<338. Stop before further candidate processing/review; retain unprocessed IDs with NOT_RUN_AFTER_STOP, never substitute targets or manufacture verdicts.

Budgets:32 fixed targets;25 GrabCut invocations,5 internal iterations each;<=75 candidate masks;<=75 initial/reference-adjudication decisions and<=225 candidate/adjudication decisions. Zero parameter revisions, extra samples, neural calls, training, installs or real-case retries. Reproducibility repeats use synthetic fixtures only. Reuse completed units only with exact receipt validation; partial/conflicting units STOP without overwriting or automatic re-call.

Terminal statuses: STOP_REFERENCE_UNAVAILABLE; STOP_METHOD_INFEASIBLE_UNDER_FROZEN_CONTRACT; STOP_INTEGRITY_OR_EXECUTION_FAILURE; FEASIBLE_FOR_SEPARATE_QUALIFICATION_REVIEW. None means Task4 passed. The375 cohort remains development-exposed; an unseen family-isolated G-full is still required by the original delivery plan.

## Execution strategy

Three waves: (1) freeze/independent references, (2) conditional CPU implementation and bounded diagnostic, (3) report and independent final verification. Dependency chain1→2→3→4→5; F1–F4 after5. A STOP from any task bypasses subsequent algorithm work and goes directly to5 and the final verification wave.

Future writes only to new `public_label_rework/contour_diagnostic.py`, isolated supporting modules/tests with `contour_` prefix, new `outputs/contour_diagnostic_v1/<run-id>/`, and new `.omo/evidence/public-label-rework-redesign/` evidence. Refuse existing output paths except verified completed-unit reuse. Existing SAM/decisions/labels/weights/code are read-only. `$PY -B -m public_label_rework.contour_diagnostic <subcommand>` is a **planned CLI**, not an existing verified command.

## Todos

- [ ] 1. Freeze contracts and validate the cached diagnostic inputs
  Recommended task executor category: deep-low - explicit bindings and data accounting.
  References: original plan Task4; human-v2 policy, sample, run records and run-gate; new scope above.
  Work: implement `prepare`; bind375 roster/36 failures/25 cached raws/32 sample, source and code versions; record six-change user approval before any processing.
  Acceptance: `$PY -B -m public_label_rework.contour_diagnostic prepare` yields all expected counts and hashes, zero model imports/calls or source changes.
  QA: valid cache passes; altered sample, missing raw, unauthorized override or absent approval rejects before processing. Evidence `01-preflight.json`.
  Commit: N.

- [ ] 2. Establish independent raw-image references and the information limit
  Recommended task executor category: deep-low - bounded independent image judgments.
  References: reference contract; visual-review32 mapping; original Tasks7/18 independence.
  Work: dispatch isolated source-only author roles and at most one adjudicator per target; validate/rasterize reference artifacts; retain null7 and all unprocessed IDs. Check required probes first, then controls, then other eligible IDs in image/target order.
  Acceptance: `validate-references` verifies source binding, role isolation, budget, point conflicts and early-stop arithmetic; references are frozen before Task3.
  QA: a mandatory U yields STOP without algorithm work; two non-required non-successes yield upper337; a conflicting reference/negative point cannot be silently repaired. Evidence `02-reference-gate.json`.
  Commit: N.

- [ ] 3. Implement and fixture-test the frozen CPU candidate generator
  Recommended task executor category: deep-low - deterministic segmentation and proposal geometry.
  References: candidate policy above; `snap.py:164-202` historical clipping counterexample; `decisions.py:189-229` incompatible receipt guard.
  Work: implement `propose`, typed proposal records and synthetic tests; no production adapter. The Task3 executor receives only the frozen policy and synthetic fixtures: it must not open 02-reference-gate.json, reference polygons, actual raw masks or source images. An orchestrator provides a go/no-go signal without reference content. Freeze config/code after synthetic checks, before Task4 opens real inputs.
  Acceptance: `$PY -B -m public_label_rework.contour_smoke` and existing Ruff/basedpyright commands exit0. Ellipse+stem, two touching bodies, real-lobe analog, thin crescent, edge-truncated body, positive erased, tied cuts and conflicting labels have explicit assertions. Source images remain untouched.
  QA: identical fixture inputs yield identical masks/receipts; a narrow real lobe is never automatically certified; invalid candidate cannot acquire `tight=true`; no-fit case STOP, no fallback. Evidence `03-fixtures.json`.
  Commit: N.

- [ ] 4. Run one bounded diagnostic and independently certify its candidates
  Recommended task executor category: deep-low - controlled CPU execution with separate visual reviewers.
  References: frozen policy, references and budgets; human-v2 comparison report.
  Work: `run` and `validate-results`; process required probes then controls then remaining IDs; independent certificate roles; stop at first failed mandatory/control case or second eligible non-success. Candidate generation never sees reference polygons.
  Acceptance: actual calls/candidates/decisions within bounds; every32 ID accounted; full raw/base/candidate/source evidence; no unrecorded retries; qualification feasibility only if all necessity conditions hold.
  QA: inject mask/hash mismatch and verify no success; no-call/U/early-stop accounting retained; source/cached outputs rehash unchanged; view actual masks, not just scores. Evidence `04-diagnostic.json` and visual report.
  Commit: N.

- [ ] 5. Report the result and specify downstream integration impacts without executing them
  Recommended task executor category: writing - evidence-based handoff and synchronized documentation.
  References: Tasks1–4 receipts; original Tasks4,7,9–14,18,19.
  Work: `report` produces Korean summary, full outcome ledger, conservation hashes and terminal status. For each newly certified S, include old IoU/area/padded-border diagnostic columns against the preserved old input/padding; if no old candidate box exists, record NOT_APPLICABLE rather than manufacturing one. Explain which accepted outputs depend on the changed method guards. Even if feasible, stop for separate full375 qualification authorization; do not launch it.
  Acceptance: numbers reconcile, terminal evidence paths open, no Task4 or pilot completion claim. English/Korean conclusions agree.
  QA: report distinguishes32 sample,25 eligible and375 cohort; STOP paths include unprocessed work without fake results. Evidence `05-summary.md`.
  Commit: N.

## Final verification wave

- [ ] F1. Audit scope, unchanged denominator and actual invocation budgets
  Independent reviewer; verify all artifact bindings and STOP conditions. Evidence `F1-scope.md`. Commit: N.
- [ ] F2. Review CPU code, deterministic fixtures and strict failure propagation
  Independent reviewer; inspect implementation and actual static/test exits, not producer summaries. If reference STOP prevented Task3, record its implementation/tests as NOT_RUN with that gate evidence, not a fabricated test PASS. Evidence `F2-code.md`. Commit: N.
- [ ] F3. Inspect real source, references and masks with the documented limitations
  Independent reviewer; inspect every generated result or explicitly account for STOP before generation; do not invent masks for skipped targets. Evidence `F3-visual.md`. Commit: N.
- [ ] F4. Verify honest outcome and preserve the original delivery gates
  Independent reviewer; compare IS1–IS5 to evidence. A valid STOP completes the diagnostic work, not Task4 or the dataset. Evidence `F4-fidelity.md`. Commit: N.

## Commit strategy

No commits or PRs; non-Git direct-local workspace. No production changes in this diagnostic plan.

## Success criteria

| Property | Tasks / evidence |
|---|---|
| IS1 fixed accounting |1,2,4,F1:375/338 and36 baseline failures unchanged |
| IS2 body quality |2–4,F3: all mandatory probes/controls certified or explicit STOP |
| IS3 independence |2,4,F3: raw-only references, generator isolation, separate candidate reviewers |
| IS4 bounded decision |1,4,5,F1: actual budgets, no tuning, one diagnostic |
| IS5 preservation |1,5,F4: original hashes, no pilot/inference expansion |

### Downstream continuation map — not authorized by this plan

Original Task4 can close only after a separately approved full375 evaluation with actual>=338 qualified targets and independent review under an explicitly approved new method contract. A feasible32-case result is insufficient.

Before returning to the original pipeline, explicitly redesign Task7 gold and Task18 honeypots to use independent references rather than the tested generator as truth; adapt Task9 receipt validation and Tasks10/11 proposal/review consumers to a versioned new source/certificate without forging SAM provenance; adapt Task12 metrics and Task19 packaging/reporting accordingly. Their prior completed results remain historical, not automatically valid for new source semantics. No such production adapter or gold creation is performed here.

Then the original100-image pilot (Tasks13/14) still requires all original dependencies and user STOP. The remaining1960 processing, final audits and delivery remain gated. No new reference image may overlap the eventual G-full by source/augmentation family; if no untouched family-disjoint set is available, report that limitation and stop rather than relabel exposed data as holdout.
