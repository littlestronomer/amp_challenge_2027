# Competition readiness and evidence plan

Date: 2026-09-21. Baseline inspected: `ccf8ef6` plus pre-existing local changes
to `pyproject.toml` and `uv.lock`. The local implementation is ready for SSH
execution. Implemented code has had syntax and lint checks; remote generation,
old experiment inventory and scientific review remain pending.

| Task | Status |
|---|---|
| 0 baseline/worklog | Implemented locally |
| 1 evidence inventory | Implemented locally; run on SSH |
| 2 remote report | Template and runbook ready; SSH review pending |
| 3 claim correction | Abstract and claim register updated conservatively |
| 4 strict generation | Implemented locally; full-size SSH run pending |
| 5 top-100 audit | Implemented locally; run on strict output on SSH |
| 6 competitive/independent evaluation | External-data gate added; official score and external dataset unavailable |
| 7 choose intervention | Pending verified evidence |
| 8 remote experiment/release | Pending evidence and one justified decision |

## 1. Objective and definition of completion

Produce a defensible submission and a documented decision about the next
scientific improvement. Establish separately:

1. Whether the packaged submission executes the intended models reproducibly.
2. What the computational evidence says about competition standing.
3. What independent evidence supports the selected peptides' predicted quality.

Completion does not require beating the incumbent or claiming likely wet-lab
selection. A justified decision to retain the incumbent is a successful result.
Do not assign a numerical selection probability without a defensible basis.

Required final deliverables:

- A hash-linked evidence inventory and a readable status report.
- A corrected abstract and disclosure with claims tied to actual evidence.
- An opt-in strict generation path and recorded release validation.
- A report assessing the entire top-100 without treating predictions as assays.
- One decision memo: retain incumbent, improve labels/predictors, improve
  selection, or propose a generator experiment.
- If evidence supports an experiment, a frozen protocol and a paired result.

## 2. Instructions for the implementing model

Work on one numbered task at a time. Read its listed files before editing.
Keep changes small enough to review independently. After each task report:
files changed, checks run, result, and the next task. Do not silently broaden
scope to resolve scientific uncertainty through more training.

### Immutable inputs and boundaries

- Preserve the existing uncommitted Jev dependency edits. Do not revert,
  complete, or commit them incidentally.
- Do not modify `checkpoint/`, existing training CSVs, old experiment outputs,
  historical manifests, or the official reference FASTA.
- Preserve default generation bytes and scoring weights during engineering
  work. New strict behavior is opt-in until a separate release decision.
- No automatic checkpoint promotion, Kaggle upload, access grants, or publishing.
- Local work: code, fixtures, unit tests, documentation, read-only analysis.
- Real generation, embeddings, training and large comparisons: existing SSH
  machine, using the user's established execution workflow. Do not initiate SSH
  access or assume remote files are present locally.
- Existing completed test results may be summarized. Do not rerun tuning against
  them, change their splits, or describe them as an untouched final test.
- Preserve missing, conflicting and unsupported observations explicitly. Do not
  turn unknown labels into negative labels or missing metrics into zero.
- Do not add Jev/Laya, a larger backbone, RL, or new generators in Tasks 0–7.
- Avoid dependency changes. Use standard library code for evidence reporting.
- Do not spawn agents unless separately requested or required by applicable
  repository instructions.

### Before starting

Run `git status --short` and read applicable `AGENTS.md` files. Inspect current
code rather than relying on this document's line positions. The working tree
may have advanced since this plan was written.

Use the existing environment. Typical local checks:

```bash
.venv/bin/python -m pytest -q tests/test_inference_parity_bounds.py tests/test_pipeline.py tests/test_reproducibility.py
```

These 20 tests passed during the preceding assessment; run them again only when
relevant code changes. Never install or upgrade the environment just to repeat
an unchanged baseline test.

## 3. Current evidence and limitations

The incumbent uses two trained autoregressive generators with a 3:1 blend,
ESM2-35M activity/panel heads, property/distribution components, novelty checks
and diversity selection. Hemolysis selection weight is zero.

`PHASE1_RESULTS.md` records a September 5 cold-clone validity and repeatability
pass. It reports hybrid 650M metrics including FBD 0.221 and MMD 0.357. Treat
these as historical, protocol-specific measurements until source artifacts are
verified; do not infer current leaderboard rank from them.

Historical random-split predictor evaluations have exact/near-sequence overlap.
Later runbooks describe completed generalization work and an already-inspected
test; older runbooks still say the benchmark is pending. Resolve this discrepancy
from remote artifacts, not document timestamps or optimistic interpretation.

The current abstract contains superseded claims about calibrated probabilities,
bounded homolog leakage and lack of concentrated safety risk. Do not reuse it
unchanged. The current evaluation script does not implement the hidden official
aggregation score. The raw-generator external benchmark is not a comparison of
complete native starter-kit submissions.

Official sources checked during assessment:

- https://www.kaggle.com/competitions/amp-challenge/overview
- https://github.com/szczurek-lab/amp-challenge-2027
- https://szczurek-lab.github.io/amp-challenge-website/

The indexed competition overview describes advancement of up to 20 teams and
random selection of 25 peptides from each advancing team's top-100. Recheck
current rules before release; cached web text is not a submission receipt.

## 4. Task 0 — Freeze the working baseline

Read:

- `src/amp_challenge_2027/generate.py`
- `src/amp_challenge_2027/pipeline.py`
- `src/amp_challenge_2027/inference_metadata.py` and its JSON registry
- `scripts/experiment_utils.py`
- `docs/RUNBOOK_INFERENCE_AND_LABEL_AUDIT.md`

Create **proposed** `docs/COMPETITION_WORKLOG.md` with a task checklist and a
short baseline entry. Record HEAD, dirty filenames and the intended incumbent
recipe. Record hashes of checkpoint files and effective hash-bound inference
metadata when producing the remote evidence bundle, not guessed historical
settings from shared checkpoint configs.

Acceptance: no source, model, data, dependency or default-recipe changes.

## 5. Task 1 — Implement the evidence inventory

Create these **proposed** files:

- `src/amp_challenge_2027/evidence_inventory.py`: pure, CPU-only readers,
  validation and deterministic report generation.
- `scripts/collect_competition_evidence.py`: CLI wrapper.
- `experiments/competition_evidence_v1.json`: explicit source list.
- `tests/test_evidence_inventory.py`: small synthetic fixtures in temporary dirs.

Do not recursively ingest every result directory. A source config entry contains:

```json
{
  "id": "family_benchmark_test",
  "kind": "family_benchmark",
  "root": "sweep_results/reward-generalization-test-v1",
  "required": false,
  "inspection_status": "already_inspected",
  "files": ["results.csv", "paired_architecture_deltas.json"],
  "marker": null
}
```

Root paths resolve relative to `--repo-root`, not the config's directory.
File paths resolve inside that root. Reject absolute/escaping file entries and
symlinks resolving outside the configured source root. IDs must be unique.
Discover actual marker names/schemas from each producer before specifying a
marker; never fabricate one. A null marker means provenance is incomplete,
even if current content hashes can be computed.

Initially support generic JSON/CSV/Markdown inventory. Add typed metric adapters
only for verified schemas; unknown schemas remain inventoried and uninterpreted.
Never search arbitrary fields for a number that resembles an expected metric.

Candidate inputs to inventory, subject to actual remote discovery:

| Evidence | Existing source/location to inspect |
|---|---|
| Historical outcomes | `PHASE1_RESULTS.md` |
| Effective heads | `checkpoint/reward/`, `checkpoint/reward_hemo/`, metadata registry |
| Inference checks | `sweep_results/inference-metadata-v1/report.json`, `inference-probes-v1/report.json` |
| Family benchmark | `reward-generalization-test-v1/results.csv`, `paired_architecture_deltas.json`, task/architecture `per_output_metrics.csv` |
| Observation audit | `label-observations-v1/report.json` |
| Metadata audit | `metadata-reconciliation-v1/report.json` |
| Molar-label candidate audit | Inspect `scripts/build_molar_candidates.py` for outputs |
| Label ablation | `label-ablation-fit-v1/validation_summary.csv`, `validation_deltas.csv`, `validation_seeds.csv` |
| Selection diagnosis | `selection-diagnosis-v1/baseline_results.csv`, `results.csv`, `inventory.json` |
| Top-100 comparison | Inspect `scripts/compare_top100.py` and its runbook |
| External baseline/pool accounting | Inspect producers and actual run roots; do not guess latest version |
| Official standing | User-provided submission receipt/score, initially unavailable |

Reuse the concepts in `experiment_utils.sha256`, `verify_files`, and `mark_files`.
Avoid introducing package imports from `scripts/`; either keep script helpers
at the wrapper boundary or implement the small necessary pure hashing helper.

Output into a new directory only:

- `inventory.json`: configured sources, resolved paths, file hashes, sizes,
  marker verification, available producer provenance and failure reasons.
- `metrics.csv`: only supported parsed measurements.
- `STATUS.md`: engineering, predictor evidence, selection evidence, competition
  standing, missing items and next actions.
- `complete.json`: hashes of the report files, written last.

Each metric row must retain source ID, relative file, row/key locator, task,
model/recipe identity, population/split, metric name, value and evidence status.
Use null/empty for unavailable dimensions, never infer them from a directory name.
Use a record index or row hash if the CSV lacks a unique key.

Distinguish statuses: `verified`, `unverified`, `missing`, `invalid`, `unsupported`.
Hash verification means file integrity, not scientific correctness. Do not label
an experiment successful just because its completion marker is intact. Verify
nested artifact dependencies when the producer contract requires them.

Exit behavior: 0 when inventory completes with no invalid inputs or missing
required source; 2 after writing diagnostics if either occurs; 1 for malformed
configuration or output-write failure. Optional absent inputs are reported, not
silently omitted. Existing output directories are refused.

Proposed CLI after implementation:

```bash
uv run --no-sync python scripts/collect_competition_evidence.py \
  --repo-root . --config experiments/competition_evidence_v1.json \
  --out sweep_results/competition-evidence-v1
```

Required tests: valid source, missing optional/required source, corrupt hash,
unsupported schema, duplicate ID, path escape, non-finite metric, mismatched
population exclusion from paired comparisons, deterministic report ordering,
output overwrite refusal, and no changes to input hashes.

Acceptance: a local checkout with no remote results produces an honest partial
report. The collector performs no network requests, model loading or training.

## 6. Task 2 — Collect remote evidence and make a status decision

Create **proposed** `docs/RUNBOOK_COMPETITION_READINESS.md` with the Task 1 command,
expected outputs and troubleshooting. Run collection on SSH through the existing
user workflow, then inspect the report and referenced small summary artifacts.

If remote results are unavailable, complete independent Tasks 3–4 and leave
scientific decisions explicitly pending. Do not recreate expensive experiments
merely because their outputs were not copied to this laptop.

Write **proposed** `docs/COMPETITION_STATUS.md` with:

1. What ran, what passed, what failed, what is unknown.
2. Predictor metrics and support by task/output; raw versus calibrated results.
3. MLP versus linear paired results, without comparing incompatible populations.
4. Label-ablation effects separated into filtering, relabeling and expansion.
5. Incumbent versus candidate library/selection trade-offs.
6. Official-score status and baseline-comparison limitations.
7. One highest-priority unresolved issue and the evidence required to resolve it.

Gate A: do not select a scientific intervention until this report establishes
the relevant evidence. A partial report may conclude only that more evidence is
needed. No mandatory AUROC threshold is invented in this plan.

## 7. Task 3 — Correct submission claims

Edit `docs/ABSTRACT.md` and, where necessary, `docs/DATA_DISCLOSURE.md`.
Add **proposed** `docs/CLAIMS_REGISTER.md` with columns: claim, source artifact,
population/instrument, limitation, disposition (retain/rewrite/remove/pending).

Concrete corrections:

- Remove the current assertion of no concentrated safety risk unless supported
  by the applicable current audit; never convert model scores into measured safety.
- Replace generic claims of calibrated probabilities with the exact calibration
  method and population, or call them model scores when evidence is insufficient.
- Remove the assertion that one clustered result bounds homolog leakage of the
  deployed heads. Distinguish models, splits and datasets explicitly.
- Say genus-level predicted breadth; do not imply measured 20-strain activity.
- Label FBD/MMD as local distribution measurements using the stated reference
  and embedder, not the official aggregation score.
- Describe data availability accurately. A fetch script or file on SSH does not
  itself mean a training snapshot is publicly released under a verified license.
- Preserve historical results; add corrective context rather than erase history.

Acceptance: every numerical abstract claim has traceable evidence or is omitted.
No speculative replacement performance numbers. This documentation correction
can proceed while remote metric verification remains pending.

## 8. Task 4 — Add strict submission execution

Purpose: a successful release run must mean the intended stack actually ran.
This is an engineering change; it must not change successful generation results.

Read `generate.py`, `pipeline.py`, `score.py`, `inference_metadata.py`, relevant
tests and `scripts/verify_submission.py` before editing.

Implement in small patches:

**4A. Add `--strict` to `generate`.** Default false. In strict mode:

- Require a nonempty reference and all requested model/config artifacts.
- For the default incumbent recipe require both shipped generators and the 3:1
  blend. Never silently disable the secondary. Explicit alternate recipes must
  identify themselves and cannot be reported as the incumbent.
- Raise on generator failure instead of invoking `generate_fallback`.
- Reject missing nonzero-weight scorer components. Safety weight zero does not
  require loading the hemolysis head.
- Check actual computed component names and finite outputs, including lazy
  scorer failures. Inspect `CompositeScorer.score`; preflight alone is insufficient.
- Require exact requested library/top sizes and standard validity checks before
  publishing completed outputs. Preserve the existing reference novelty rule.
- Make all strict failures exit nonzero with the failing component named.

Thread a default-false strict argument through necessary helpers only. Preserve
legacy permissive behavior for existing callers. Do not refactor model loading
or change numerical inference in the same patch.

**4B. Write `generation_manifest.json` for strict runs.** Include seed, resolved
CLI recipe, checkpoint/config/reference hashes, effective head metadata, actual
scorer components, source identity, runtime versions, device, output hashes and
actual backbone revisions when recoverable. Unknown revisions are explicit and
block a claim of fully pinned execution. Never guess revisions.

Write a success manifest only after all required checks pass. Reject a nonempty
strict output directory; stage outputs inside the new directory and ensure a
failure leaves no success marker. Byte-comparison applies to FASTA outputs,
not timestamps or runtime fields in manifests.

**4C. Extend validator with opt-in `--strict-generation`.** It must run
`uv run --no-sync generate --strict` twice and check success manifests as well
as existing FASTA validity and byte equality. Keep the ordinary official-style
validation path intact. Adapt validator cleanup to avoid strict nonempty-output
conflicts; do not weaken the strict rule to make the second run succeed.

Required tests with tiny/mocked components:

- Missing reference, primary or secondary checkpoint causes strict failure.
- Generator exception cannot produce fallback output under strict mode.
- Missing/lazily failing requested scorer causes strict failure.
- NaN/Inf scores and undersized output fail before success marking.
- Successful strict and permissive paths produce identical FASTA bytes under
  identical synthetic inputs and recipe.
- A zero-weight safety component is optional.
- Manifest hashes detect output/metadata tampering.
- Existing permissive behavior and relevant regression tests still pass.

Acceptance: no changes to checkpoints, defaults, weights or scientific selection
policy; strict runs cannot silently report a reduced pipeline as success.

## 9. Task 5 — Audit the entire incumbent top-100

Prefer existing `compare_top100.py`, `selection_audit.py`, `select.py`,
`props.py` and pool-accounting outputs. Do not create another scoring pipeline.
If a consolidated adapter is needed, add **proposed**
`scripts/report_top100_readiness.py` and corresponding pure-function tests.

Inputs: exact ordered library/top FASTAs, reference, verified score artifacts,
effective scorer identities, and optional independently produced assessment
tables. Join by canonical sequence plus source identity, never row position
alone. Duplicate/conflicting or missing required rows are errors.

Report:

- Validity, membership, uniqueness and exact maximum reference similarity.
- Sequence-family redundancy and pairwise diversity using a declared method.
- Score distributions and lower/upper tails; per-genus outputs where available.
- Predictor disagreement only across genuinely different identified models.
- Hemolysis scores as predictions, not safety certification or therapeutic index.
- Available synthesis/solubility assessments with source and limitations;
  otherwise explicitly `not_assessed`. Do not invent a heuristic and label it
  experimentally validated synthesizability.
- Coverage of all 100 records and missing evidence per record.

Optional diagnostic: for fixed per-peptide surrogate values, simulate selection
of 25 without replacement using a fixed RNG seed and 10,000 draws. Report only
the distribution of subset mean surrogate scores and family representation.
Label it sampling variability conditional on fixed predictions. It is NOT a
predicted wet-lab success probability and does not model biological uncertainty.

Outputs: `candidates.csv`, `summary.json`, `REPORT.md`, file-hash marker in a
new output directory. Do not alter the top-100 as part of this audit.

Tests: shuffled score rows, duplicate sequence, missing score, invalid top member,
exact-0.8 reference boundary, deterministic subset sampling and no input writes.

Gate B: identify whether observed weaknesses concern label support, predictor
reliability, candidate availability or selection. A higher predicted score alone
does not settle this question.

## 10. Task 6 — Establish competitive and independent evaluation

Two distinct workstreams; do not combine their conclusions.

### 6A. Competition standing

Prepare a frozen release candidate and a submission checklist from current
official instructions. If the user provides an official score, record submission
ID/date, recipe and output hashes, score, rank and leaderboard snapshot context.
If no score exists, say unavailable. Do not infer rank from entrant counts.
Actual upload requires the user's submission instruction, not merely this plan.

Keep existing raw-generator comparisons. If comparing complete native baselines,
use a separately named protocol with pinned upstream versions and their own
generation/selection recipes; do not relabel raw-generator results as native
pipeline performance. First inventory completed runs to avoid duplicate compute.

### 6B. Independent predictor evidence

Before coding an evaluator, write **proposed**
`experiments/external_predictor_protocol_v1.json` and a short protocol document.
Identify a legally usable experimental source, snapshot hash, label definitions,
assay conditions, modifications/termini, missing-value handling and compatibility
with our task. If no suitable source exists, record the blocker; do not fabricate
independence by repartitioning familiar data.

Audit exact and near overlap against all training/validation/calibration sources
and any previously inspected test. Use existing family utilities and document
source/study relationships. Unknown pretraining exposure remains a limitation.
Define comparators, metrics, sample/family support checks, and analysis before
opening outcomes. Freeze models and calibration; do not fit on the external set.

Use existing metric implementations where applicable. Preserve masked outcomes;
report undefined metrics and uncertainty instead of forcing a complete table.
For paired uncertainty use families/studies as appropriate to the design, not
independent-row assumptions on correlated measurements.

Gate C: meaningful improvement requires independent support and acceptable
uncertainty. No post-hoc split-seed search, outcome-based dataset exclusion, or
calibration fitting on the final evaluation set.

## 11. Task 7 — Choose exactly one next intervention

Write **proposed** `docs/NEXT_EXPERIMENT_DECISION.md`. Include source evidence,
the chosen hypothesis, alternatives deferred, explicit stop criteria and the
minimum experiment that can discriminate the hypothesis.

| Finding | Next intervention |
|---|---|
| Labels/metadata are unreliable | Complete existing molar-label reconstruction and controlled label ablation |
| Predictors lack transferable signal | Improve predictor/data evaluation before selection optimization |
| Credible scores reveal suitable candidates missed by selection | Bounded selector comparison on frozen pools |
| Credible evaluation shows candidate pool itself is inadequate | Separate generator/data protocol |
| Differences are small or uncertain | Retain incumbent; document uncertainty |

Do not automatically run all branches. Do not pick an intervention solely
because an implementation already exists or a model is newly available.

For a selector comparison, reuse `scripts/sweep_top100_selection.py` and its
existing normalization/novelty checks where possible. Predeclare one candidate
policy against the incumbent, compatible metrics, tolerable losses and seeds.
Numeric tolerances need a scientific rationale; do not invent defaults merely
to unblock a run. Reuse frozen libraries and avoid redundant embedding metrics.

For labels/predictors, reuse `run_label_ablation.py` and existing family splits
for development. Preserve the original inspected test status. Any new evaluation
requires genuine independence or an explicitly declared nested design.

Jev/Laya remains optional later: benchmark assay-text classification against
reviewed source records and deterministic rules; split by source/related record,
reserve separate calibration data, and keep model suggestions distinct from
experimental evidence. It is not a replacement for potency measurements.

## 12. Task 8 — Remote experiment and release closeout

Run only the chosen frozen experiment using a new output directory. Measure one
representative cell before estimating runtime. Do not guess GPU availability;
follow the existing SSH workflow. Record failure/negative results as carefully
as positive results. Reuse verified completed artifacts, never rewrite history.

Before any promotion, produce a paired result table on matching populations,
uncertainty/support report, input/model hashes and a retain/promote recommendation.
If criteria fail, retain the incumbent; do not expand the sweep until it wins.

Release checks:

1. Correct abstract/disclosure/claims register.
2. Strict local tests pass; targeted tests first, broader suite after integrated
   source changes. Distinguish skipped dependencies from passing coverage.
3. Cold-clone ordinary validator and strict validator pass on intended runtime.
4. Both runs produce exactly 50,000 valid unique library sequences and 100 valid
   unique members passing current novelty requirements; FASTA bytes match.
5. Strict manifests identify both generators and every required scoring component.
6. If preserving incumbent, compare against frozen incumbent FASTA hashes. If
   bytes differ, investigate instead of calling it equivalent.
7. All required data/model artifacts are available with documented provenance
   and verified distribution terms; external downloads/revisions are declared.
8. Official upload/access actions remain separate from preparing the package.

## 13. Suggested implementation sequence and handoff prompt

| Unit | Scope | Dependencies | Compute |
|---|---|---|---|
| 0 | Baseline/worklog | None | Local CPU |
| 1a | Config, inventory, hashes | 0 | Local fixtures |
| 1b | Supported parsers and status report | 1a | Local fixtures |
| 2 | Remote collection/status | 1b | Remote CPU, no inference |
| 3 | Claims correction | 0; refine after 2 | Documentation |
| 4a | Strict generation/scorer behavior | 0 | Mocked local tests |
| 4b | Manifest and validator | 4a | Local tests, remote release later |
| 5 | Full top-100 report | 2; verified artifacts | Primarily CPU; reuse scores |
| 6 | Official/external evidence protocol | 2 | Research; bounded execution later |
| 7 | Choose one intervention | 2, 5, relevant 6 evidence | Analysis |
| 8 | Execute and close out | 7 and engineering checks | Remote, bounded |

Paste this to the implementing model:

> Read `docs/PLAN_COMPETITION_READINESS.md` and applicable repository instructions.
> Implement the next incomplete local task only, preserving existing user edits,
> deployed weights, data and successful generation bytes. Start with Task 0, then
> Task 1a. Inspect existing producers before adding result parsers. Use synthetic
> fixtures and run targeted tests. Update `docs/COMPETITION_WORKLOG.md` with actual
> outcomes. Do not invent remote results, run training, unlock tests, promote
> checkpoints or upload a submission. If remote evidence is absent, identify the
> exact required artifacts and continue independent engineering/documentation
> tasks. End with files changed, verification result and next task.
