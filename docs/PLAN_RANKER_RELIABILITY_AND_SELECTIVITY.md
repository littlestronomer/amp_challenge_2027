# Ranker reliability fixes and frozen-pool selectivity experiment

Date: 2026-09-21. Planning baseline: commit `de53b2e`.
Audience: a smaller coding model implementing one bounded task at a time.
Status: implemented in commit `a444a5d`; acceptance tests are prepared but not
run locally, and the new GPU experiment has not run.

## 1. Decision and scope

Prioritize trustworthy reporting and an exploratory activity–hemolysis selection
comparison. Retain the current generator, deployed heads and default selector.
The question is whether the existing candidate pool contains alternatives with
lower predicted hemolysis risk at a tolerable cost in existing activity signals.
This experiment cannot demonstrate improved biological selectivity by itself.

Deliver three separately reviewable increments:

1. Correct validation, evidence inventory and provenance reporting defects.
2. Implement a hash-bound hemolysis cache and a two-policy offline comparison.
3. Produce a decision report and a separate predictor-data improvement backlog.

Do not add generator training, Jev/Laya integration, RL, backbone upgrades,
automatic checkpoint promotion or a default safety weight to this work.

## 2. Established evidence and limits

These observations were supplied by the user from SSH. The local implementation
machine does not contain all remote experiment artifacts. Distinguish reported
observations from files personally verified by the new tools.

| Observation | Result / interpretation |
|---|---|
| Strict release | 50,000 unique valid library sequences; 100 valid selected members; no exact reference overlap |
| Repeat run | Library, top and score CSV hashes match across two runs on the same machine/runtime |
| Top novelty | Ten candidates have maximum local Levenshtein ratio exactly 0.8; retain the existing inclusive threshold |
| Top diversity | No top pair has Levenshtein ratio >=0.8 |
| Historical/current top | Current top and historical hybrid seed-42 top have identical SHA-256 |
| Normalization comparison | Hybrid retains all 100 members at every seed; candidate retains 99/98/100; no compelling normalization change |
| Incumbent predicted hemolysis | Mean scores 0.7462/0.7686/0.7142 for generation seeds 42/43/44 |
| Production inference probes | All three heads agree with eight reconstruction probes each; this is limited inference parity |
| Reconstructed hemolysis | AUROC 0.8460; Brier 0.1596 versus prevalence baseline 0.2484; not independent validation |
| Reconstructed overlap | Hemolysis: 665/951 validation rows have training neighbor >0.8; activity: 193/403 exact overlaps; panel: 1667/2213 neighbors >0.8 |
| Chemistry/data mismatch | 652/4752 hemolysis records contain nonstandard characters; their meaning needs inspection |
| Family benchmark | MLP AUROCs: activity 0.733, panel 0.725, hemolysis 0.850; all paired MLP-minus-linear intervals include zero |
| External evidence | No independent evaluation of the generated top-100, synthesis assessment or official score supplied |

Important interpretation rules:

- Breadth/MDR are fractions of genus probabilities exceeding 0.5, not confidence
  probabilities. A score of 1.0 does not mean certain biological activity.
- Positive hemolysis label means risky under the candidate label definition,
  not percent red-cell lysis. The exact assay threshold/units must come from
  label-building provenance; do not infer them from a generic classifier config.
- The recorded `val_auroc` in a head config is metadata, not a new validation.
- The old artifact audit's `no_matching_member` can be superseded by later
  frozen-run tensor matches. Neither establishes original data/split provenance.
- The family benchmark evaluates its own models. Do not assign its performance
  to the deployed head without an explicit artifact match.
- Reconstructed/random-split results and already-inspected family tests cannot
  become untouched confirmation sets for this new experiment.

Known fingerprints to preserve and cross-check on SSH:

```text
incumbent library: 74f73e71e6d867083aede2cd773ec4761e0f2533edf7485416447d8023c4b84b
incumbent top: eea9528d64e3d125f3639772239ccfc39fed25fb3746100670f0e27b58c08221
incumbent top_scores: 96264c6c58daed03950e458abf89f751d26dd00434a6350201c6c2cc32523afd
hemolysis head: ec48d490001b54f90d0e2f077b9ff3fa274234b003fdade932648dda9ce0ba7b
hemolysis explicit config: b5c923689ce9e1c0a5722ced68e029a33543c83e748047510693f7b73e69a197
hemolysis backbone revision: 6fbf070e65b0b7291e7bbcd451118c216cff79d8
reference: cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9
```

## 3. Implementer rules and entry point

Before edits: read applicable AGENTS.md files, inspect `git status --short`, and
read this plan plus the files listed in the current task. Local `pyproject.toml`
and `uv.lock` have pre-existing Jev edits: preserve and exclude from incidental
commits. Reinspect the actual checkout because it may have advanced.

Work one task at a time. Each task ends with: files changed, targeted checks,
observed result, unresolved assumptions and next task. Do not claim execution
of an SSH-only command from a local fixture. Keep a checklist in a proposed
`docs/SELECTIVITY_WORKLOG.md`.

Immutable inputs: checkpoints, reference FASTA, original label CSVs, existing
experiment directories and their manifests. New experiments use new directories.
Never edit an old run.json to bypass resume checks. Local work uses synthetic
fixtures and the existing environment. Real inference runs on SSH, GPU 1:
`CUDA_VISIBLE_DEVICES=1`; within the process use `--device cuda`, not cuda:1.
No package installation or environment upgrade is needed for this plan.

## 4. Task A1 — Fix ordinary validator repeatability

Read/edit: `scripts/verify_submission.py`. Add focused tests in proposed
`tests/test_submission_repeatability.py`.

Confirmed defect: ordinary mode assigns both runs the same output paths, runs
generation again, and then compares two reads of those same paths. A changed
second output can pass. The user's independent strict-run hash comparison is
unaffected; this defect concerns the ordinary validator path.

Implementation:

1. Save first-run library and top bytes before invoking generation a second time.
2. Compare saved bytes against second-run bytes. Preserve existing validity and
   novelty checks and strict mode's two separate directories.
3. Keep strict mode's score-CSV byte comparison. Do not compare whole manifests:
   output-directory fields legitimately differ.

Acceptance tests mock cloning/environment setup and generation: identical runs
pass; library-only changes fail; top-only changes fail; strict score-only changes
fail; missing second outputs fail. No real cloning, downloads or inference.

## 5. Task A2 — Fix evidence inventory and report states

Read/edit:

- `src/amp_challenge_2027/evidence_inventory.py`
- `scripts/collect_competition_evidence.py`
- `experiments/competition_evidence_v1.json`
- `tests/test_competition_readiness.py`

Prefer a new `experiments/competition_evidence_v2.json` for the expanded source
set; preserve old configs for historical reconstruction.

Required changes:

1. Inventory the actual completed normalization source:
   `sweep_results/selection-normalization-v1`. Include its run.json, results,
   paired deltas, component distributions/effective weights and complete marker.
   Discover required marker fields from the actual producer. Preserve any
   separate legacy selection-diagnosis source as a separately named entry.
2. Add explicit reconstruction, predictor-artifact, inference-probe and strict
   repeat source entries. Different audit versions must remain separate records;
   directory names alone must not decide which historical claim supersedes another.
3. Derive STATUS.md from observed evidence. Do not keep the unconditional
   'run strict generation' message when its outputs verify. Report single-run
   integrity separately from verified two-run byte equality.
4. Root existence should be called present, not scientific verification. Define
   each status in the output. A null marker remains provenance-unverified.
5. Fix hash failure semantics: a configured expected hash mismatch is invalid,
   even if a producer marker agrees with the current modified file. An invalid
   marker must not be marked verified merely because its filename matches.
6. Include missing/invalid marker reasons in diagnostics, not just file failures.
   Missing required configured markers fail the required-source gate; a null
   marker deliberately configured as absent is not the same case.
7. Validate config fields/path types before creating output. Handle duplicate
   IDs, traversal, escaping symlinks, malformed hash maps and unknown adapters
   with explicit diagnostics. Unknown schemas may be inventoried without parsing.
8. For a two-run comparison, verify both output manifests/hashes first; compare
   actual library/top/score bytes or SHA-256 values, not just marker existence.
   Verify compatible effective recipes after ignoring only documented fields
   such as output path. Preserve runtime differences as comparison limitations.

Tests: old generic configs still load; actual normalization root appears;
stale snapshot text disappears after a verified strict result; repeatability is
not inferred from one run; tampered expected hash/marker fails; partial optional
sources remain explicit; source bytes never change; output remains deterministic.

## 6. Task A3 — Correct provenance semantics and resume diagnostics

Read/edit:

- `src/amp_challenge_2027/release_manifest.py`
- `src/amp_challenge_2027/score.py` (inspect; avoid changing inference)
- `scripts/verify_submission.py` and inventory manifest reader
- `scripts/experiment_utils.py`

Manifest changes:

1. Conformity is a property-density scorer with no neural backbone. Mark revision
   applicability explicitly; exclude it from neural revision completeness checks.
2. Separate 'resolved immutable revision recorded' from 'explicitly pinned at
   load time'. A runtime `_commit_hash` proves resolution, not an explicit loader
   pin. Introduce clearly named fields for both. Do not claim all loads are
   explicitly pinned unless the actual loader calls enforce that.
3. Hash the actual config source used by each scorer, preserving effective
   metadata and whether it came from an explicit file or the hash-bound registry.
   Record explicit head/config hashes even when those files are untracked.
4. Record the reference-embedding cache's content hash and input identity when
   used; a cache filename alone is insufficient. If a resource identity cannot
   be obtained, record unavailable with a reason; do not fabricate it.
5. Record CUDA visibility and logical device alongside the existing runtime.
   Do not pretend the logical index alone identifies a physical GPU.
6. Use a versioned schema if changing existing field meanings. Readers accept
   old and new versions; old manifests remain byte-for-byte untouched.

Resume diagnostics:

- Keep exact recipe matching in `prepare_run`; do not remove code/commit hashes.
- When mismatched, show deterministic differing key paths and categories
  (code/runtime, data, model, protocol). Bound output length and do not dump
  arbitrary file contents. Clarify that --list validates sources, not resumability.
- Give the existing run path and a new-output suggestion. A read-only summary
  reader must work without resuming the old experiment.

Tests: property-only scorer is not reported as missing a backbone; a resolved but
unpinned neural model is represented accurately; config/cache mutations change
their recorded identity; old manifest reading works; mismatch messages identify
changed commit versus changed input while continuing to refuse unsafe resume.

## 7. Task A4 — Update the evidence decision documents

Update `docs/COMPETITION_STATUS.md`, `docs/NEXT_EXPERIMENT_DECISION.md`,
`docs/COMPETITION_WORKLOG.md`, and `docs/CLAIMS_REGISTER.md` using Section 2.
Do not silently promote user-pasted reports to independently verified local files.

Record the decision: retain incumbent; normalization intervention not supported;
authorize an exploratory fixed-pool risk trade-off study, not a validated upgrade.
Explain explicitly why this limited sensitivity experiment is compatible with
weak predictor-generalization evidence: it describes the selector's behavior and
candidate availability; it cannot justify promotion on its own.

Separate statuses: engineering, reconstructed discrimination, family benchmark,
deployed-head provenance, generated-candidate evidence, official standing.
No safety claim, experimental success probability or external-validation claim.

## 8. Task B1 — Freeze one small experiment protocol

New file: `experiments/selectivity_tradeoff_v1.json`.
New runbook: `docs/RUNBOOK_SELECTIVITY_TRADEOFF.md`.

Read first: `scripts/selection_cache.py`, `scripts/compare_top100.py`,
`scripts/analyze_selection.py`, `src/amp_challenge_2027/selection_audit.py`,
`src/amp_challenge_2027/score.py`, `experiments/reward_reconstruction_v1.json`.

Protocol contract:

- Source: `sweep_results/epoch58-top100-v1`; use existing six-cell loader to
  validate its complete source contract, but evaluate only hybrid seeds 42/43/44.
- Freeze all library membership/order, reference, activity/panel/precision/
  conformity predictions, scorer identities and backbone revisions.
- Two policies only: C0 (incumbent replay) and R1 (one risk penalty).
- Ranking seed remains 42, top size 100, shortlist 2000, same plausibility,
  novelty <=0.8 and farthest-point selection. No manual replacement of the ten
  boundary candidates and no full-library regeneration.
- Use production `HemoScorer.p_risky`, with the matched head/config/temperature
  and explicit immutable backbone revision from the frozen source.
- Set exploratory penalty strength once: lambda=0.25. This is an engineering
  sensitivity setting, not a clinically calibrated trade-off or an optimized value.
  Do not sweep or alter it after examining results.

Exact score definition (arrays aligned to each frozen FASTA):

```text
S0 = cell['combined']   # original five-component score; preserve arithmetic
mu, sd = mean/std of p_risky over ALL 50,000 hybrid seed-42 sequences
d = sd when sd > 1e-8, otherwise 1.0
S1 = float32(float64(S0) - 0.25 * (float64(p_risky) - mu) / d)
```

Freeze mu/d from the anchor and apply unchanged to seeds 43/44. Baseline scales
stay exactly as in each original C0 cell; do not mix this with normalization P1.
Implement lambda=0 as an explicit copy of S0 to guarantee exact baseline parity.
Stop with an uninformative-risk diagnostic if anchor risk is effectively constant.

Predeclare exploratory comparison criteria, chosen before running R1:

- Both policies pass all validity/membership/novelty requirements.
- For every seed, risk mean decreases by at least 0.05 absolute score units and
  risk p75 does not increase.
- Per seed, activity mean, panel mean probability and each genus mean probability
  decrease by no more than 0.03 absolute units; breadth/MDR means decrease by no
  more than 0.02; mean pairwise distance decreases by no more than 0.02.
- Report every metric even when criteria fail. These are exploratory surrogate
  tolerances, not evidence of biological noninferiority or safety.
- Passing means 'candidate for independent evaluation'. Never auto-promote.
- Failing means retain incumbent and record the negative result; no expanding
  the weight search until a favorable result appears.

## 9. Task B2 — Implement a resumable, sequence-bound risk cache

Proposed files:

- `scripts/cache_selectivity_risk.py` (CLI and inference orchestration)
- `src/amp_challenge_2027/selectivity.py` (pure score/cache validation helpers)
- `tests/test_selectivity_risk_cache.py`

Do not import scripts from the installed package. Script-to-script use of
`selection_cache`, `compare_top100`, `experiment_utils` follows current layout.
Keep model loading at the CLI boundary so tests can use an injected fake scorer.

CLI design:

```text
--source PATH --protocol PATH --out PATH --reference PATH
--device cuda --batch-size 64 --list
--seeds 42 [43 44] --max-new-sequences N (optional execution budget)
```

Algorithm:

1. --list loads and validates source manifests/cache hashes, source head metadata,
   reference and protocol; prints counts and expected work; creates no files and
   loads no model. Missing SSH inputs are errors, not generated substitutes.
2. Build a deterministic ordered union of selected hybrid libraries: ascending
   generation seed, then original FASTA order, first occurrence wins. Build
   per-cell index maps. Expected maximum across all three seeds is 150,000
   sequence occurrences; report actual unique count before inference.
3. Bind cache to exact sequence order/hash, per-cell FASTA hashes, head and config
   hashes, effective temperature, label meaning, tokenizer/backbone revision,
   preprocessing, protocol, runtime and code identity. Verify the deployed
   artifact matches the source before loading; require scorer load success.
4. Before pool scoring, replay risk predictions for all cached source top-100
   sequences in requested cells. Require agreement within rtol=atol=1e-5 with
   original risk rows. Stop on mismatch, don't reinterpret new scores as old.
5. Infer each unique sequence once, in bounded fixed chunks. Store completed
   chunk arrays plus sequence-index intervals and hashes atomically. Validate
   shape, finite values and [0,1] bounds. Missing data must never become zero.
6. On resume verify every completed chunk and full identity before skipping it.
   A corrupt marked chunk is an error; an unmarked partial write is recomputed.
   Do not mark a partially processed union complete. A sequence budget stop
   writes an explicit incomplete status and no full-success marker.
7. Support completing seed42 first and extending coverage to seeds43/44 with
   the SAME frozen union identity/protocol. Precompute the union for all three
   cells at initialization; --seeds controls coverage, not semantic cache identity.
   Each cell can be marked complete once all its mapped risk values verify.
8. Write per-cell aligned float32 risk arrays and index-map hashes, manifest,
   coverage table, runtime/throughput report, and completion markers. Markers
   cover all files consumed downstream, not merely report summaries.
9. Reverify source identity before finalizing. Never write inside source roots.

Tests: alignment/reordered FASTA rejection; cross-cell dedup; head/config/revision
change refusal; parity mismatch stop; corrupt chunk refusal; interruption/resume;
partial coverage is never complete; missing/NaN/out-of-range risks rejected;
seed42-to-three-seed extension; no downloads/model loads in --list; source hashes
unchanged. Use small synthetic libraries and deterministic fake inference.

## 10. Task B3 — Implement the two-policy comparison

Proposed files:

- `scripts/compare_selectivity.py`
- extend `src/amp_challenge_2027/selectivity.py` with pure policy/report helpers
- `tests/test_selectivity_comparison.py`

CLI: `--source`, `--risk-cache`, `--protocol`, `--reference`, `--out`, `--list`.
CPU-only: it must never load HemoScorer, another neural model or a generator.

Execution order:

1. Validate all source and risk-cache dependencies; require complete aligned
   risks for all three hybrid cells. Reject missing seeds rather than silently
   reducing the experiment. Expose partial cache status in preflight diagnostics.
2. Replay C0 on all three cells using `selection_stages`, existing combined
   scores and FASTA writer. Require byte-identical original top files for ALL
   C0 cells before computing any R1 selection.
3. Fit/hash anchor risk normalization once, then compute R1 and pass its score
   array through the same selector. Enforce full output counts, uniqueness,
   membership and exact reference novelty on every result.
4. Preserve ranking order and export each candidate's source library index,
   original and risk-adjusted scores, raw five components, risk, each genus
   probability, maximum reference similarity and relevant sequence properties.
5. Write C0/R1 per-cell files under distinct directories with atomic markers.
   Resume only matching identities. Reject tampering rather than overwrite it.

Do not reuse `selection_audit.paired_deltas` unchanged: its policy/case names
are hardcoded for P0/P1 and hybrid/p3_s1. Add a dedicated explicit-pair helper
for C0/R1 rather than weakening the existing producer's contract.

Required outputs:

- `run.json`, `normalization_anchor.json`, `inventory.json`, `complete.json`
- `C0/hybrid/seedN/` and `R1/hybrid/seedN/`: top.fasta, top_scores.csv,
  summary.json, stages.npz, complete.json
- `results.csv`, `paired_deltas.csv`, `seed_summary.csv`, `criteria.json`
- `REPORT.md` with provenance, trade-offs, coverage, missing independent evidence,
  and explicit exploratory/retain-or-further-evaluate decision

Metrics: risk mean/median/p75/max, activity mean, all genus probability means,
panel mean, threshold breadth/MDR, conformity/precision, sequence overlap,
pairwise diversity, novelty max, membership/validity and full score coverage.
Report changed membership separately from order-only changes. Keep score names
explicit; do not call fractions above 0.5 confidence or risk scores percent lysis.

Pair seeds exactly. Summarize three generation seeds descriptively with per-seed
values and range/mean; they are not three independently trained models or new
held-out test sets. No peptide-level bootstrap masquerading as independent model
replicates. Unknown external/assay/synthesis outcomes remain unavailable.

Tests: C0 byte replay; any C0 mismatch prevents all R1 work; lambda=0 parity;
correct penalty sign; fixed anchor across seeds; order-safe joins; complete risk
coverage; changed membership has risk values; wrong source/reference/revision
refused; ties deterministic; underfilled top fails without completion marker;
paired criteria apply to each seed; pure score improvement cannot auto-promote.

## 11. Task B4 — SSH runbook and stop rules

The commands below are PROPOSED interfaces and must not be run until implemented.
Use the existing feature branch containing implementation; never blindly pull
main from an older runbook. Record the delivered commit in the new runbook.

```bash
# CPU preflight; no new inference
uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --out sweep_results/selectivity-risk-v1 --list

# Measure a bounded first batch on GPU 1; incomplete cache is expected
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --out sweep_results/selectivity-risk-v1 --device cuda --batch-size 64 \
  --seeds 42 --max-new-sequences 1024

# Resume anchor coverage using the same semantic cache identity
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --out sweep_results/selectivity-risk-v1 --device cuda --batch-size 64 --seeds 42

# Complete the predeclared three-seed coverage; no policy tuning in between
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/cache_selectivity_risk.py \
  --source sweep_results/epoch58-top100-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --out sweep_results/selectivity-risk-v1 --device cuda --batch-size 64 --seeds 42 43 44

# CPU comparison, after full cache coverage
uv run --no-sync python scripts/compare_selectivity.py \
  --source sweep_results/epoch58-top100-v1 \
  --risk-cache sweep_results/selectivity-risk-v1 \
  --protocol experiments/selectivity_tradeoff_v1.json \
  --out sweep_results/selectivity-comparison-v1
```

Measure throughput after warm-up before estimating total runtime. Report cached
versus newly inferred counts. OOM, identity mismatch, score mismatch, invalid
input or insufficient eligible candidates stops the affected run; do not switch
models, relax novelty, drop components or silently change precision/batch size.
If runtime settings must change, use a new explicitly recorded execution identity.
One writer per output directory. No automatic SSH access from the local agent.

## 12. Task C — Predictor-data improvement specification, not retraining

Write proposed `docs/PLAN_PREDICTOR_DATA_REPAIR.md` after the engineering tasks.
Reuse existing observation, reconciliation, molar-label and family-benchmark
tools. Do not implement another competing parser or split system without a gap.

Required future design:

1. Preserve raw observations, source identifiers, sequence chemistry, assay type,
   units, inequality operators, tested concentration and explicit label provenance.
2. Separate supported canonical unmodified linear sequences from known modified
   chemistry, unknown chemistry and malformed records. No automatic stripping,
   canonicalization or conversion of nonstandard characters into another molecule.
3. Review discordant/censored assays using explicit rules. Missing observations
   are unknown, not negative; do not use a language model as ground truth.
4. Family grouping and deduplication occur before split assignment. Keep related
   records in one split and audit train/validation/calibration/test overlap.
5. Reserve calibration independently from fitting/selection. Existing inspected
   test results remain historical descriptive evidence. Any future confirmatory
   evaluation needs genuinely new appropriate data or a declared nested design.
6. Start with linear and current-size MLP baselines on identical examples and
   fixed protocols. Capture dataset/split/model hashes, uncertainty/support and
   calibration metrics. Do not presume a larger model will help.
7. Promotion requires matched-population evidence and an appropriate independent
   assessment of the proposed selector, not its optimization scores alone.

A separate training request/protocol follows that design. This plan does not
authorize unattended training sweeps, new test-set tuning, or checkpoint replacement.

## 13. Verification and delivery sequence

Recommended implementation units: A1 -> A2 -> A3 -> A4 -> B1 -> B2 -> B3 -> B4 -> C.
Complete local code and tests before any real SSH inference. Keep commits small
and omit pre-existing dependency edits. Run relevant existing tests when touching
shared code, especially:

```bash
.venv/bin/python -m pytest -q tests/test_competition_readiness.py \
  tests/test_pipeline.py tests/test_selection_audit.py tests/test_top100_comparison.py
```

Also run the new targeted tests, Ruff on changed files and `git diff --check`.
Do not fix unrelated existing lint issues as part of this task. Default generator
weights/behavior must remain unchanged; mocked checks protect paths locally and
known SSH artifact hashes document the existing successful baseline.

Definition of completion:

- Ordinary validator detects changed second outputs.
- Evidence reports identify completed existing experiments without stale claims.
- Manifest distinguishes non-neural, resolved and explicitly pinned revisions.
- Cache preserves exact source/model/sequence identity and resumes safely.
- All C0 replays match; R1 results are paired, complete and honestly interpreted.
- Default submission remains the incumbent unless a separate evidence-based
  promotion decision is made.
- Remote results are either actually reviewed or explicitly marked pending.

## 14. Handoff prompt for the implementing model

> Read docs/PLAN_RANKER_RELIABILITY_AND_SELECTIVITY.md and applicable repository
> instructions. Inspect the working tree and preserve pre-existing Jev dependency
> edits. Implement Task A1 first with a regression test that changes second-run
> output. Continue through the local tasks in dependency order, using the exact
> contracts and acceptance criteria in the plan. Keep existing checkpoints,
> reference data, historical results and deployed generation/selection defaults
> unchanged. Build the new experiment as separate offline CLIs, tested on synthetic
> data, with fail-closed provenance and resumable caches. Do not invent remote
> results, tune on the inspected test, run SSH jobs automatically, or promote a
> new selector because it improves its own predictor scores. Update the worklog
> after each task and deliver exact GPU-1 SSH commands once the CLIs exist.
