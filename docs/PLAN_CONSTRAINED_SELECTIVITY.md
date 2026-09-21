# Constrained selectivity: implementation contract and task queue

Date: 2026-09-22. Status: proposed; commands and modules below do not yet exist.
Audience: a coding model implementing small, independently reviewable tasks.

## 1. Objective and scope

Find a diverse top 100 with lower predicted hemolysis while preserving declared
activity and panel coverage, using the frozen incumbent library first. Establish
whether candidate supply, selection, or predictor reliability limits progress.
Do not describe a computational result as hemolysis-free or experimentally safe.

Three workstreams, in priority order:

1. **A: frozen-pool feasibility and constrained selection.** CPU work using
   existing score caches; no generator training or new neural inference required.
2. **B: assay-aware outcome data and predictor repair.** Necessary for stronger
   biological claims; computational pipeline can be built with synthetic data.
3. **C: selective conditional generation.** Activate real training only after
   A identifies a supply limitation and B provides supported training targets.

Implement A first. Its deliverable is useful even if no acceptable top 100 exists.
Implement B/C software separately; do not automatically train or deploy them.
The related [ESM design specification](PLAN_ESM_CONDITIONAL_PEPTIDE_DESIGN.md)
and [task queue](TASKS_ESM_CONDITIONAL_PEPTIDE_DESIGN.md) specify B/C foundations.
This document adds the frozen-pool optimization, dose-response extension, and
explicit decision linking the workstreams. Do not create duplicate data frameworks.

## 2. Evidence and protected state

The user supplied native-baseline comparisons on SSH: all libraries contain
50,000 sequences and use the same 650M ESM evaluator and reference. The incumbent
has FBD 0.221416 and MMD 0.356674, versus AMP-Diffusion 1.651497/5.979657 and
HydrAMP 9.086869/48.19417. These are reported local component metrics, not an
official aggregate score or evidence of activity.

Top-100 deployed-score means:

| Method | Activity | Hemolysis risk |
|---|---:|---:|
| Incumbent | 0.8424 | 0.7462 |
| AMP-Diffusion | 0.7487 | 0.6790 |
| HydrAMP | 0.5636 | 0.6509 |

These are biased toward models used in selection. Do not convert a classifier
output to MIC, HC50, percent lysis, or clinical probability. Our prior R1 penalty
experiment failed coverage/diversity criteria; preserve that outcome unchanged.

Inputs available on SSH, not necessarily in the development checkout:

- `sweep_results/epoch58-top100-v1`: frozen scores and libraries, hybrid seeds
  42/43/44 and p3_s1 controls. A uses hybrid cells only after source validation.
- `sweep_results/selectivity-risk-v1`: complete aligned risk cache.
- `experiments/selectivity_tradeoff_v1.json`: original cache's generating protocol.
- `generate/competition-readiness-v1`: incumbent submission artifacts.
- `sweep_results/native-baselines-v1`: baseline snapshots and comparisons.

Never modify existing manifests, checkpoints, reference data, old experiment
directories, or production `generate` defaults. Preserve unrelated changes in
`pyproject.toml` and `uv.lock`. New roots: `sweep_results/constrained-selectivity-v1/`
with separate sibling stage directories. Native baselines remain external
comparators, not sources of peptides for the incumbent shortlist.

Selecting different members of the identical library leaves its library metrics
unchanged. Once generation or library membership changes, rerun the full library
evaluation; never carry forward the incumbent's scores.

## 3. Files and reuse

Read these before implementing:

| Existing file | Purpose |
|---|---|
| `scripts/selection_cache.py` | Validate source, alignment, reference, model provenance |
| `scripts/cache_selectivity_risk.py` | Union mapping and chunk/completion contracts |
| `scripts/compare_selectivity.py` | Risk import checks, C0 replay, candidate summaries |
| `scripts/compare_top100.py` | Frozen model metadata and score schema |
| `src/amp_challenge_2027/selection_audit.py` | Existing selector stage behavior |
| `src/amp_challenge_2027/props.py` | Custom plausibility diagnostics |
| `scripts/experiment_utils.py` | Recipes, hashes, completion markers |
| `scripts/build_hemolysis_labels.py` | Existing label limitations; do not reuse as HC50 truth |

New package: `src/amp_challenge_2027/selectivity_research/` with `contracts.py`,
`pool.py`, `eligibility.py`, `constraints.py`, `solver.py`, `metrics.py`,
`reporting.py`. Thin `scripts/selectivity_*.py` CLIs may import script helpers;
package modules must not import `scripts` modules. Use SciPy's existing optional
ML dependency for MILP; do not introduce a commercial solver or change base deps.

## 4. A data contract: frozen pool bundle

`pool.csv` has one row per library sequence, in original order:

```text
sequence, sequence_sha256, library_index, source_seed
activity, hemolysis_risk, conformity, precision_proxy
panel:<genus> for every genus in the source's declared order
incumbent_member, incumbent_rank (null for nonmembers)
```

All scores finite and in their documented ranges. Sequence identity is exact;
never join scores by rank or silently sort arrays. Avoid rounding until display.
Each seed is a distinct bundle and optimization problem: never pool the three
libraries to produce a shortlist that is not a subset of one submission library.

Files: `pool.csv`, copied `incumbent_top.fasta`, `source_identity.json`,
`run.json`, `complete.json`. Source identity includes source/cache markers,
original protocol hash, reference, model/config hashes, resolved backbones,
ordered sequence digest, producer code/runtime, and consumer code/runtime.

### Historical cache import

The existing `_read_risk_cache` binds to current code identity and original
protocol. The new selector must NOT pass its new optimization protocol as the
old cache's generating protocol. Accept separate `--cache-protocol` and
`--protocol` arguments.

Implement a read-only historical adapter that checks every required file and
marker, union/index map, source/model/reference/revision identity and original
protocol hash. Record the old code identity without pretending it equals the
new consumer. Validate supported producer schema explicitly; unknown schemas
fail. A code-version change alone is not proof of corruption. Do not disable
the old reader's checks, overwrite its recipe, or regenerate 145k predictions
merely to satisfy a consumer-code comparison. Missing required hashes fail.

If extracting a shared pure verifier, regression-test old reader behavior.
Record all input hashes before and after every stage. Never write below an input.

## 5. Eligibility across the full library

Evaluate all 50,000 rows for each seed; the old 2,000-candidate shortlist is not
an eligibility boundary. Existing source scores can be reused for all rows.

Rules: valid canonical sequence/length, no exact reference overlap, existing
custom plausibility policy, and maximum exact Levenshtein ratio <=0.8 against
the reference. Report plausibility as a custom heuristic, never assay safety.
Reference similarity is the package's `Levenshtein.ratio`, not alignment identity.

Performance: chunk reference comparisons, stop once a rejection is proven,
and allow exact upper-bound pruning only when exhaustively checked on small
fixtures. Never substitute approximate nearest neighbors for the final gate.
An incomplete eligibility scan is `incomplete`, not an all-pool feasibility result.

Output `eligibility.csv` with sequence ID, eligible flag, reason codes, exact
maximum similarity if fully computed, otherwise a clearly named rejecting
similarity lower bound. Cache completed chunks by sequence/reference/policy
identity. Invalid/missing scores exclude with an error, not a substituted zero.

## 6. Mathematical selection problem

For each eligible candidate i, binary x_i indicates membership. Let a_i be
activity, r_i risk, p_ig panel probability for genus g, and K=100.
Calculate all incumbent comparison values from unrounded arrays for that seed.

Primary objective: minimize `sum(r_i*x_i)/K`.

Hard constraints in the default research protocol:

1. `sum(x_i) = 100`.
2. Mean activity >= incumbent mean minus 0.03.
3. Mean panel probability for EACH genus >= its incumbent mean minus 0.03.
4. Mean breadth >= incumbent breadth minus 0.02; breadth per peptide is the
   fraction of genus probabilities strictly >0.5.
5. Mean MDR-genus breadth >= its incumbent value minus 0.02. This is a genus
   proxy and must never be relabelled strain-specific MDR efficacy.
6. For any selected pair with ratio >=0.8, `x_i + x_j <= 1`.
7. Tail guard: count of candidates with risk >0.8 <= that count in incumbent.

Clamp lower bounds to zero. Keep the > versus >= conventions explicit. Include
all genera, not only those that look favorable. Reject empty MDR mappings.

Primary acceptance additionally requires mean pairwise sequence distance >=
incumbent mean distance minus 0.02. This nonlinear statistic is evaluated on the
final set; it is NOT enforced by the initial MILP. If it fails, report
`feasible_linear_constraints_failed_portfolio_acceptance`, not full success.
Do not silently run repeated exclusion searches to find a passing answer.

Novelty and pairwise rejection are different: reference equality at 0.8 passes;
selected-pair equality at 0.8 conflicts. Connected components are diagnostic
only: one-per-transitive-component would be stricter than this pairwise rule.

### Tail profiles and scope of conclusions

One main policy: `min_mean_risk` under the above constraints. Four additional
predeclared diagnostic policies apply per-candidate risk ceilings 0.25, 0.50,
0.75, 1.00 with the same constraints. They characterize sensitivity, not clinical
safety categories. Report every profile, including infeasible/unfinished ones.
Do not choose the most flattering profile after viewing results and call it
confirmatory. The 1.00 profile can alias the main solve to avoid duplicate work.

Also report mean, median, p75, p95, maximum risk; activity p05 and median;
each genus mean; breadth; pairwise distance; reference similarity distribution;
overlap with incumbent; constraint margins. Do not multiply activity and
one-minus-risk probabilities into a claimed biological joint probability.

## 7. Solver and feasibility semantics

Use `scipy.optimize.milp` / HiGHS with sparse constraints. Initial defaults:
300-second budget per profile/seed across all cut rounds, relative MIP gap 0.001,
and at most 20 cut rounds. Record actual settings and solver/runtime versions.
These are bounded engineering defaults, not optimized scientific parameters.

Avoid allocating a 50,000-by-50,000 distance matrix. Start with the linear
problem, inspect the <=4,950 selected pairs, add all violated pair conflicts,
and solve again with the remaining shared budget. Every cut is valid globally.

- An optimal solution with no pair violations proves optimality for the full
  pair-constrained problem within recorded numerical/MIP tolerances.
- An infeasible relaxation proves infeasibility of the stricter problem for
  that eligible pool and exact profile. State solver numerical limitations.
- A time limit is `unknown` unless a returned integer incumbent passes every
  constraint; that incumbent is `feasible_unproven_optimal`.
- A pair-violating or fractional solution is not exportable as a top 100.
- Posthoc mean-distance failure does not prove the full problem infeasible.
- Never remove constraints, shrink K, or extend the budget automatically.

Cheap necessary-condition checks can precede MILP: count eligible candidates,
sum of top K activity values, and separate per-genus maxima. These can prove
certain impossibilities but passing them does not establish joint feasibility.
Do not Pareto-prune individuals: apparently dominated candidates may be needed
because of pairwise conflicts and portfolio coverage.

Membership determinism: sort input candidates by sequence bytes before building
the MILP. Do not promise unique membership across solver/platform versions when
multiple optima exist. Save the accepted solution and verify it on resume;
never resolve a completed run. Stable output rank is activity descending, risk
ascending, sequence ascending. Record that rank is cosmetic for random sampling.

## 8. Acceptance and honest conclusions

Default main-policy improvement gate: risk mean decrease >=0.05 on every
historical seed and no increase in risk p75, alongside all section 6 constraints
and mean-distance tolerance. These reuse the prior descriptive tolerances;
they do not establish efficacy. All existing seeds were inspected: label results
`retrospective_exploratory`, even if every gate passes.

Decision categories:

| Finding | Next action |
|---|---|
| Passing feasible portfolio | Candidate for separately frozen evaluation; retain incumbent pending decision |
| Valid solution with tiny risk benefit | Limited selection opportunity under this model/protocol |
| Proven infeasible at a low risk ceiling | Pool lacks a feasible portfolio under that specific ceiling/constraints |
| Timeout, incomplete scan, mean-distance failure | Unresolved; do not call this a generator limitation |
| Poor/mismatched risk-model evidence | Prioritize B before optimizing that score further |

Even a proven computational supply limitation does not establish biological
impossibility. Conditional generation is a hypothesis to address that limitation.
New generation seeds test sampling robustness of a fixed generator, not
independent training replicates. A held-out study/assay set is a different test.

## 9. Artifact contract and command surfaces

Each stage has one output registry shared by producer, tests, and inventory.
Every completion marker hashes `run.json` and all advertised files. Safe relative
paths only; validate files before marking complete. Existing completed outputs
are read-only; changed recipes require new directories. `--list` makes no writes,
loads no model, and describes counts/dependencies. Partial stages have explicit
status and verified resume chunks, never a success marker.

Runnable commands after the implementation is pulled:

```bash
uv run --no-sync python scripts/selectivity_import_pool.py \
  --source sweep_results/epoch58-top100-v1 \
  --risk-cache sweep_results/selectivity-risk-v1 \
  --cache-protocol experiments/selectivity_tradeoff_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/constrained-selectivity-v1/pools

uv run --no-sync python scripts/selectivity_scan_pool.py \
  --pools sweep_results/constrained-selectivity-v1/pools \
  --protocol experiments/constrained_selectivity_v1.json \
  --reference data/antibacterial.fasta \
  --out sweep_results/constrained-selectivity-v1/eligibility

uv run --no-sync python scripts/selectivity_solve.py \
  --pools sweep_results/constrained-selectivity-v1/pools \
  --eligibility sweep_results/constrained-selectivity-v1/eligibility \
  --protocol experiments/constrained_selectivity_v1.json \
  --out sweep_results/constrained-selectivity-v1/solutions

uv run --no-sync python scripts/selectivity_report.py \
  --pools sweep_results/constrained-selectivity-v1/pools \
  --solutions sweep_results/constrained-selectivity-v1/solutions \
  --out sweep_results/constrained-selectivity-v1/report
```

No GPU is needed for A. Report stage consumes persisted solutions, never launches
optimization or neural inference. Output includes `comparison.csv`, `REPORT.md`,
`decision.json`, per-seed/profile `solver_status.json`, `constraint_margins.json`,
and, only for verified feasible solutions, `top.fasta` and `top_scores.csv`.
Acceptance failures may still have a feasible shortlist, labelled accordingly.
Do not emit an underfilled or invalid `top.fasta`.

## 10. Small-model implementation queue: workstream A

Use this prompt for each task:

> Implement task Axx from docs/PLAN_CONSTRAINED_SELECTIVITY.md. Read the listed
> reuse files and preceding contracts. Implement only this task and focused
> synthetic tests. Preserve production defaults and old artifacts. Do not run
> real training, download models, or promote a candidate. Record changed files,
> test commands/results, schema changes, known limitations and the next task in
> docs/CONSTRAINED_SELECTIVITY_IMPLEMENTATION_LOG.md. Missing prerequisites must
> produce explicit errors, never fabricated outputs or production stubs.

### A00 — Contracts and protocol

Create package, schemas/output registry, `experiments/constrained_selectivity_v1.json`
and log. Encode every threshold, objective, output rank, source role, solver budget,
status and decision rule above. Public APIs: `validate_protocol`,
`validate_pool_bundle`, `validate_solution`. Dataclasses plus explicit JSON
validation suffice. No new configuration framework.

Tests: malformed ranges/unknown schema, path escape, missing marker entries,
mutated recipe, optional missing evidence vs invalid required input.

### A01 — Historical source/cache adapter

Create `selectivity_import_pool.py` and `pool.py`; reuse source verification and
implement the historical-cache contract in section 4. Export three hybrid seed
bundles with all rows. Copy C0's top bytes; reproduce summary scores from arrays.
Validate the p3_s1 cells required by the source schema without using them as
candidate supply. Explicitly verify the seed-42 library/top hashes against the
incumbent paths when provided; otherwise say incumbent identity not crosschecked.

Tests: shuffled risk rows, bad index maps, missing seed, changed checkpoint or
reference, partial cache, historical code identity preserved, corruption rejected,
no write on `--list`, no neural inference. Fixtures need not contain 50k rows:
inject K/pool sizes only through a test contract, not silent production fallback.

### A02 — Full-pool eligibility

Implement chunked scan and resume, input hashes, custom heuristic reasons and
exact reference checks. Source-order invariance refers to decisions by sequence;
retain original indices. Reuse chunk results only under identical identities.

Tests: >0.8 rejected, exactly 0.8 accepted, invalid residues/length, no input
mutation, pruning agrees with brute force, partial scan cannot produce complete
feasibility, new protocol invalidates resume.

### A03 — Baseline metrics and constraints

Implement per-seed incumbent recomputation and sparse linear constraint builder.
API: `build_problem(pool, eligible, incumbent, protocol, profile) -> Problem`.
Report all unrounded floors/limits; no hardcoded 0.8124 from printed summaries.
Implement tail count and cheap necessary-condition diagnostics.

Tests: exact floor boundaries, genus order alignment, empty MDR mapping error,
strict >0.5 breadth, >0.8 tail count, K not replaced by available count.

### A04 — Bounded MILP with conflict cuts

Implement solver loop, shared wall-clock budget, sparse conflict storage and
status normalization. Use actual solver bound/gap; never invent zero gap.
Validate integer solution and every original numeric constraint in float64 with
documented tolerance (default 1e-6); reject values outside allowed tolerance.

Tests: enumerate all subsets of a <=12-candidate synthetic pool and compare
optimal objective/feasibility; pair conflict discovered after first solve;
fractional/timeout/cut-limit responses cannot masquerade as optimal; no dense
quadratic distance matrix; no fallback relaxation. Mock timeouts reliably.

### A05 — Portfolio diagnostics and export

Recompute pairwise distances and all metrics from selected IDs, not solver
reported aggregates. Apply acceptance separately from feasibility. Export
original sequences only, with input membership and exact reference gates.
Optional random-25 summaries are descriptive; they do not estimate assay yield.

Tests: average-distance failure stays explicit; fake solver success rejected;
stable rank; baseline copy/replay parity; corrupt solution fails resume;
no file named top.fasta for an invalid/incomplete solution.

### A06 — Reports, native-baseline context and SSH runbook

Implement report CLI and `docs/RUNBOOK_CONSTRAINED_SELECTIVITY.md`. Show every
seed/profile/status, risk reduction and activity/coverage margins, source identity,
retrospective exposure, and `retain_incumbent` default recommendation. Optional
native-baseline metrics are context only; require sidecar identity or label them
user-reported, never pretend they came from these caches.

Tests: infeasible/unknown/null metrics displayed honestly, all seeds present,
no claim of safety from scores, no automatic promotion. Run fixture end-to-end
and existing selection/risk-cache regression tests. Confirm every CLI flag with
`--help`; no real model downloads in CI. Final handoff includes exact runnable
CPU commands and files to paste back from SSH.

## 11. Workstream B: more meaningful hemolysis evidence

Implement the ESM plan's T00–T06 data/predictor foundations first. Reuse their
chemistry-aware molecule IDs, observation bounds, study/family splits, train-only
transforms, censored concentration likelihood and frozen ESM cache.

### B01 — Endpoint parser repair and adapter tests

Retain raw measurement type, concentration, lysis interval, RBC source, assay
conditions, chemical identity, source locator and study ID. Explicit HC50 only
maps to HC50. MHC must not automatically map to 50% lysis; IC50 requires confirmed
erythrocyte-lysis semantics. Current parser's automatic assignments are legacy
behavior to test against, not a truth source. Do not overwrite old labels.

Accept numerical/interval percent lysis in [0,100], positive dose with units,
and HC50 exact/left/right bounds as different observations. No midpoint collapse
of bands, guessed chemistry conversions or imputation of absent measurements.
Tests must include 0%, 100%, ambiguous MHC, non-RBC cell death, and same sequence
with different termini. Each rejected row has a reason and traceable source.

### B02 — Source extensions and paired support

Add adapters for explicitly versioned DBAASP, Hemolytik/DRAMP and complete study
supplements only when raw fixtures, license and source fields are available.
Deduplicate primary experiments across databases. Track measured low-lysis
examples and measured inactive variants, not assumed negatives. Within-study
variant pairs are useful but must stay in the same held-out family/study grouping.
Report paired activity/hemolysis support and chemistry compatibility before fitting.

### B03 — Dose-conditioned hemolysis model (optional supported extension)

New modules under the existing design_research package: `dose_response.py` and
tests. First implement a deliberately simple monotone logistic-normal curve:

`logit(lysis_fraction) ~ Normal(intercept(features, context) + slope*log2(c), sigma)`.

Use global positive slope and sigma initially, parameterized by softplus with
declared numerical floors; per-sequence slopes are a later ablation. The sigmoid
of the location parameter is the median, not the mean fraction. For a band [l,u]
in fraction units, transform endpoints with logit and use the stable Gaussian
interval likelihood already implemented in the ESM plan. Map 0/1 endpoints to
negative/positive infinity; a [0,1] interval is uninformative. Numeric exact 0/100
measurements require source-declared detection/rounding intervals; unknown
resolution cannot be guessed merely to make the likelihood defined. Interior
exact fractions use the logit-normal density including its Jacobian when that
measurement interpretation is declared. Fix sigma across dose so the predicted
distribution is stochastically monotone. Never substitute interval midpoints.

This curve imposes substantial assumptions. Test finite tail losses, monotonicity,
interval widening, dose support, missing context, endpoint boundaries, and
held-out calibration. Exclude incompatible sources. Do not extrapolate a HC50
from a single low-dose low-lysis observation. Report insufficient identifiability
when no suitable concentration coverage exists. The HC50 censored model can
proceed independently if its own support gate passes.

### B04 — Frozen evaluation roles

Create an evaluator registry with model/training-data/calibration identities,
endpoint meaning, overlap disclosure and roles: selection, validation, final test.
APEX is an activity cross-check, not independent of AMP-Diffusion's selector.
Different heads trained on the same data are not independent biological evidence.
An evaluator used to tune the selector loses its untouched evaluation role.
Implement same-sequence scoring and manifest joins for all native top-100 sets;
never compare one model's raw probability numerically with another model's MIC.

Before final evaluation, freeze choices and audit uninspected study outcomes.
Report family/study uncertainty and assay support. If no credible independent
evidence exists, state it and retain surrogate status. No fabricated evaluator
or automatic promotion to make the workflow complete.

## 12. Workstream C: candidate supply and conditional generation

Use ESM plan T07–T13 for training/export infrastructure and controls. Add these
requirements before real selective generation:

### C01 — Activation decision and experimental budget

Record A's exact limiting profiles and solver status plus B's support report.
Unknown computational feasibility is not proven supply failure. Predeclare a
fixed draw budget, all training seeds, validation criteria and no automatic
extension after unfavorable results. Missing paired outcomes disables joint
selectivity conditioning. Chemistry must match the competition's allowed form.

### C02 — Conditional decoder before architecture expansion

First compare the existing decoder fine-tuned with measured joint biological
conditions against matched continued-training unconditional and activity-only
controls. Use masks for missing observations; do not turn missing hemolysis
into low risk. Mix permitted broad AMP training data as a declared retention
mechanism; select its fixed mixing ratio using validation only. Reuse biological
condition schemas from the ESM plan and keep physicochemical conditions distinct.

Only then compare sequence-CVAE and ESM-CVAE at matched candidate budgets.
No unbounded RL on deployed scores. Adding reward optimization is a separate
registered experiment after validated targets exist.

### C03 — Yield and integration

Report eligible distinct sequences and sequence clusters satisfying declared
joint surrogate criteria per raw draw, together with failures and compute cost.
Apply identical feasibility criteria to every generator. Repaired predictors
change the score scale: rescore incumbent and all baselines rather than compare
new probabilities with historical numbers. Generation and evaluator exposure
must respect the same held-out boundaries.

Any specialist-derived top candidate must belong to the submitted library.
If adding specialist sequences, use a separately versioned deterministic blend
with explicit membership accounting, then rerun all full-library metrics and
repeatability checks. Exact preservation of old library metrics is no longer
claimed. A separate decision is required before adopting the new submission.

## 13. Definition of completion

A software completion: A00–A06 pass fixtures and relevant regressions; all CLI
interfaces exist; frozen historical bundles can be imported without mutation;
feasibility and uncertainty are correctly distinguished; runnable SSH runbook
delivered. Real-data results may be infeasible or unknown and still constitute
an honest software outcome. No new training is required to complete A.

B/C completion is tracked separately using the linked queue and B/C additions.
Insufficient biological support is a reported result, not permission to invent
labels. Final claims of improved hemolysis require prospective measurements at
specified conditions alongside activity; software completion cannot supply them.
