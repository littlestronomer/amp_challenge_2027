# Small-model implementation queue: ESM conditional peptide design

This is a plan, not evidence that these modules/commands already exist.
Read [the design specification](PLAN_ESM_CONDITIONAL_PEPTIDE_DESIGN.md) first.
Implement one task at a time in order. Do not replace scientific stop conditions
with convenient defaults merely to make a command complete.

## How to execute this queue

Use this prompt for each task:

> Implement task Txx from docs/TASKS_ESM_CONDITIONAL_PEPTIDE_DESIGN.md using
> docs/PLAN_ESM_CONDITIONAL_PEPTIDE_DESIGN.md as the contract. Read the listed
> existing files first. Implement only this task and its focused tests. Preserve
> production checkpoints, datasets, outputs and default generation behavior.
> Use synthetic fixtures and fake ESM features for tests. Run the task's checks,
> summarize changed files and limitations, and record the handoff. Do not start
> model downloads, real-data training, or final-test evaluation just because the
> implementation is complete. If a dependency is absent, report the exact
> dependency rather than silently replacing it with a stub in production code.

An implementation handoff records: task ID, commit/diff, modules/APIs, tested
commands/results, artifact schema changes, known limitations, and next task.
Use `docs/DESIGN_RESEARCH_IMPLEMENTATION_LOG.md` once implementation begins.
Do not mark a task complete because its functions exist; acceptance behavior
must pass. No requirement to ask the user after every small implementation task
when a broader implementation request has already authorized that work.

All package paths below are relative to
`src/amp_challenge_2027/design_research/`. The normal test suite needs neither
downloads nor a GPU. New libraries require an actual unmet need; NumPy, Torch,
Transformers and existing project dependencies are adequate for v1.

## Dependency graph

```text
T00 contracts -> T01 measurements -> T02 dataset -> T03 splits -> T04 conditions
                                           T03 -> T05 ESM cache -> T06 outcomes
                              T04 + T05 -> T07 decoder -> T08 CVAE
                                      T06 + T08 -> T09 bounded training
                                  T04 + T09 -> T10 generation
                                  T06 + T10 -> T11 candidate selection
                             T03 + T09 + T11 -> T12 sealed comparison
                                     T10 + T12 -> T13 release and runbook
```

T07 can be tested with fixture conditions without real data. Data feasibility
blocks real training/claims, not isolated implementation and numerical tests.

## T00 — Define contracts and output manifests

Read: `scripts/experiment_utils.py`, `evidence_inventory.py`, existing
`tests/test_competition_readiness.py`, and the design's sections 5 and 14.

Create:

* `__init__.py`, `schemas.py`, `artifacts.py`.
* `tests/test_design_artifacts.py`, `tests/test_design_schemas.py`.

Implement explicit record validation using dataclasses or small validators.
Fail on unknown schema versions, invalid enums, empty identifiers, nonfinite
numbers and incompatible endpoint fields. Preserve optional unknown values.
Define a small static per-stage output registry; include `run.json` in every
stage's completion hashes. Validation verifies all required dependencies and
safe relative paths. Reject directory overlap with any source/protected root.

Create `experiments/design_research_v1.json` only as a validated research
protocol: immutable seeds, budgets, support rules, architectures and split
rules. ESM revision starts explicitly unresolved; real encoding refuses until
an exact revision is recorded. No fabricated hash or `main` revision fallback.
The implementation log must distinguish unresolved from usable configuration.

Acceptance: manifest inventory round-trip succeeds; changing any file including
`run.json` fails; missing marker entries have an explicit message; existing
outputs cannot be silently resealed; `--list` helpers make no writes.

## T01 — Preserve measurements and chemical identity

Read: `scripts/audit_label_observations.py`,
`scripts/reconcile_observation_metadata.py`, `scripts/build_molar_candidates.py`,
and legacy activity/hemolysis label builders.

Create `measurements.py`, `chemistry.py`, `tests/test_design_measurements.py`.
Adapt existing deterministic parsers rather than copying divergent versions.
Normalize values/units into explicit intervals and retain original strings.
Implement exact threshold event logic with open/closed boundaries. HC50,
percent hemolysis, generic cytotoxic IC50 and stability are separate endpoints.
Unknown terminal chemistry remains unknown, and mass conversions require
known supported chemistry. Attach decision codes and source locators to every
conversion, exclusion, and unresolved record.

Acceptance examples:

* `MIC >64 uM` becomes a right-censored lower bound, not 64 or an exact label.
* `MIC >8 uM` is UNKNOWN for the event MIC<=16; it is still useful to NLL.
* `MIC =16 uM` implies LE_16; `MIC >16 uM` implies GT_16.
* `HC50 >128 uM` implies GE_128; `HC50 <128 uM` implies LT_128.
* `10% lysis at 32 uM` is not HC50=32 and does not create a HC50 label.
* Same residue string with free vs amidated termini has different molecule IDs.
* Lower-case stereochemical notation is not silently uppercased away.
* Unsupported units/chemistry yield reason-coded unresolved records.

## T02 — Build an auditable observation dataset and feasibility report

Create `dataset.py`, `scripts/design_prepare.py`,
`tests/test_design_dataset.py`, and small source fixtures under
`tests/fixtures/design_research/`.

Start with adapters for existing DBAASP observation/raw tables. Add DRAMP 4 and
one explicitly selected study-supplement adapter only with representative raw
fixtures and documented source fields. Do not advertise support for arbitrary
paper supplements without a parser. Keep source records read-only.

Outputs: `molecules.jsonl`, `observations.jsonl`, `source_registry.json`,
`curation_events.jsonl`, `feasibility.json`, `REPORT.md`, and manifests.
Feasibility reports endpoint/chemistry support, paired outcomes, unique
molecules/families where available, studies, exclusions and duplicates.
Physical duplicates referencing one source measurement are deweighted; repeated
biological observations retain identity. Do not select the minimum MIC.

Acceptance: duplicate source copies do not increase effective count; chemistry
variants remain separate; file-order permutation leaves identities unchanged;
row accounting reconciles exactly; source-less or unsupported records cannot
enter supported supervised data. Fixture `--list` makes no output directory.

## T03 — Implement family and study transfer splits

Read `generalization.py` and `scripts/prepare_reward_generalization.py`.
Create `splits.py`, `scripts/design_split.py`, `tests/test_design_splits.py`.
Update `design_prepare.py` to call split/feasibility logic for complete datasets.
Follow the nested immutable data-bundle contract in design section 14; a
standalone split command never adds files to an already sealed dataset stage.

Outputs: `split_assignments.jsonl`, `family_map.json`, `split_audit.json`,
`exposure_audit.json`, and partition support tables.
Family grouping acts on all endpoints/sources and chemistry variants sharing a
sequence. Study holdout is a separate named protocol with post-removal of
training near-neighbors. Use stable seeds/order and train-only vocabularies.
Report dropped boundary rows and giant components, never break them secretly.

Record previously inspected datasets and generator pretraining exposure. A
metadata-only source inspection is distinct from reading target values, but
already viewed test outcomes cannot become a new untouched test.

Acceptance: transitive near-neighbor bridge stays within one partition;
endpoint duplicates never cross splits; source order cannot alter assignments;
study holdout reports zero declared-study overlap; generator corpus overlap is
detected; insufficient support raises an ineligible-data result.

## T04 — Encode experimental conditions and support checks

Create `conditions.py`, `scripts/design_prepare_conditions.py`,
`tests/test_design_conditions.py`.

Outputs: `condition_schema.json`, `condition_examples.jsonl`,
`condition_support.json`, `condition_rejections.jsonl`.
Finish the `design_prepare.py` orchestrator and `data_manifest.json` so the
documented downstream `--data` root resolves observations, splits and conditions
without any omitted preparation command.
Implement the section 9 schema and exact boundary rules. Fit vocabularies on
training only. Use actual observations and retain supporting IDs. UNKNOWN and
MASKED are distinct; masks never change a biological label into a negative.
Molecule weighting must not grow with its number of observations.

Only same-chemistry, supported-context pairs may form the primary joint event.
Do not infer a joint condition from independent counts of activity and HC50
labels. Reject unsupported joint requests before loading a generator.

Acceptance: measured negatives remain usable conditions; incomplete pairs do
not become selectivity examples; test-family counts cannot make a request pass
support; a fixed RNG produces identical record choices/dropout masks.

## T05 — Extract and verify frozen ESM features

Read `reward_benchmark.FrozenEncoder` and the cache handling in
`scripts/benchmark_reward_generalization.py`.
Create `embeddings.py`, `scripts/design_cache_esm.py`,
`tests/test_design_embeddings.py`.

Use dependency injection for a fake encoder in tests. Real encoder loads pinned
35M ESM and validates residue-only mean pooling. Cache float 32[N,480], ordered
sequence IDs and backbone metadata. ESM gradients are disabled. Chemistry
variants can share a sequence embedding but not outcome labels or weights.
Save recoverable chunks and verify them before reuse; partial output has no
root completion marker. Do not silently change precision on OOM.

Acceptance: fake special-token/padding embeddings cannot contaminate pooling;
wrong order, dimension 320, dtype, revision and corrupt chunk fail; resume does
not recompute completed rows; test extraction requires an explicit sealed
evaluation context. `--list` never loads/downloads ESM.

## T06 — Fit endpoint distributions with censored losses

Create `likelihoods.py`, `outcome_model.py`, `calibration.py`,
`scripts/design_train_outcomes.py`, and focused likelihood/outcome tests.

Implement exact/interval/one-sided Gaussian NLL in log2 concentration. Use
stable tail calculations and document likelihood units/reductions. Support
linear and MLP models with identical data/context. Train per prescribed seeds;
retain all members. Fit predictive scale on calibration only when supported.
Never train on HC50 values fabricated from dose/lysis-band observations.

Also implement H1's legacy-style binary-label baseline as an explicit model
variant: use the documented legacy aggregation on training observations only,
the same allowed ESM features and matched head capacity, and binary event loss.
Record exclusions caused by aggregation. Evaluate it and the repaired model on
the identical determinate-event cohort with the T12 metrics. This ablation tests
the whole label-repair package; it cannot by itself attribute gains separately
to context, censoring and aggregation. Such attribution needs further registered
ablations. Never reuse a deployed head trained on held-out families as this control.

Outputs: each member checkpoint/config/history; `validation_metrics.json`,
`calibration.json`, `support_profile.json`, `model_inventory.json`, manifests.
No final-test outputs belong in training directories.

Define support deterministically: supported context plus a nearest-training
similarity threshold fitted as the 5th percentile of valid validation queries'
nearest-training similarities, per supported endpoint/context. This is a
heuristic domain-of-support boundary, not an accuracy guarantee. If too little
validation support exists, return unknown/unsupported rather than fitting a
threshold on generated candidates. Freeze the profile before sample selection.

Acceptance: stable finite losses/gradients in extreme tails; widening intervals
does not decrease their probability; exact-loss reference parity; unsupported
entirely one-sided-censored fits fail; masked labels have zero gradient; split-specific access
checks reject training on calibration/test; changing test data cannot change fit.

## T07 — Implement the prefix-conditioned research decoder and AR controls

Read production `model.py`, `generator.py`, `tokenizer.py`; do not modify their
checkpoint formats/default paths.
Create `conditional_model.py`, `tests/test_design_decoder.py`.
Implement `ConditionEncoder` and `ResearchDecoder` with the exact sizes and
alignment in section 10. Keep a causal sequence mask and separate padding mask.
Use the existing token vocabulary, not text tokenization for amino acids.

Support `ar_unconditional` and `ar_conditioned` via neutral vs condition-derived
prefixes. Unit fixtures can overfit a tiny artificial sequence mapping to verify
that conditioning is wired in; do not interpret that as biological evidence.

Acceptance: prefix positions receive no next-token targets; altering future
residues cannot change earlier logits in eval mode; prefix gradients exist;
PAD is ignored; conditional and unconditional models have explicit configs;
production generation/checkpoint tests remain unchanged.

## T08 — Add the conditional latent prior and posterior

Extend `conditional_model.py`; add `tests/test_design_cvae.py`.
Implement Gaussian posterior/prior, explicit-generator reparameterization,
analytic KL, warmup, and diagnostics. Add `cvae_sequence` and `cvae_esm` with
the same prior/decoder/condition architecture. The sequence control encoder may
use residue embeddings plus a bidirectional GRU and project to 480; report its
parameter/compute difference from ESM rather than claiming identical encoders.

Acceptance: posterior/prior identical -> KL 0; posterior gradient nonzero;
frozen-ESM feature inputs have no trainable backbone; latent prefix actually
affects logits; de novo prior sampling requires no x/target embedding; seeded
latent samples repeat; dropout condition is shared consistently across q,p,decoder.

## T09 — Implement bounded training and training-state resume

Create `training.py`, `scripts/design_train_generator.py`,
`tests/test_design_training.py`.
Implement the frozen pilot/final budgets, AdamW, gradient clipping, explicit
RNG state, data-loader order and molecule-normalized weights. Train all controls
on the same allowed sequence/condition data and record parameter/compute counts.
Generator validation uses a fixed declared diagnostic recipe and latent RNG;
do not call noisy reconstruction loss exact sequence likelihood.

Checkpoint selection: minimum validation objective within each architecture;
architecture selection is a predeclared multi-metric validation decision using
prior-sample diagnostics, never final-test metrics. Record posterior-collapse
diagnostics and condition sensitivity; do not secretly alter beta after failure.
No automatic final-test evaluation, new architecture sweep or checkpoint promotion.

Acceptance: interrupted/resumed fixture training matches uninterrupted same-
runtime training; changed recipe fails with actionable differences; loss is
finite on partial-label records; training never accesses test partition;
checkpoint selection does not choose the best random seed.

## T10 — Generate from the prior and export standalone research models

Create `sampling.py`, `scripts/design_sample.py`, `scripts/design_export.py`,
`tests/test_design_sampling.py`.
Implement section 11 budgets, constraints and exports. Derive all RNG streams
from explicit recorded seeds. Requests pass support checks before inference.
Write `raw_candidates.jsonl`, `accepted.fasta`, `sampling_summary.json`, request
records and manifests. Include attempted draws, rejected draws, duplicate and
novelty counts; a rejection is never silently replaced without counting attempts.

Sampling loader needs only decoder/prior/condition components. The export can
retain training/posterior files for provenance, but inference must not load them.
Keep library-scale generation distinct from a 2,048-draw pilot and never present
an undersized research pool as a valid competition library.

Acceptance: export samples after posterior/ESM loaders are monkeypatched to
raise; same recipe produces identical fixture bytes; different seeds change
stochastic draws; unsupported conditions fail; length/EOS/special-token handling
works at both boundaries; checkpoint kind cannot be misread as production.

## T11 — Score candidates and select a supported, diverse portfolio

Create `selection.py`, `scripts/design_score_candidates.py`,
`scripts/design_select.py`, `tests/test_design_selection.py`.
Score generated candidates with the frozen T06 models. Predictions are linked
by sequence/molecule/request identity and include support/calibration flags.
Candidate ESM embeddings have their own cache identity, separate from training
features. Reference novelty uses the official filtering implementation.

S0/S1 consume the identical pool/predictions and apply section 12 rules. Output
`top.fasta`, `top_scores.csv`, `funnel.json`, `subset25.json`, `REPORT.md`,
and manifests. Rank with stable ties. Implement conservative marginal constraints
without inventing independence or lab-confirmed safety.

Acceptance: pool order cannot change tie policy; novelty rejection is strictly
`>0.8`; family caps and output membership hold; unsupported rows never acquire
finite substituted certainty; underfill fails; all funnel counts reconcile;
subset simulation is explicitly labelled surrogate/descriptive.

## T12 — Seal evaluations and generate matched comparison reports

Create `reporting.py`, `scripts/design_seal.py`, `scripts/design_evaluate.py`,
`scripts/design_compare.py`, `tests/test_design_evaluation.py`.

`design_seal.py` freezes protocol/checkpoint/support-profile hashes, selected
architecture, selection policies, requested metrics, exposure audit and reason
for the decision before final-test reads. Test ESM extraction/evaluation needs
the matching seal. A changed model invalidates it; completed evaluations resume
by verifying hashes, not by creating a new untouched test claim.

The H1 binary-label baseline and repaired outcome model are compared on the
same determinate threshold events using Brier/rank metrics. Do not compare
binary cross-entropy numerically to continuous-density NLL as if identical
targets. Gaussian linear/MLP comparisons use the same endpoint NLL convention.

Reports distinguish: training/validation/calibration/test; family/study split;
posterior reconstruction/prior generation; raw/postselection candidates;
retrospective/prospective observations; and software completion/scientific result.
Include grouped uncertainty, all seeds, budget parity, condition support,
missing-data counts, label provenance, data exposure and null/failed outcomes.
The report's deployment recommendation defaults to `retain_incumbent` unless
an explicit later promotion protocol has been fulfilled.

Acceptance: target access before seal fails; modified checkpoint fails;
paired grouping preserves family/study identity; undefined metrics remain null
with reason; shuffled conditions cannot be omitted from a registered ablation;
surrogate improvements never print 'experimentally validated'.

## T13 — Integrate artifacts, write the runbook and verify a clean environment

Create `docs/RUNBOOK_ESM_CONDITIONAL_DESIGN.md` and update the implementation log.
Add optional evidence inventory sources for completed research stages using
their actual registry output names; keep old evidence configs intact where
possible and use a new config version for the research pipeline.

Deliver runnable CPU fixture commands, GPU 1 pilot commands, expected output
filenames, resume behavior and failure recovery. Confirm every documented CLI
flag with `--help`; clearly identify any not-yet-implemented optional stage.
Add `tests/test_design_research_pipeline.py` using the full tiny fixture flow.

Run a clean-environment research export smoke without development-only imports.
Separately document the organizer-style `uv sync` plus default generation check
for the incumbent. Do not change production entry points simply to make the
research export available. A later production adoption is a separate explicit
task after a scientific decision; this plan does not automatically authorize it.

Acceptance: all planned software tasks pass; no protected artifacts changed;
documented commands are real; fake-ESM end-to-end tests require no network;
real-data pilot is either recorded with limitations or honestly ineligible;
the user can pull the implementation and run it on physical GPU 1.

## Planned evaluation command surfaces

These complete the interfaces from the design document and are not runnable
until T11–T13 have been implemented:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/design_score_candidates.py \
  --samples sweep_results/design-research-v1/pilot-samples \
  --outcomes sweep_results/design-research-v1/outcomes --device cuda \
  --out sweep_results/design-research-v1/pilot-predictions

uv run --no-sync python scripts/design_select.py \
  --samples sweep_results/design-research-v1/pilot-samples \
  --predictions sweep_results/design-research-v1/pilot-predictions \
  --protocol experiments/design_research_v1.json \
  --reference data/antibacterial.fasta --policy S1 \
  --out sweep_results/design-research-v1/pilot-selection

uv run --no-sync python scripts/design_seal.py \
  --protocol experiments/design_research_v1.json \
  --data sweep_results/design-research-v1/data \
  --outcomes sweep_results/design-research-v1/outcomes \
  --generator sweep_results/design-research-v1/cvae-esm-seed42 \
  --decision experiments/design_evaluation_decision_v1.json \
  --out sweep_results/design-research-v1/seal

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/design_evaluate.py \
  --seal sweep_results/design-research-v1/seal --device cuda \
  --out sweep_results/design-research-v1/test

uv run --no-sync python scripts/design_compare.py \
  --protocol experiments/design_research_v1.json \
  --runs experiments/design_comparison_runs_v1.json \
  --out sweep_results/design-research-v1/comparison
```

The decision/run-list JSON files are explicit manifests of completed artifacts;
do not implement automatic recursive discovery of whichever experiment happens
to score best. Final matched experiments include all prescribed architectures
and seeds even if the illustrated seal command names only one pilot generator.

## Suggested implementation milestones

| Milestone | Tasks | Useful result even if later models fail |
|---|---|---|
| M1 data feasibility | T00–T04 | Audited observations, split integrity, actual joint-condition support |
| M2 trustworthy targets | T05–T06 | Reusable ESM cache and tested censored-endpoint baselines |
| M3 ESM design hypothesis | T07–T10 | Working prior-sampled conditional models and informative controls |
| M4 evidence and handoff | T11–T13 | Matched evaluation, reproducible research artifacts, SSH runbook |

Before the competition deadline, prioritize M1 and a bounded M2 pilot only if
supported. M3 can be implemented/tested with fixtures independently; do not
promise a scientifically evaluated new submission in that time. Subsequent
prospective assays, novel chemistry, active learning, and clinical-development
claims require a separately designed research stage.
