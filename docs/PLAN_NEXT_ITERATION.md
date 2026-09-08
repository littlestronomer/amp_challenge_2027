# Next iteration: separate candidate quality, selection effects, and predictor reliability

Planning date: 2026-09-08. Baseline code: `9358eab97fe0d396805df4e50649698b3ba539ee`.

Status: **selection ablation completed on SSH; predictor reconstruction ready for SSH**.
Phases 0/1A and the P0/P1 mechanics are implemented, together with a CPU
artifact-identity inventory for Phase 1B. The remote audit matched all three
heads to frozen-training members. Original split/data manifests were not found;
a separately labelled, hash-pinned reconstruction evaluator is now implemented.
See [the selection runbook](RUNBOOK_SELECTION_DIAGNOSIS.md) and
[the reconstruction runbook](RUNBOOK_REWARD_RECONSTRUCTION.md). Historical
validation remains unverified; pool hemolysis inference, risk-weight tuning
and prospective runs remain gated follow-ups. Nothing here promotes a
checkpoint or replaces a classifier. The proposed names below describe the full plan; the runbook maps
the currently available outputs and commands.

User-reported SSH update: P0 replay and the P1 anchor passed. Fixed normalization
retained all 100 incumbent selections per seed and 99/98/100 candidate selections
at seeds 42/43/44. It did not consistently improve candidate activity/panel scores
and reduced mean candidate sequence distance. No promotion followed this ablation.

## 1. Objective and decision to produce

Determine whether the next improvement should target:

1. Top-100 selection: useful candidates exist but the current selector misses them.
2. The evaluation/scoring models: the risk or activity signals are insufficiently
   reproducible, calibrated, or supported outside training data.
3. Generator/data quality: the available libraries lack an acceptable candidate
   trade-off even after a controlled selection comparison with credible signals.

The iteration succeeds when it produces a reproducible diagnosis and a justified
next action. It need not produce a winning submission. Do not repeatedly expand
a sweep merely to obtain a favorable result.

Retain `hybrid` as the incumbent comparison and `p3_s1` as the epoch-58 75/25
candidate. Neither is designated experimentally safe. Keep their library choice
separate from the choice of top-100 selector.

### Starting evidence

Means over the three already-inspected generation seeds, 42–44:

| Readout | Incumbent hybrid | Epoch-58 75/25 | Interpretation |
|---|---:|---:|---|
| Library FBD | 0.218783 | 0.211041 | Candidate lower |
| Library MMD | 0.360370 | 0.233094 | Candidate lower |
| Top activity score | 0.843936 | 0.837368 | Candidate lower on average |
| Top panel mean probability | 0.792783 | 0.785156 | Candidate lower on every seed |
| Top predicted hemolysis risk | 0.742986 | 0.766799 | Candidate higher on every seed |
| Top mean pairwise sequence distance | 0.673019 | 0.666009 | Candidate lower on every seed |

The candidate is not an all-metric winner. These are local distribution metrics
and model predictions, not measured potency, hemolysis, or synthesizability.
The top-distance statistic is not seqme's library Diversity metric.

### Non-goals for the initial iteration

- No new generator architecture, training, decoding sweep, or blend-ratio sweep.
- No changes to the six measured 50k libraries or their existing source manifests.
- No new 650M FBD/MMD runs for byte-identical libraries.
- No arbitrary probability threshold advertised as a biological safety cutoff.
- No automatic promotion of a better surrogate score into production defaults.
- No large multi-objective grid before the diagnostic stages finish.

## 2. Execution and data contract

Development, unit tests and Git operations happen locally. Full scoring and
experiments run on the user's SSH machine; the user handles SSH access.

- Local repository: `/home/littlestronomer-mobile/Documents/Projects/amp_challenge_2027`.
- SSH repository: `/home/istke/Documents/Project/amp_challenge_2027`.
- Commands in the eventual runbook will assume the SSH repository as the working directory.
- Code handoff: implement and test locally, commit/push, then the user runs
  `git pull --ff-only origin main` and the documented commands remotely.
- Preserve the existing locked environment. Do not insert an unnecessary
  dependency upgrade between related experiments. Record any required change.
- Use a free GPU, typically `CUDA_VISIBLE_DEVICES=1`, and `tmux`. No concurrent
  writers to an output/cache directory; no unattended monitoring is implied.

All relative paths below are relative to the appropriate repository root.

### Frozen inputs

- `sweep_results/epoch58-top100-v1/`: six completed comparison cells, including
  ordered `library.fasta`, `scores.npz`, `top.fasta`, `top_scores.csv`, summaries,
  completion markers, root run manifest and resolved backbone revisions.
- `sweep_results/checkpoints-seed44-v2/`: seed-42 source libraries and measurements.
- `sweep_results/checkpoint58-confirm-v1/`: seed-43/44 source libraries and measurements.
- `sweep_results/epoch58-blends-v1/`: seed-42 candidate blends.
- `sweep_results/epoch58-blend-confirm-v1/`: seed-43/44 candidate blends.
- `data/antibacterial.fasta`: reference sequence set and file hash.
- The activity, panel and hemolysis weights, their actual per-artifact configs,
  and any original training-member files, logs, label snapshots and split records.

The existing `scores.npz` contains activity, panel, conformity and precision
arrays for each full library. **It does not contain full-library hemolysis
scores.** Those exist only for the selected top-100 in `top_scores.csv`.

New analysis may import verified scores from the older code revision, but must
record their original provenance explicitly. Do not edit old `run.json` files
or rerun an old experiment in place to bypass exact-code resume checks.

## 3. Phase 0 — preserve and inventory the evidence

Compute: CPU, file inspection and hashing only.

### Work

1. Validate all six source cells against completion markers, library hashes,
   array schemas, sequence order and counts. Reject missing, nonfinite or
   misaligned scores. Never join array rows to a newly sorted FASTA.
2. Record code/runtime, source manifests, classifier hashes/configs, reference
   hashes and backbone revisions in a new iteration manifest.
3. Recompute the reported aggregate means and paired deltas from the actual
   remote artifacts. Deduplicate control rows by source/case/seed/hash; copied
   control CSVs are not additional replicates.
4. Label seeds 42–44 as development/exploratory evidence for this iteration.
   We have already examined their outcomes and used them to choose this plan.
5. Preserve historical records, but mark the earlier low-risk/no-safety-exposure
   narrative as not applicable to the current frozen scorer and selections.
   Do not erase the older measurement or represent a scorer change as biology.

### Outputs

Proposed new root: `sweep_results/selection-diagnosis-v1/`.

- `run.json`: immutable input and protocol record.
- `inventory.json`: artifact paths, hashes, schema and completeness checks.
- `baseline_results.csv` and `baseline_paired_deltas.csv`.
- `DECISIONS.md`: starting questions, chosen metrics, stage decisions and caveats.

### Gate G0

All six cells must be complete and traceable before a pooled conclusion.
If a cell is unavailable, report the reduced scope explicitly; do not silently
substitute another seed, model or library. A cache issue is not a reason to
change weights or regenerate the entire experiment.

## 4. Phase 1A — explain the current selector using cached scores

Compute: CPU only; no new neural inference. This work can proceed independently
of the provenance investigation in Phase 1B.

### Questions

- How much do per-library standard deviations change effective metric weights?
- Which components determine the shortlist and which distinguish the selected tail?
- Are breadth and MDR saturated only in the top-100, or earlier in the selection process?
- Does diversity loss reflect the shortlist, the final selector, or both?

### Measurements

For each library and each component, record mean, standard deviation, quantiles
(p10, p25, p50, p75, p90, p99), near-zero variance and tied-value fractions.
Compute the effective coefficient `weight / denominator`, using the current
code's denominator of 1 when standard deviation is <= 1e-8.

Inspect five populations separately:

1. All 50,000 library members.
2. Members passing the unchanged plausibility filter.
3. The current score-ranked shortlist, at most 2,000 sequences.
4. Shortlist members surviving the exact reference novelty check.
5. The selected top-100.

Report standardized component contributions, rank correlations, activity and
panel score distributions, breadth/MDR threshold saturation, and top overlaps.
Report continuous per-panel probabilities alongside threshold counts. MDR
breadth overlaps the broader panel and must not be presented as an independent
validation instrument.

The implementation standardizes separately within each library. Changing the
standard deviation changes relative ranking weights; subtracting a different
mean alone only adds a constant and does not change within-library ordering.
This is a possible explanation of the outcome, not an established cause.

### Outputs

- `component_distribution.csv`, `effective_weights.csv`.
- `selection_funnel.csv`: counts and distributions at each selection stage.
- `component_contributions.csv`, `rank_correlations.csv`, `top_overlap.csv`.
- A concise diagnosis distinguishing observations from causal hypotheses.

### Gate G1A

The report must quantify normalization and saturation rather than merely
asserting that they matter. No ranking weights change in this phase.

## 5. Phase 1B — audit the deployed predictors and their evidence

Compute: CPU provenance checks first; then a small 35M-backbone inference job
on the original validation examples if their identity can be established.
Prioritize hemolysis; audit activity and panel with the same provenance rules.

### B1. Establish artifact identity

Locate the actual original run directories, member summaries and label inputs
on the SSH machine. Match deployed heads to member heads by hash or exact
tensor equality if serialization differs. Verify the model ID, frozen backbone,
complete head tensors, output-label order and stored temperature.

Record the training seed, split algorithm/code revision, winning member,
data snapshot and record ordering. A current `members.json` in a reused
directory is not sufficient by itself; binary and panel training may have
overwritten shared metadata.

Do not infer the winner from rounded AUROC or use the default `--member 2` in
`eval_deployed_panel.py` without evidence. That helper assumes `seed = 42 + member`
and reconstructs a panel random split only; it is not a general provenance audit.
Binary stratified and panel shuffled splits need their respective original logic.

Classify each claim as verified, reconstructed with stated assumptions, or
unrecoverable. If historical data identity cannot be established, do not claim
an exact historical reproduction simply because today's code produces a split.

### B2. Reproduce predictions and assess calibration

Using the actual deployed forward path and stored temperature, report:

- Validation example count, class prevalence, duplicates and train/validation overlap.
- AUROC and average precision; for panel, per-output support and masked metrics.
- Brier score and log loss, alongside a constant-probability baseline using
  training prevalence; report evaluation prevalence separately.
- Reliability bins with sample counts and uncertainty; report binning explicitly.
- Uncalibrated versus stored-temperature predictions, without refitting temperature
  to make the audit look better.
- Error/distribution breakdowns by sequence length, charge and hydrophobicity,
  with minimum-support warnings. These are diagnostics, not causal explanations.

Empty or single-class strata must be marked as undefined with their counts,
not assigned an apparently measured AUROC of 0.5 by a fallback. Uncertainty
estimates must describe the resampling unit; use peptide-family grouping when
it is defensible and do not treat repeated policy evaluations as new samples.

A difference greater than 0.001 in validation AUROC from a sufficiently precise
original record is a proposed **reproduction investigation trigger**, not a
scientific performance cutoff. Explain data, split, head, backbone or metric
differences; do not try several seeds and select the closest AUROC.

The current trainer selects checkpoints and fits temperature on the same
validation predictions. It also uses class-weighted BCE. Therefore stored
temperature plus a good AUROC does not establish calibrated probabilities on
generated peptides. A successful replay is a reproduction result, not an
independent generalization estimate.

### B3. Confirm the meaning and limits of the hemolysis target

Verify `active = risky`, source assay targets, concentration conversion, band
parsing, conflict handling, and the actual thresholds used in that training run.
Current builder defaults use lysis-band evidence and a 128 micromolar ceiling;
the resulting binary label is not measured percent hemolysis at a prescribed
therapeutic concentration and is not a therapeutic index.

If an independent evaluation set is available, document exclusion from head
training/model selection/calibration and audit sequence similarity across the
boundary. Do not repartition data already used to train a frozen head and call
that partition a new holdout. A future clustered training design must verify
cross-split similarity explicitly: greedy leader-cluster labels alone do not
guarantee that every high-similarity pair lies on the same side.

### Outputs and gate G1B

- `reward_artifact_audit.json`: provenance status and exact evaluated artifacts.
- `validation_predictions.csv`, `validation_metrics.json`, `reliability_bins.csv`.
- `label_audit.json` and a short assessment of usable versus unsupported claims.

If artifact identity/direction/reproduction fails, investigate the scorer before
safety-driven optimization. If reproduction passes but independent evidence is
missing or calibration is weak, allow explicitly exploratory relative-score
diagnostics; prohibit safety certification, probability-based clinical claims,
or automatic promotion. Do not insert unplanned classifier retraining into the
main selector iteration; propose a separate recovery branch when needed.
If an audited artifact changes, invalidate its dependent scores and scaling
parameters explicitly. Reuse unaffected components only through a verified
dependency manifest, never by mixing probabilities from different calibrations.

## 6. Phase 2 — isolate normalization with a bounded CPU ablation

Depends on G0 and Phase 1A. Predictor audit may still be in progress.

Evaluate exactly two policies on both libraries and all three existing seeds:

| Policy | Scaling | Component weights | Hemolysis weight |
|---|---|---|---|
| P0: replay | Per-library, identical to current production | Existing five weights | 0 |
| P1: frozen scaling | Fit once on the complete seed-42 incumbent, then freeze | Same five weights | 0 |

This is 12 selection cells, of which six P0 results are existing baselines to
replay/verify and six P1 selections are new. All reuse cached neural scores.
Use the existing zero-variance policy, fit without outcomes, and serialize
the anchor library hash plus every mean/standard deviation/denominator.
Do not fit scaling separately on candidates or later confirmation libraries.

Keep plausibility rules, selection seed 42, shortlist size 2,000, exact
Levenshtein novelty threshold <= 0.8, top size 100 and final diversity algorithm
unchanged. Do not simultaneously replace breadth with continuous probabilities,
change diversity strength, widen the shortlist or add risk penalties.

### Acceptance and outputs

- P0 must reproduce the saved `top.fasta` byte-for-byte using the stored scores.
- P1 on the seed-42 incumbent must match P0: its normalization anchor is identical.
- Every output library must be byte-identical to its source; every top list must
  contain 100 unique, valid, plausible members of that library passing exact novelty.
- Report top overlap, raw activity/panel distributions, component contributions
  and sequence diversity. Report paired library contrasts within each policy
  and paired policy contrasts within each library. Do not count them as new
  independent generation replicates.

New P1 selections may contain sequences without cached hemolysis predictions.
Leave that readout explicitly unavailable until Phase 3 scores them; never
substitute the original top-100 risk mean.

Outputs: `normalization.json`, `selection_results.csv`, per-cell `top.fasta`,
selection diagnostics and paired policy/library deltas.

### Gate G2

If the candidate gap changes materially under common scaling, record evidence
for a selector interaction, not proof of biological improvement. If it persists,
retain that negative result. P1 is an experimental policy, not a mandatory new
default; do not promote it merely because the interpretation is cleaner.

## 7. Phase 3 — measure predicted-risk enrichment in the available pools

Depends on G0 and an explicit G1B instrument status. Scoring for diagnosis is
allowed with a limited-evidence label; using scores to justify promotion is not.

Compute: one hemolysis-head inference pass over the union of the six libraries,
deduplicated by sequence. At most 300,000 unique inputs; actual count is measured.
No activity/panel/precision recomputation and no 650M evaluator.

Cache raw logits and stored-temperature risk scores keyed by sequence and by
head/config/backbone/runtime identity. Map results back to each original library
order. Verify overlapping sequences against existing top-100 risk scores;
unexpected differences beyond a predefined numerical tolerance stop merging.

Compare all-library, plausible-pool, shortlisted, novelty-passing-shortlist and
selected distributions.
Record mean, median, p75/p90/p95, empirical score quantiles, correlations with
activity/panel/property scores and enrichment relative to matched random controls.
Match random controls on length/charge as an additional diagnostic; do not use
matching to conceal unadjusted results or call controls experimentally safe.
Selection controls must satisfy the same plausibility and exact novelty rules;
report insufficient matched candidates rather than relaxing those rules.

Construct an exploratory activity/panel/risk trade-off report over plausible
candidates. A candidate cannot be counted as available for a valid top-100
until it passes the exact reference novelty check. Run that check on candidate
shortlists first and label any unverified feasibility counts as provisional.

### Gate G3

- Broad usable pool, but selected tail has substantially higher predicted risk:
  prioritize selector changes, conditional on predictor credibility.
- Risk elevated across the pool or strongly coupled to activity scores:
  inspect predictor/data support and the trade-off; do not assume the generator
  is the cause or simply increase a penalty until scores look favorable.
- Predictor disagreement or poor audit evidence: prioritize scorer validation.

Outputs: `hemolysis_scores.csv` (or a typed equivalent), integrity manifest,
`risk_enrichment.csv`, per-population distributions and updated P0/P1 top audits.

## 8. Phase 4 — optional, gated safety-aware selection

Start only after reviewing G1B, G2 and G3. This phase is not automatically
required. Define candidate retention tolerances and the interpretation of
risk scores before running the sweep; proposed weights are not biological thresholds.

### Bounded experiment

On seed 42 only, run both libraries at safety weights `0, 0.25, 0.5` with a
single frozen-scaling policy. Fit the safety component's scale on the same
seed-42 incumbent, after its full-pool risk scores exist. Keep the other five
weights and all selection constraints unchanged. The safety component is
`1 - predicted_risk`; its zero-weight cell must reproduce P1 exactly.

This is six selection cells: two zero-weight controls and four nonzero cells,
all using cached scores. Do not sweep normalization, breadth definitions,
diversity weights and risk weights together. If P1 is not retained as the
experimental common scale, explicitly amend and lock the policy before this stage.

Report the trade-off rather than an invented aggregate winner: risk mean and
upper tail, activity mean and lower tail, panel mean/per-output probabilities,
breadth, MDR breadth, pairwise diversity and nearest-reference similarity.
Declining risk alone cannot qualify a candidate if activity or diversity collapses.

Require predeclared tolerances for score/diversity losses and an explicit review
of the trade-off. The plan intentionally does not declare a universal acceptable
loss in surrogate probability. If tolerances are not agreed, present the full
small frontier without automatic selection. If no tested setting is acceptable,
stop and report it rather than automatically expanding the grid.

Carry at most one selected nonzero policy, applied consistently to both libraries,
to seeds 43 and 44 for exploratory consistency checks. These seeds have already
influenced this iteration and are not a clean held-out confirmation set.

### Gate G4

Once risk is used in selection, lower scores from that same head are an optimized
objective, not independent evidence of reduced hemolysis. Promotion requires
separate review and stronger supporting evidence. Neither predictor agreement
nor a favorable three-seed average constitutes wet-lab safety validation.

## 9. Prospective confirmation and final decisions

Only if a selector/library policy survives the diagnostic stages, freeze its
code, weights, calibration, normalization anchor, shortlist and decision rule.
Reserve fresh generation seeds before looking at outcomes: proposed 45, 46 and
47, subject to checking the SSH experiment inventory for prior use/inspection.

Generate paired incumbent/candidate libraries under that locked recipe and
apply the same selected policy and relevant controls. Reuse each generated
pool across policies. This is a separate, explicitly approved compute stage,
not part of the initial no-generation diagnostic commitment. New library bytes
require fresh 650M measurements if their FBD/MMD behavior is claimed.

Fresh generation seeds assess sampling stability conditional on fixed models;
they are not new training replicates or independent classifier/biological tests.
Do not keep tuning after inspecting the reserved results while retaining the
word "confirmation" for that same set.

The final decision memo must choose and justify one outcome:

| Evidence | Decision |
|---|---|
| Useful candidate trade-off exists, current ranking misses it | Improve selection; keep generator weights fixed |
| Predictor artifact, calibration or external support is inadequate | Improve/audit scorers; defer safety claims and promotion |
| No adequate candidate trade-off under credible evaluation | Propose a separate generator/data iteration |
| Benefits are small, unstable or come with unacceptable trade-offs | Retain incumbent and document a negative experiment |
| Candidate policy has reproducible, acceptable evidence | Prepare a candidate release for explicit review, not automatic promotion |

For any eventual release, require exact library/top validity, byte-reproducible
entry-point inference on the intended environment, correct frozen-head metadata,
complete weight packaging, and updated claims/disclosure. Independent potency,
hemolysis and synthesizability evidence remain separate scientific workstreams.

## 10. Proposed implementation and testing work

Implementation coverage to date:

| Addition | Current responsibility / remaining gate |
|---|---|
| `src/amp_challenge_2027/selection_audit.py` | Pure cache-schema checks, score summaries, normalization fit/apply, paired comparisons |
| `scripts/analyze_selection.py` | Verified cache import, inventory, baseline replay, normalization/saturation diagnostics |
| `scripts/audit_reward_artifacts.py` | Implemented: explicit artifact inventory and tensor matches; split validation/calibration and pool scoring remain gated |
| `scripts/eval_reward_reconstruction.py` | Implemented: explicit provisional random-split reconstruction with pinned current CSVs/heads, fixed-temperature metrics and similarity audit; never claims recovered historical validation |
| `scripts/sweep_top100_selection.py` | Implemented: P0/P1 ablation and exact source-library preservation; risk-weight comparison remains gated |
| Corresponding tests and an SSH runbook | Reproducibility, failure handling and ordered user-run commands |

Prefer reusing existing selectors, scoring functions and file-integrity helpers
without changing their production semantics. Keep experiments in new directories.
Any needed classifier-training or clustering changes are a separate scoped task,
not incidental refactoring during this audit.

Required tests across the full plan (risk-cache/weight tests apply to their later delivery):

- Cached rows bind to the exact ordered source library and reject tampering.
- Missing provenance, incomplete heads, changed model/config/backbone or nonfinite
  values fail closed; no fallback components or guessed split seeds.
- Baseline replay and anchor-policy identity are byte-exact.
- Common normalization is identical across libraries, independent of processing
  order, and handles zero variance without exploding an effective weight.
- Known synthetic examples demonstrate normalization-induced ranking changes.
- No alteration of source FASTAs, manifests or current defaults.
- Exactly 100 valid, unique, novel selections; insufficient candidates fail.
- Hemolysis zero weight reproduces the no-risk selector; risk direction is correct.
- Risk-cache joins preserve per-sequence identity and reject missing entries.
- Paired summaries do not duplicate controls or count policy cells as new seeds.
- Interrupted stages resume only verified work; changed recipes require new output.
- Model wiring tests use tiny local backbones; no full experiments on the laptop.

### Delivery sequence

1. Implement/test cache analysis and P0/P1 mechanics locally; push code and the
   exact runbook. User pulls and runs CPU stages on SSH.
2. Review diagnostics while locating original classifier records. Implement/test
   the explicit predictor audit; user runs the bounded validation job on SSH.
3. Score hemolysis once across the verified union if warranted; review enrichment
   and the updated P0/P1 results.
4. Decide whether Phase 4 is justified. Implement/enable only that bounded sweep
   if the review supports it.
5. Review again before allocating prospective generation/evaluation compute or
   altering a deployment artifact.

Do not estimate wall time from the old runbooks. Measure one representative
cell, report CPU/GPU time and memory, then estimate the remaining stage. The
initial plan has zero generator-training runs, zero new 50k generation runs,
and zero redundant 650M evaluations; expensive work is conditional on evidence.
