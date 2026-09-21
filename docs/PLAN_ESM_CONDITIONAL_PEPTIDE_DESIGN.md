# Implementation plan: ESM-informed conditional peptide design

Prepared 2026-09-21. Status: proposed research and implementation specification,
not an implemented model or a demonstrated scientific improvement.

Companion execution queue: [TASKS_ESM_CONDITIONAL_PEPTIDE_DESIGN.md](TASKS_ESM_CONDITIONAL_PEPTIDE_DESIGN.md).

## 1. Objective and scientific claim

Build a research pipeline that tests whether assay-aware data and ESM-informed
conditional generation improve discovery of novel, experimentally active and
selective peptides under a fixed candidate and measurement budget.

The research program aims to demonstrate the value of combining:

1. Observation-level, chemistry-specific experimental data with inequalities,
   failures, study identity, and explicit missingness preserved.
2. Predictions of measured endpoints with calibrated uncertainty on supported
   populations, including explicit rejection of unsupported extrapolation.
3. Conditional generation from a learned latent prior, using frozen ESM
   representations during training.
4. Evaluation on unseen families/studies and eventually prospective assays,
   with complete outcomes and matched controls released.

An ESM embedding plus a decoder, a conditional VAE, and uncertainty-aware
selection are not independently novel inventions. HydrAMP already uses a
conditional VAE; ApexGO combines generative models with optimization. Novelty
must come from a defensible methodological distinction and measured benefit.
No implementation result alone permits claims of clinical usefulness, safety,
MDR activity, or improved competition standing.

### Hypotheses and failure outcomes

| ID | Hypothesis | Comparison | Evidence that would reject it |
|---|---|---|---|
| H1 | Keeping assay/chemistry information improves transfer | Matched legacy-style label baseline vs repaired outcomes | No improvement on untouched supported endpoints; calibration worsens |
| H2 | ESM-derived latent representation improves conditional design | Matched sequence-encoder CVAE vs frozen-ESM CVAE | Only reconstruction improves; prior samples or independent evaluations do not |
| H3 | Biological conditioning changes useful outputs | Conditional AR vs unconditional AR; full vs masked conditions | Conditions ignored, memorization increases, or benefit exists only in the training reward model |
| H4 | Supported selection improves experimental yield | Same pool, fixed selectors, matched assay budget | No prospective benefit or diversity/feasibility losses negate benefit |

Failure is reportable. Never increase the model/search budget after seeing the
final test merely to obtain a positive result.

## 2. Competition scope and timing

The official website checked on 2026-09-21 lists October 1, 2026 AOE as the
deadline. It requests a 50,000-sequence library and a top-100 subset; 25 peptides
are sampled randomly for each advancing team. It limits peptide chemistry to
standard amino acids, linear chains, free termini, and lengths 8–50. Recheck
the current rules before release. These constraints govern the submission
adapter, not every possible future research dataset.

The approximately ten-day competition horizon is not a credible schedule for
completing the whole research program. Near-term scope is data feasibility,
isolated software, a bounded pilot if supported, and protecting a reproducible
incumbent submission. The larger program requires additional evaluation and
experimental collaboration. Challenge assays cannot be assumed available as
feedback before submission.

## 3. Current repository and reuse map

The user-reported SSH evidence snapshot is
`sweep_results/competition-evidence-v7`. It records byte-identical strict runs,
verified selection/predictor experiment markers, and no invalid evidence.
Those SSH artifacts have not been independently copied into this checkout.
The frozen selectivity experiment failed its predeclared criteria: retain C0.

| Existing component | Reuse | Limitation / adaptation |
|---|---|---|
| `tokenizer.py` | Residue vocabulary, BOS/EOS/PAD encoding | Keep vocabulary unchanged; conditions are continuous/prefix inputs |
| `model.py`, `generator.py` | Production AR baseline and sampling conventions | Do not change production checkpoint loading or defaults |
| `conditioning.py` | Charge, hydrophobicity, hydrophobic-moment descriptors | These are physicochemical controls, not measured biological outcomes |
| `scripts/audit_label_observations.py` | Parsing and observation provenance patterns | Verify every field against new contract; do not assume chemistry identity |
| `scripts/reconcile_observation_metadata.py` | Metadata joins and rejection accounting | Preserve unresolved cases explicitly |
| `scripts/build_molar_candidates.py` | Bounds-aware conversion precedent | Do not treat its binary labels as the new regression dataset |
| `generalization.py` | Family components, boundary audit, grouped bootstrap patterns | Add study-held-out protocol; old inspected tests are not untouched tests |
| `reward_benchmark.FrozenEncoder` | Pinned ESM inference and mean pooling | Cache by sequence/model revision/pooling; 35M features have dimension 480 |
| `scripts/experiment_utils.py` | Hashing, immutable run recipes, completion markers | Every new producer must seal `run.json` and all advertised outputs |
| `select.py`, readiness audits | Membership, validity and official novelty checks | Research selector gets a separate entry point |

New code belongs under `src/amp_challenge_2027/design_research/` with thin
`scripts/design_*.py` CLIs. Do not import script modules from package modules.
CLI code can reuse script-level experiment utilities. Avoid a general framework
rewrite; small explicit modules are sufficient.

### Protected state

Do not overwrite `checkpoint/generator`, `checkpoint/generator_blend`, deployed
reward heads, canonical processed datasets, previous experiment directories,
or `generate/competition-readiness-v1`. Do not change `uv run generate` defaults.
Preserve unrelated local edits in `pyproject.toml` and `uv.lock`.

All new real-data results use `sweep_results/design-research-v1/<stage>` and
research checkpoints stay inside those stage directories. Source data is
read-only. Each changed recipe uses a new stage directory/version.

## 4. Architecture choice

Implement the smallest defensible sequence of models:

1. Frozen-ESM linear and MLP endpoint predictors.
2. Unconditional AR and directly condition-controlled AR decoder controls.
3. A sequence-encoder conditional VAE control.
4. The same CVAE using a frozen ESM posterior representation.

Do not begin with latent diffusion, a new foundation model, RL, docking,
molecular dynamics, or several generator architectures. These are possible
later hypotheses after the controls identify a bottleneck.

### Correct use of ESM

ESM is a protein representation model, not a calibrated activity oracle or an
inverse decoder for arbitrary embedding vectors. Never concatenate a target
peptide's embedding into a reconstruction decoder and call reconstruction
accuracy evidence of de novo generation.

For target sequence x and condition c, use:

```text
Training:
  h = stop_gradient(mean_residue_pool(ESM(x)))        [B, 480]
  c = ConditionEncoder(request)                      [B, 128]
  q(z | h,c) = diagonal Gaussian posterior           [B, 64]
  p(z | c)   = diagonal Gaussian learned prior       [B, 64]
  z ~ q
  p(x | z,c) = autoregressive peptide decoder

De novo generation:
  c = ConditionEncoder(supported request)
  z ~ p(z | c)
  x ~ Decoder(z,c)
```

Generation must work without a target sequence or ESM embedding. The posterior
is used during training and separately labelled reconstruction diagnostics.
Analogue generation from a seed sequence is a distinct optional mode, excluded
from the initial study. The ESM encoder need not be loaded for prior sampling.

The learned prior is essential: adding Gaussian noise directly to arbitrary ESM
vectors is not the specified model and is not assumed to stay on a useful
peptide distribution.

## 5. Data contracts

Use UTF-8 JSONL plus JSON manifests for v1; use NumPy for dense arrays. Avoid a
new database or Arrow dependency until data size demonstrates the need. Every
record has `schema_version`, stable identifiers, and source provenance. JSON
uses null and explicit flags, never NaN or Infinity.

### 5.1 Molecule record (`molecules.jsonl`)

```text
molecule_id: sha256(canonical chemistry identity JSON)
sequence_id: sha256(exact canonical residue string)
sequence: canonical 20-AA string or null if not representable
raw_sequence: original source value
length: integer or null
n_terminus, c_terminus: explicit chemistry categories, including unknown
chirality, topology, intrachain_bonds, modifications: explicit source-backed values
chemistry_status: compatible | modified | unknown | malformed
source_record_ids: sorted list
eligibility_reasons: sorted list
```

Missing modification fields do not establish free termini. An adapter may use
a documented source-wide default only if it records the exact source schema
rule and version. Retain different chemistry identities even when sequences
match. Normalize harmless whitespace/case only when it cannot erase chirality
or modification notation. Never uppercase a source's lower-case D-residue
notation into an ordinary L-peptide without preserving and interpreting it.

### 5.2 Observation record (`observations.jsonl`)

```text
observation_id, molecule_id, study_id, source_record_id
source_url, source_file_sha256, source_locator, license_id
endpoint: mic | hc50 | hemolysis_percent | cytotoxicity | stability
organism_species, strain_id, resistance_annotations
assay_method, medium, ph, temperature, incubation_duration
rbc_species, cell_type                         # where applicable
replicate_group_id, technical_replicate_id, experimental_batch_id
raw_value, raw_unit, normalized_unit
lower, upper: endpoint bounds or null for an unbounded side
lower_inclusive, upper_inclusive: booleans
censoring: exact | interval | left | right
tested_concentration_um: number or null        # hemolysis_percent etc.
quality_flags, conversion_rule, extraction_status
```

All context fields are nullable; null means unknown. Missing endpoints are not
negative measurements. Source-level percentages and dose remain distinct.
`hc50` is permitted only for an explicitly identified half-hemolysis endpoint;
generic cytotoxic IC50 is not HC50. Human and nonhuman RBC observations stay
distinguishable.

Validate bounds by endpoint: MIC/HC50 concentrations must be positive; measured
hemolysis percentages may include zero and must lie in [0,100]. Stability needs
an explicit measurement type and unit and is retained without conversion into
a concentration target. Only supported concentration endpoints enter log2 NLL.

For supported unmodified chemistry:
`concentration_um = concentration_ug_ml * 1000 / molecular_weight_g_mol`.
Use the existing peptide mass implementation only after checking its termini
convention. Unknown chemistry or units cannot receive guessed conversions.
Keep inequalities and inclusive/exclusive bounds through every transformation.

### 5.3 Derived training examples

Predictor example: observation ID, molecule/sequence IDs, endpoint, context,
log2 concentration bounds, exact-value flag, and split/family/study IDs.

Generator example: molecule/sequence IDs, encoded condition, supporting
observation IDs, condition support status, sample weight, and split ID.

Do not join measurements from different chemistry identities. Paired endpoint
conditions need source-backed compatible contexts: same chemical molecule,
identified MIC organism, identified hemolysis target/assay, and a documented
pairing rule. Prefer within-study pairs. Cross-study pairs are separately
labelled and excluded from the primary joint-condition experiment.

Retain every raw observation. Deterministically group duplicated database
records referring to the same primary measurement so source duplication does
not multiply its weight. Distinct biological repeats remain distinct. During
training, normalize total weight per molecule and endpoint, then divide across
its eligible observations; add optional study weighting only as a predeclared
ablation. Missing study IDs are unknown, not distinct fabricated studies.

### 5.4 Data extension registry

`sources.json` lists ID, URL, retrieval timestamp, content hash, citation,
license/redistribution status, parser version, and original filenames.

First adapter: existing DBAASP raw/observation outputs. Second: DRAMP 4.0
experimental tables. Third: explicitly listed complete study supplements.
Do not write a broad autonomous crawler for v1. A source contributes only after
reporting compatible molecules, families, studies, endpoints, paired endpoints,
duplicates, unresolved chemistry and conversion failures.

Model-predicted labels and mined untested peptides cannot enter the experimental
label table. They may form a labelled-as-unmeasured pretraining dataset in a
later ablation. LLM extraction may propose records with source spans; numeric
conversion is deterministic and uncertain chemistry/endpoint mapping requires
review. Jev/Laya are not biological label generators.

## 6. Split design and leakage boundaries

Build splits before preprocessing that learns from data, embedding-dependent
fitting, supervised generation, model selection, or calibration. Frozen ESM
extraction itself is deterministic feature computation; access controls should
still keep final-test artifacts out of training jobs.

Maintain two separate protocols rather than pretending family and study
independence are equivalent:

* **Family protocol:** group exact sequences and transitive similarity
  components jointly across endpoints, chemistry records, and data sources.
  Primary identity cutoff is 0.8 using the repository's documented Levenshtein
  ratio. Here edges include equality (`>=0.8`); submission rejection remains
  `>0.8`. Record this distinction. Audit all cross-partition pairs exactly.
* **Study-transfer protocol:** reserve whole identified studies by a frozen
  rule; remove duplicate molecules and training near-neighbors of evaluation
  sequences. Report exclusions, retained support, and overlap with all prior
  experiments. Never claim both study and family independence without audits.

Use train/validation/calibration/test fractions 0.60/0.15/0.10/0.15 at group
level, allowing actual row fractions to differ. A component that prevents
adequate support is a failed feasibility gate, not permission to split it.
Blocking upper bounds on edit similarity may accelerate exact checks; verify
any pruning algorithm against exhaustive synthetic cases.

Audit generator training and any in-domain pretraining corpus against the same
held-out families. A generator pretrained on the entire current AMP corpus is
not a clean family-held-out experimental control. Train matched research
controls on permitted data; keep the deployed incumbent as a separately
identified historical reference. ESM pretraining overlap is not fully known;
disclose that limitation rather than claiming complete pretraining independence.

Known test labels from previous work are already inspected. QMAP can serve as
a standardized retrospective benchmark with its required overlap filtering,
but DBAASP-derived QMAP labels are not automatically independent of our data.
Freeze a new, exposure-audited study cohort for any future confirmatory claim.

### Feasibility defaults

These are engineering stop conditions, not sufficient sample sizes for a
scientific claim. Freeze them after the initial source inventory and before
fitting; report counts for each endpoint/context and partition.

* Endpoint training pilot: at least 100 molecules and 10 training families.
* A reported held-out endpoint: at least 30 molecules and 5 families; otherwise
  label its metrics exploratory/insufficient support.
* A jointly requested MIC/HC50 condition: at least 30 training molecules,
  10 families, and 3 identified studies supporting that actual combination.
* An endpoint with only right-censored observations cannot identify its location
  and scale; disable fitting that endpoint rather than claim learned potency.
  Apply the same rule to entirely left-censored data. Mixed censoring still
  requires an identifiability/support check; it is not automatically sufficient.
* Calibration methods must state endpoint-specific support requirements. When
  unavailable, expose `calibration_status=insufficient_support`.

If HC50 or joint support fails, continue an activity-only research pilot and
disable joint-selectivity claims/requests. Software work and fixture tests can
continue independently of biological training feasibility.

## 7. Frozen ESM cache

Use `facebook/esm2_t12_35M_UR50D`, exact resolved revision pinned in the run
protocol, final-layer mean of residue embeddings, excluding BOS/EOS/PAD.
Call `eval()`, `inference_mode()`, FP32, and disable gradients. The v1 hidden
size is 480; obtain it from the backbone config and validate the cache shape.
Do not accidentally reuse the 320-dimensional ESM-8M reference cache.

Cache identity includes model/revision, tokenizer revision, layer, pooling,
dtype, software versions, sequence bytes/order, and relevant runtime. Deduplicate
sequence computation without merging chemistry-labelled observations. Save:

```text
run.json
sequence_ids.json
features.npy                    # float 32 [N,480]
backbone.json
chunks/<index>/...              # optional resume granularity
complete.json
```

Validate finite values, IDs/order, feature dimension and hashes before reuse.
Do not impose cross-device bitwise identity; record runtime and compare numeric
tolerances in portability tests. Record logical device and
`CUDA_VISIBLE_DEVICES`; physical GPU 1 appears as logical cuda:0.

Train-only feature centering/scaling can be saved as a separate transform.
Do not standardize on validation, calibration, test, or the generated pool.

## 8. Endpoint predictor specification

Implement `OutcomePredictor(features, context, endpoint)` with matched linear
and MLP variants. Output Gaussian `mu` and positive `sigma` for log2 micromolar
concentrations. Start with diagonal marginal endpoint predictions; do not claim
a learned joint MIC/HC50 distribution when paired data is insufficient.

Context uses training-only vocabularies for species, strain, assay/RBC category,
and explicit unknown tokens. Rare strain IDs back off to species with a flag;
unseen conditions trigger `out_of_support`, not silent confidence. Numerical
context uses observed-value masks and training-only transforms.

Initial MLP: input -> 256 -> 128 -> 2, GELU, dropout 0.1, with endpoint/context
embeddings. Linear baseline uses the same features/context and loss budget.
Ensemble seeds 42/43/44 are retained rather than selecting the best seed.

### Censored likelihood

Let y be log2 concentration and F/f the predicted Gaussian CDF/density.

```text
exact y:       loss = -log f(y)
interval l,u: loss = -log(F(u) - F(l))
right bound l:loss = -log(1 - F(l))
left bound u: loss = -log F(u)
```

Use stable log-CDF/log-survival and log-difference implementations (e.g.
`torch.special.log_ndtr` with a stable logdiffexp). Do not subtract nearly equal
floating CDF values directly. Validate strict positivity of concentrations,
ordered bounds, and the distinction between exact and finite-interval records.
Zero-width exact records use the density case. Set a documented minimum sigma
such as 0.1 log2 units as numerical regularization, not measured assay precision.
Known dilution brackets should remain intervals rather than false exact values.

Hemolysis-at-dose records are preserved but excluded from HC50 v1 training.
An optional later monotonic dose-response head needs its own endpoints,
identifiability study and tests; it must not manufacture HC50 from isolated
percent-lysis rows.

Early stopping uses validation censored NLL. Fit a positive predictive scale
factor on the separate calibration partition by its censored likelihood when
supported. Report exact-target MAE/RMSE only where exact targets exist, censored
NLL on all supported rows, calibration/coverage diagnostics with censoring
limitations, and threshold-event Brier scores only for observed determinate
events. Compare against training-only constant distribution baselines.

Return per candidate: endpoint means/scales, ensemble disagreement, training
similarity, support counts, missing-context flags, calibration status, and
`in_support`. Uncertainty is empirical and does not guarantee OOD coverage.

## 9. Biological condition schema

Separate this from existing physicochemical conditioning. Define a versioned
`DesignCondition` containing:

```text
species_id; strain_id or UNKNOWN; assay_context_id or UNKNOWN
mic_event: UNKNOWN | MASKED | LE_16_UM | GT_16_UM
hc50_event: UNKNOWN | MASKED | GE_128_UM | LT_128_UM
hemolysis_context_id or UNKNOWN
length_bin: 8_15 | 16_25 | 26_35 | 36_50 | MASKED
support_profile_id
```

`support_profile_id` is provenance used by the support checker; it is not an
input feature or learned embedding. Never encode record, family, study, split,
or artifact IDs as shortcuts to the target outcome.

The thresholds are initial research conditions tied to the challenge context,
not definitions of clinical efficacy or safety. Make them protocol fields, not
buried constants. A measured bound yields an event only if its entire admissible
range implies the event; crossing the threshold yields UNKNOWN. Include tests
for every inclusive/exclusive boundary. UNKNOWN represents unavailable evidence;
MASKED represents intentionally withheld conditioning. Unsupported categorical
values fail, and unknown metadata is not replaced by a desired condition.

Build train/validation condition examples only from eligible observations;
never from the same predictor's inferred labels in the primary experiment.
Sampling desired conditions is a request to a model, not an assertion that the
output achieves them. Enforce the support gate before generation. Do not infer
an MDR condition from a species name.

Keep molecule total sampling mass stable when it has multiple eligible condition
records. For each sequence occurrence, select one compatible record with seeded
sampling; do not create every possible Cartesian product of observations.

## 10. ESM conditional VAE and controls

New independent model class `ConditionalPeptideVAE`; no changes to production
checkpoint schema. Initial architecture:

| Component | Specification |
|---|---|
| Condition encoder | Categorical embeddings + MLP to 128 dimensions |
| Posterior | ESM 480 + condition 128 -> MLP 256 -> mu/logvar 64 |
| Prior | condition 128 -> MLP 128 -> mu/logvar 64 |
| Latent | 64 dimensions; analytic Gaussian KL |
| Prefix | concatenate z (64) + c (128) -> projection -> four 256-dimensional tokens |
| Decoder | 4 causal Transformer layers, width 256, 8 heads, FFN 1024, dropout 0.1 |
| Vocabulary | Existing tokenizer IDs, 24 tokens; emissions restricted to residues/EOS |
| Positions | At least 4 prefix tokens + 1 BOS + maximum 50 residues |

The decoder returns logits aligned only to sequence input positions. Prefix
positions have no token target/loss. Teacher input is BOS+residues; targets are
residues+EOS, padding ignored. Causal and padding masks must prevent future
residue leakage. Maximum length includes residues only, not prefix/BOS/EOS.

Loss per molecule/example:

```text
L = token_mean_cross_entropy + beta(step) * mean_latent_dimension_KL(q || p)
```

Both reductions are explicit to avoid scaling beta with sequence length/latent
dimension. Start beta at zero and linearly reach 1 over 30% of planned optimizer
steps, then keep it fixed. Log unnormalized KL and per-dimension KL as well.
Clamp log-variance to a declared numerical range, e.g. [-8, 4]. Reparameterize
using a passed Torch generator. Biological condition dropout 0.15 uses MASKED,
consistently in posterior, prior and decoder. It does not relabel a peptide.

V1 has no predictor reward, latent-property penalty, classifier-free guidance,
or reconstruction-from-generated-ESM loss. This isolates whether the basic
representation helps and avoids giving the model a shortcut into the evaluator.

### Required controls

* `ar_unconditional`: same decoder, learned neutral prefix, no biological labels.
* `ar_conditioned`: same decoder, c-derived prefix, no latent posterior.
* `cvae_sequence`: same prior/latent/decoder; a small bidirectional sequence
  encoder produces the 480-dimensional posterior input instead of frozen ESM.
* `cvae_esm`: proposed frozen-ESM posterior.
* Incumbent: historical operational reference; overlap caveats are reported.

Use identical permitted sequence sets, conditions, sampling attempts, seeds,
selection recipes and training-step limits across trainable controls. Report
parameter counts, encoding cost, training time, and total GPU-hours; equal
steps do not imply equal compute. A condition-shuffled training control is a
predeclared optional ablation, never introduced after final-test failure.

### Posterior-collapse diagnostics

Report active latent dimensions, prior/posterior discrepancy, reconstruction
with sampled/zero/shuffled z, and prior-sample diversity. Compare held-fixed z
with changed conditions. Decoder reconstruction accuracy alone cannot promote
the model. If z has no effect, record H2 unsupported; do not silently claim the
CVAE is better than conditional AR. Any free-bits or alternative schedule is a
new validation-stage ablation with a changed recipe, not an automatic rescue.

## 11. Training and sampling budget

Defaults below are a bounded starting protocol, not optimized hyperparameters.
An implementation should support overrides and hash them into the recipe.

* Cache ESM once; FP32 inference batch 64, explicitly pinned revision.
* Predictor heads: AdamW lr 1e-3, weight_decay 1e-4, batch 128, maximum 100 epochs,
  patience 10; seeds 42/43/44. Checkpoint by validation NLL.
* Generator feasibility smoke: synthetic data, <=200 updates, CPU, no downloads.
* First real generator pilot: batch 64, lr 3e-4, weight_decay 0.01, gradient_clip 1,
  FP32, maximum 20 epochs, patience 5, training seed 42.
* Choose protocols on validation diagnostics; matched final runs use all three
  predeclared seeds, maximum 50 epochs, patience 8, and a fixed sampling recipe.
* AdamW optimizer and all RNG states are resumable within an identical run
  recipe. Different effective batch/precision/device settings create a new run.

Pilot each architecture with 2,048 raw draws under the same request set.
Final research comparison uses a frozen 10,000 raw-draw budget per model/seed;
report valid/unique/novel/support-qualified yields per raw draw. A later
submission-scale run may target 50,000 accepted sequences with a fixed attempt
cap and explicit accounting. Do not let one model receive unlimited rejection
sampling while another receives a fixed budget.

Sampling: temperature 1.0, top_p 0.9, no repetition penalty in the matched v1
research comparison; mask nonresidue specials, delay EOS until 8 residues,
stop after 50 residues. Reject invalid sequences visibly. Use an explicit RNG
for prior latent samples, token draws, condition allocation, and candidate
ordering. Record batch size; do not promise batch-size-invariant RNG behavior.

### Export contract

```text
research_model_config.json      # kind, architecture, latent/prefix sizes, tokenizer
condition_schema.json
condition_support.json
decoder.pt, prior.pt, condition_encoder.pt
posterior.pt                   # training/reconstruction artifact, not needed to sample
training_summary.json
run.json, complete.json
```

Sample through `scripts/design_sample.py`, not the production entry point.
The independent release smoke must generate with posterior/ESM absent from its
load path. Reject attempts to load research checkpoints with production loader.

## 12. Candidate prediction and portfolio selection

Freeze the predictor before generator/selector comparisons. Predictions must
retain row identity, endpoint/context identity, model hashes, calibration and
support fields. Descriptor values are separate from biological measurements.

V1 endpoint utility uses a declared species/context, initially a sufficiently
supported MIC event. Compute mean ensemble probability of MIC<=16. If selectivity
data is supported, impose a separately declared marginal HC50 criterion. Do not
multiply marginal event probabilities and call the product a joint probability.
A joint event/ratio estimator requires supported dependence modelling or an
explicitly labelled assumption and sensitivity analysis. Never turn the existing
binary HemoScorer score into HC50 or a clinical safety window.

Research selectors:

* `S0`: activity-only utility on supported candidates.
* `S1`: rank by the lower-quantile ensemble activity probability, with
  diversity limits and an optional supported hemolysis constraint.

For S1 initial uncertainty statistic use the 10th percentile across ensemble
event probabilities; label it an ensemble summary, not a calibrated confidence
bound. Coarse ensembles may provide weak uncertainty discrimination; report it.
Use stable sorting and deterministic greedy acceptance with a protocol-defined
family cap (initial research choice: at most 5 of 100 from one sequence cluster).
Define clusters using the same transitive sequence-family algorithm at >=0.8
within the frozen candidate pool, with stable cluster IDs. Break score ties by
sequence bytes. The optional HC50 constraint must specify its probability
threshold in the frozen protocol; if absent or unsupported, disable that
constraint and label the result activity-only. Do not silently tune it until
100 candidates pass. S0 uses ensemble mean activity probability and the same
support/validity/novelty gates, without the S1 family cap or HC50 constraint.
Apply identical validity/reference-novelty conditions. Log counts and rejection
reasons at every stage. An underfilled top 100 is a failed candidate protocol;
do not silently relax thresholds or invent missing predictions.

Decouple the 50,000 library quality assessment from top 100 utility. Report
sequence diversity, novelty, property distributions and configured seqme metrics
for the full library. Do not claim a better shortlist establishes better phase-1
computational standing; organizer weighting may not be completely specified.

Simulate random subsets of 25 from each top 100 to describe fixed surrogate-score
variation, including worst-performing subsets and shared-family concentration.
This is not experimental power, true biological success probability, or an
estimate of wet-lab selection odds. A random 25 has the same expected average
as the full 100 under uniform sampling; reordering cannot hide weak candidates.

## 13. Evaluation and decision gates

### Software gate

All contracts, leakage tests, numerical tests, manifest tests and fixture
end-to-end tests pass. Every output is linked to its producing recipe. No public
network/GPU is needed for the standard test suite.

### Retrospective predictor gate

Report matched linear/MLP comparisons on validation, then one sealed test run.
Use molecule/family or study grouped paired intervals, not row-level bootstrap
that counts repeated observations as independent. Define a primary endpoint and
metric before training. Proposed first primary metric: supported MIC censored
NLL against a training-only constant baseline and a matched simple linear model.
Report event Brier/MAE and calibration as secondary measures, with support counts.
If interval evidence is inconclusive or calibration fails, no performance win
is declared. Merely reducing validation loss cannot replace final evaluation.

### Generator gate

Report raw and postselection validity, uniqueness, train/reference similarity,
novel-family coverage, condition sensitivity, prior-sample quality and runtime.
Evaluate all required controls and all seeds. Reconstruction quality and higher
scores from the frozen optimization predictor remain proxy findings. Independent
prediction tools can be sensitivity checks but are not biological validation
and may share training data. No automatic production promotion.

### Prospective gate

Design a separate preregistered, blinded comparison with an experimental partner:
equal total candidate and synthesis/assay budgets, matched controls, complete
failed/inactive results, specified endpoints and censoring, and justified sample
size. Distinguish a predictive hit-rate study from mechanistic/clinical work.
The challenge's 25 peptides are one pipeline evaluation; they do not establish
causal superiority of every module. Do not request or run wet-lab procedures as
part of this coding implementation.

### Application gate

Choose an intended setting with the experimental collaborator. Assess relevant
host-cell toxicity, stability, activity in the intended environment, synthesis,
solubility, and formulation feasibility. Do not add hard computational stability
filters without validated data. Formulation and modified-chemistry research is
a later program separate from the unmodified submission.

## 14. Artifacts and implementation interfaces

Proposed package modules and public surfaces (names may change only with an
explicit update to this plan and all consumers):

```python
# schemas.py / measurements.py
validate_molecule(record: dict) -> dict
validate_observation(record: dict) -> dict
to_log_concentration_bounds(observation: dict) -> dict
classify_threshold_event(bounds: dict, threshold: float, direction: str) -> str

# dataset.py / splits.py / conditions.py
build_dataset(sources: list[dict], policy: dict) -> dict
make_family_split(molecules: list[dict], policy: dict) -> dict
make_study_split(molecules: list[dict], observations: list[dict], policy: dict) -> dict
build_conditions(observations: list[dict], splits: dict, policy: dict) -> dict
check_request_support(request: dict, support: dict) -> dict

# embeddings.py
encode_sequences(sequences: list[str], *, model: str, revision: str,
                 device: str, batch_size: int) -> tuple  # ndarray, metadata
load_feature_cache(path, expected_identity: dict) -> tuple

# likelihoods.py / outcome_model.py
censored_gaussian_nll(mu, sigma, lower, upper, exact_mask, censor_codes) -> Tensor
class OutcomePredictor: ...

# conditional_model.py
class ConditionEncoder: ...
class ResearchDecoder: ...
class ConditionalPeptideVAE:
    def posterior(self, features, condition): ...
    def prior(self, condition): ...
    def forward(self, input_ids, features, condition, *, generator): ...
    def sample_prior(self, condition, *, generator): ...

# sampling.py / selection.py / reporting.py
sample_prior_sequences(model, conditions, *, n, seed, batch_size, recipe) -> dict
select_portfolio(candidates: list[dict], policy: dict) -> dict
compare_runs(run_paths: list, protocol: dict) -> dict
```

Each stage writes `run.json` at creation, resumable partial files where needed,
then `complete.json` only after all advertised outputs validate. Completion
markers include hashes for `run.json` and every declared result file. A stage
registry defines expected outputs for producer, tests, and inventory adapter
from one location, avoiding the prior filename/hash mismatches.

New statuses: `complete`, `incomplete`, `ineligible_data`, `failed_validation`.
Do not equate a completed computation with eligibility for deployment. CLI exit
codes: 0 success, 2 invalid input/ineligible data, 3 intentionally incomplete budget,
1 unexpected operational error. Clarify status in a machine-readable report.

`--list` validates metadata and describes proposed work without output creation,
downloads or GPU initialization. `--dry-run` may create fixture reports only
when explicitly requested; do not confuse the two. Never silently acquire a new
backbone revision. Missing revision/cache produces actionable instructions.

The final `design_prepare.py` is an orchestrator of package functions from
T02–T04. Its output root contains immutable `observations/`, `splits/`, and
`conditions/` substages, each with its own registry and manifests. A root
`data_manifest.json` records these relative paths/hashes; root completion is
written only after all three stages validate. Every downstream `--data` in the
workflow means this completed bundle. Training resolves splits and conditions
through that manifest, never recursive file discovery. Standalone split and
condition CLIs write new external outputs; they do not mutate a completed data
bundle. During T02/T03 development, explicitly mark the bundle incomplete until
the later required stages exist.

## 15. Required tests

Use tiny deterministic synthetic sequences and fake embeddings. Real model
downloads and GPU inference belong in opt-in integration tests.

1. Chemistry: free/amidated/unknown termini stay distinct; stereo notation is
   preserved; modified molecules cannot acquire canonical labels by cleanup.
2. Measurements: exact, left/right/interval bounds; unit conversion; boundary
   inclusivity; unresolved units; hemolysis percent is never treated as HC50.
3. Deduplication: cross-database copies do not multiply observation weights;
   distinct biological repeats survive; input reordering does not change IDs.
4. Splits: transitive family bridge; repeated molecule across endpoints;
   study holdout; near-neighbor pretraining overlap; giant-component failure.
5. Cache: pooling excludes specials/padding; shape 480 vs 320; same sequence order
   on resume; corruption, changed revision and wrong pooling rejected.
6. Likelihood: exact result against Gaussian reference, analytic one-sided cases,
   extreme tails, narrow intervals, finite gradients, larger valid intervals
   never having lower probability than their subsets.
7. Conditions: event derivation at exact/censored thresholds; unsupported joint
   request failure; known/unknown/masked distinction; no validation/test support
   counted toward training request support.
8. Decoder: no future-token leakage; prefix alignment; PAD loss excluded;
   residue length constraints; EOS handling; prefix gradients nonzero.
9. CVAE: KL zero for identical Gaussians, positive for shifted distributions;
   posterior gradients flow while ESM frozen; prior sampling never requests x;
   latent noise driven by explicit RNG; fixed recipe reproducible.
10. Sampling/selection: no reserved-token emissions; stable identity under
    caching; honest rejection counts; family cap; underfilled list fails;
    uncertain candidates cannot be labelled measured or safe.
11. Release: exported generator samples with no posterior/ESM load; manifests
    survive inventory round-trip; damaged file fails; research exports cannot
    replace/read as production checkpoints.
12. End-to-end fixture: source observations -> splits -> fake ESM -> both outcome
    baselines -> generator smoke -> samples -> selection -> full report. Ensure
    all protected files retain their original hashes.

Do not test that a real neural model always beats a baseline. Tests verify
scientific bookkeeping and software behavior; experiments test hypotheses.

## 16. Planned CLI workflow (not implemented yet)

These commands are the target interface for implementation. Do not give them
to an SSH user as runnable commands until the corresponding task is merged.
All protocol files listed here are to be created by the implementation.

```bash
uv run --no-sync python scripts/design_prepare.py \
  --protocol experiments/design_research_v1.json --list
uv run --no-sync python scripts/design_prepare.py \
  --protocol experiments/design_research_v1.json \
  --out sweep_results/design-research-v1/data

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/design_cache_esm.py \
  --data sweep_results/design-research-v1/data --split train validation calibration \
  --device cuda --batch-size 64 --out sweep_results/design-research-v1/features

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/design_train_outcomes.py \
  --data sweep_results/design-research-v1/data \
  --features sweep_results/design-research-v1/features \
  --protocol experiments/design_research_v1.json --device cuda \
  --out sweep_results/design-research-v1/outcomes

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/design_train_generator.py \
  --data sweep_results/design-research-v1/data \
  --features sweep_results/design-research-v1/features \
  --protocol experiments/design_research_v1.json --architecture cvae_esm \
  --seed 42 --device cuda --out sweep_results/design-research-v1/cvae-esm-seed42

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/design_sample.py \
  --model sweep_results/design-research-v1/cvae-esm-seed42 \
  --requests experiments/design_requests_v1.json --n 2048 --seed 42 --device cuda \
  --out sweep_results/design-research-v1/pilot-samples
```

Outcome test evaluation, candidate scoring, selection and release commands are
specified in the companion task queue. Test access is a separate explicit stage
after `seal.json` binds checkpoint/protocol hashes and model-selection decisions.
Do not embed an automatic final-test evaluation in training or sampling.

## 17. Deliverable and stopping criteria

The coding deliverable is complete when all task-queue software acceptance
checks pass, documented CLIs work on fixtures, a bounded real-data smoke is
recorded if compatible data exists, and there is a reproducible comparison
report generator. Lack of compatible data yields an honest feasibility report,
not invented labels or forced model promotion.

Scientific success is separate: it needs the declared comparisons and eventually
prospective evidence. A null result should still yield reusable audited data,
source adapters, benchmark artifacts and a clear account of what did not work.

## 18. Sources and methodological boundaries

* [Official challenge](https://szczurek-lab.github.io/amp-challenge-website/)
  and [Kaggle rules](https://www.kaggle.com/competitions/amp-challenge/rules):
  participation and release constraints; check again at submission.
* [ESM official documentation](https://github.com/facebookresearch/esm): frozen
  sequence/per-residue representations; these do not confer activity labels.
* [HydrAMP, 2023](https://www.nature.com/articles/s41467-023-36994-z): prior
  conditional-VAE peptide design. Our architecture must be compared with this
  precedent; CVAE conditioning itself is not a new contribution.
* [ApexGO, 2026](https://www.nature.com/articles/s42256-026-01237-5): generative
  peptide optimization with experimental validation; establishes a relevant
  comparison point, not an expected success rate for our model.
* [QMAP, 2026](https://www.nature.com/articles/s41598-026-56004-8): predefined
  homology-aware MIC/HC50 evaluation, with overlap controls for external data.
* [DRAMP 4.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701585/): candidate source
  of experimental endpoint/chemistry/stability annotations, subject to audit.
* [Negative-data bias study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9487607/):
  supports separating measured negatives from assumed non-AMP sequences.

The choices of latent size, decoder size, support thresholds, training budget,
condition thresholds, and portfolio policy above are proposed experimental
defaults. They are not literature-established optima or promises of efficacy.
