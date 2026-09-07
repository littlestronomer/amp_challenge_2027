# Paired top-100 comparison: incumbent vs epoch-58 75/25 blend

Run on the SSH machine, from the existing repository. This is the next gate
after confirming `p3_s1` on generation seeds 42, 43 and 44. No training,
generation, new mixture search, or promotion occurs. The six source libraries
remain read-only, and each output `library.fasta` is a byte-identical copy.

## Run

Keep the environment used for the completed sweeps; no new dependencies.
The three per-artifact classifier configs already checked on the SSH machine
must remain beside their actual frozen weights. Do not overwrite them with the
stale shared `config.json`, or reconstruct calibration from rounded prose.

```bash
git pull --ff-only origin main

uv run --no-sync python scripts/compare_top100.py \
  --out sweep_results/epoch58-top100-v1 --list

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -u scripts/compare_top100.py \
  --out sweep_results/epoch58-top100-v1
```

Use a free GPU and `tmux` if SSH may disconnect. `--list` checks source
manifests, output hashes and classifier metadata without loading models or
writing. The actual run validates complete head tensors and refuses missing
models, incomplete/nonfinite scores, failed novelty checks, or fewer than 100
eligible selections. CUDA nondeterminism raises rather than being ignored.

Default source pairs:

| Generation seed | Incumbent `hybrid` source | Candidate `p3_s1` source |
|---|---|---|
| 42 | `sweep_results/checkpoints-seed44-v2` | `sweep_results/epoch58-blends-v1` |
| 43, 44 | `sweep_results/checkpoint58-confirm-v1` | `sweep_results/epoch58-blend-confirm-v1` |

The source FASTA is `<source>/<case>/seed<seed>/library.fasta`. Path overrides
are `--checkpoint-screen`, `--checkpoint-confirm`, `--blend-screen`, and
`--blend-confirm`. `--seeds 43 44` is supported but requires its own output
directory; the default runs all six libraries.

## Fixed comparison protocol

- Activity, property-conformity proxy, reference-neighbor precision proxy,
  panel breadth and MDR breadth weights are `0.5, 0.25, 0.25, 1, 1`: the
  current production ranking recipe. No weights are tuned in this experiment.
- The 35M activity and panel heads use their per-artifact temperatures.
  A panel forward pass is shared for breadth and MDR readouts. Genus labels
  and their order must exactly match the recipe.
- The precision *ranking proxy* uses the production 8M embedder, not the 650M
  library evaluator. No new FBD/MMD measurements are needed because library
  membership and bytes are unchanged. An 8M ranking proxy is not an independent
  validation of the 650M results.
- The conformity reference subsample (12,000) and selection RNG use a fixed
  seed 42 for every cell, independently of the library generation seed.
- Existing plausibility rules, a score-ranked shortlist of at most 2,000,
  and the existing score/diversity selector are retained. Insufficient novel
  survivors fail, rather than silently extending/relaxing the protocol.
- Top sequences must be unique members of their own library, and all must
  have maximum Levenshtein ratio <= 0.8 against the full reference. There is
  no approximate similarity fallback in this experiment.
- Hemolysis is scored **after selection**, with weight zero. It is not used
  to choose top candidates or to tune a threshold. `hemo_risk_*` reports
  predicted risk, not observed hemolysis or demonstrated safety.
- Z-normalization is performed separately across each entire 50k library,
  matching production. Composite scores are not comparable probabilities
  across libraries. Compare the raw component and audit readouts instead.

Scorers load once. Reference embeddings are stored under the NEW output root,
not the legacy shared `data/cache`. Source FASTAs, configs, weights, source
provenance, code/runtime, and resolved HF model revisions are recorded.
The unused language-model head/pooler loading warnings are expected for this
architecture: the classifier averages `last_hidden_state`, not pooler output.
The unauthenticated Hub warning does not invalidate successful inference.

## Read results

```bash
uv run --no-sync python -c '
import pandas as pd
root = "sweep_results/epoch58-top100-v1"
df = pd.read_csv(f"{root}/results.csv")
cols = [
    "case", "seed", "activity_mean", "panel_mean_probability",
    "breadth_mean", "mdr_mean", "hemo_risk_mean", "hemo_risk_p75",
    "hemo_risk_max", "top_mean_pairwise_distance", "reference_similarity_max",
]
print(df[cols].sort_values(["seed", "case"]).to_string(index=False))
print("\nPAIRED DELTAS: candidate minus incumbent")
print(pd.read_csv(f"{root}/paired_deltas.csv").to_string(index=False))
'
```

`results.csv` updates after each completed cell. `paired_deltas.csv` updates
once both members of a pair are complete. Positive hemolysis-risk deltas are
worse; positive activity/panel readouts are higher surrogate scores. The
`top_mean_pairwise_distance` statistic is mean `1 - Levenshtein.ratio` over
all distinct top pairs, **not** seqme's library Diversity metric. The
`conformity_mean` and `precision_mean` columns are ranking proxies, **not**
the similarly named official library metrics.

Each `<case>/seed<seed>/` contains `top.fasta`, `top_scores.csv` (per-sequence
scores, panel probabilities, properties and reference similarity),
`summary.json`, `scores.npz` and completion/hash markers. Keep these with the
source runs and the three real classifier configs.

Rerun the identical command to resume: verified completed cells are skipped;
completed library scoring is reused if a later selection/audit stage failed.
An interrupted score write without a completion marker is recomputed.
Changing settings, source files, classifier artifacts, code or runtime
requires a new output directory. Do not run concurrent writers or edit
manifests to force reuse. Fully completed reruns do not redo model inference;
to test numerical reproducibility itself, use a fresh output directory on
the same runtime/hardware and compare `top.fasta` hashes.

The new code does not alter old experiments. Their exact-code resume guards
may reject rerunning an older generation command after this pull; that is
not a reason to regenerate or delete the completed source libraries.

## Decision boundary

Activity/panel scores are selection surrogates, so improvement in them is not
independent potency evidence. The separately trained hemolysis head is also
a surrogate with possible shared-data/representation bias. The smoke test
and matching configs establish technical compatibility, not recovery of the
training validation AUROC, calibration quality, or label provenance.

Compare paired results across all three generation seeds and inspect any
activity/risk/diversity trade-offs. Three generation seeds are not independent
training replicates. Keep defaults unchanged pending review, broader potency
and synthesizability assessment, and reproducible entry-point packaging.
