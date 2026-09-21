# Next experiment decision

Status: **pending remote evidence review**.

## Evidence required before choosing

1. Strict generation completes with the default incumbent recipe, required
   scorers and byte-stable outputs.
2. The current family-held-out benchmark and label/metadata audits are present
   and their completion hashes verify.
3. The full top-100 report identifies missing score coverage, close reference
   neighbors, and any supported score trade-offs.
4. Existing frozen-library comparisons are inventoried to avoid duplicate
   generation or embedding computation.

## Decision rule

- Material label or observation defects: prioritize the existing controlled
  label-ablation/reconstruction work.
- Weak or uncertain family-held-out prediction: prioritize data and predictor
  validation; do not optimize selector weights against weak surrogates.
- Credible scores with suitable candidates missed by the incumbent: run one
  predeclared selector comparison on frozen libraries.
- Credible evaluation with no adequate candidates: propose a separate generator
  or data experiment.
- Inconclusive or small differences: retain the incumbent and document limits.

No branch is authorized by this template alone. After evidence review, replace
this status with one selected hypothesis, exact source artifacts/hashes,
protocol, minimum differentiating comparison, criteria fixed before outcomes,
stop rules and a statement of what result would change the decision. Keep the
already inspected test excluded from future confirmation claims.
