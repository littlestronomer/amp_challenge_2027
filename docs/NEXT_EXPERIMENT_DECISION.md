# Next experiment decision

Status: **incumbent retained; one exploratory frozen-pool sensitivity study is
predeclared.**

The normalization comparison did not show a compelling reason to change the
ranker. The family-held-out activity, panel and hemolysis architecture comparisons
report no clear MLP advantage, with paired intervals including zero. The
reconstructed deployed-head evidence has material overlap and unresolved data
provenance. Do not retrain on the inspected test results or infer wet-lab success
from model scores.

## Authorized bounded experiment

Run C0 (incumbent replay) versus R1 (one fixed hemolysis-risk penalty) on the
existing hybrid libraries for generation seeds 42, 43 and 44. Use the protocol in
`experiments/selectivity_tradeoff_v1.json` exactly: fixed ranking seed 42, 2,000
candidate shortlist, 100 selections, current plausibility and inclusive 0.8
reference-similarity ceiling, and lambda 0.25 standardized once on all seed-42
risk predictions. Do not regenerate libraries, hand-replace boundary sequences,
or tune the penalty after observing results.

This experiment is descriptive of selector behavior and availability of lower
predicted-risk members in the frozen pool. It does not establish that the risk
head generalizes, that activity is preserved biologically, or that candidates
will pass synthesis or wet-lab tests. A pass only makes the proposal a candidate
for independent evaluation; it never promotes the selector automatically.

## Stop and decision rules

- Stop on a source/model/reference/protocol identity mismatch, failed source-top
  parity, corrupt cache chunk, incomplete risk coverage, failed C0 byte replay,
  or fewer than 100 eligible candidates.
- Report every seed and predeclared metric. Require all per-seed criteria in the
  protocol to pass before proposing independent evaluation.
- If criteria fail or the effect is small, retain C0 and record the result. Do not
  expand the weight search to find a favorable setting.
- Predictor data repair remains the next scientific priority. Follow
  `docs/PLAN_PREDICTOR_DATA_REPAIR.md` before requesting a separate training plan.

The inspected reconstruction/family benchmark results remain historical
descriptive evidence and cannot serve as untouched confirmation data.
