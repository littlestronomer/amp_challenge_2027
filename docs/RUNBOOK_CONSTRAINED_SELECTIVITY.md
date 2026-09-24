# Repaired constrained-selection experiment

The earlier solver reused a dense risk row for its first pair constraint.
It mistakenly constrained roughly `sum(risk*x) + x_i + x_j <= 1` instead of
`x_i + x_j <= 1`. Therefore **all previous infeasibility conclusions are
withdrawn**, including ceilings 0.25, 0.50 and 0.75. The verified incumbent
fallback established only baseline feasibility. It did not establish that
reranking cannot help, and it did not demonstrate a lower-risk set.

Version `pair-cuts-v2` uses empty rows for every pair, independently verifies
all numerical constraints, and distinguishes timeout and cut-limit exhaustion
from proven infeasibility. The report shows `milp` versus `incumbent_fallback`
and the risk delta explicitly. Tests execute a genuine second MILP solve with
conflict cuts. No model weights or default production selection are changed.

## SSH pilot

Run from the repository root after pulling. Keep old results for diagnosis;
reuse the intact pool and eligibility artifacts without recomputing neural scores.
The pilot switches do not edit the frozen protocol.

```bash
uv run --no-sync python -c 'import pandas, scipy; from scipy.optimize import milp; print(pandas.__version__, scipy.__version__)'
uv run --no-sync python -m pytest -q \
  tests/test_constrained_selectivity.py \
  tests/test_selectivity_helpers.py tests/test_selectivity_risk_cache.py \
  tests/test_authorship_readiness.py

uv run --no-sync python scripts/selectivity_solve.py \
  --pools sweep_results/constrained-selectivity-v1/pools \
  --eligibility sweep_results/constrained-selectivity-v1/eligibility \
  --protocol experiments/constrained_selectivity_v1.json \
  --reference data/antibacterial.fasta \
  --seeds 42 --time-limit 600 --max-cut-rounds 20 \
  --out sweep_results/constrained-selectivity-repaired-v1/solutions

uv run --no-sync python scripts/selectivity_report.py \
  --pools sweep_results/constrained-selectivity-v1/pools \
  --solutions sweep_results/constrained-selectivity-repaired-v1/solutions \
  --protocol experiments/constrained_selectivity_v1.json \
  --out sweep_results/constrained-selectivity-repaired-v1/report

cat sweep_results/constrained-selectivity-repaired-v1/report/REPORT.md
```

This is CPU-only; GPU 1 is not needed. The time limit is now a **shared budget
per seed/profile**, not a fresh 600 seconds for every cut round. Exact validation
may add overhead. Four profiles imply about 40 minutes of solver time at most,
plus validation/I/O. To diagnose only the unrestricted profile, add
`--risk-ceilings 1.0` and use a different output directory.

Pandas and SciPy are required and already present on the reported SSH machine.
Do not use a bare `uv sync` in the active experiment environment just to run
these tests; it can remove optional dependencies. The release validator creates
a separate environment to test the plain install.

## Reading the result

- `optimal` + `milp`: solved within recorded MIP tolerance and independently
  validated. Inspect the risk delta and each constraint margin.
- `feasible_unproven_optimal` + `milp`: valid selection, optimality not established.
- `incumbent_fallback`: old top 100 recovered; no improvement demonstrated.
- `unknown_timeout` / `unknown_cut_limit`: unresolved; not proof of impossibility.
- `infeasible`: solver proves the current repaired relaxation infeasible within
  numerical tolerances. A contradiction with a checked incumbent is flagged in
  metadata and must be investigated, never turned into a biological claim.
- `failed_validation`: numerical/distance checks failed; do not export that set.

Per-cell `constraint_margins.json` now contains actual margins. A nonnegative
margin is a pass (1e-7 numerical tolerance). An activity drop tolerance of 0.03
means **three score points on a 0–1 scale**, not a 3% relative reduction.

If a seed-42 profile improves risk while satisfying all checks, repeat the same
profile with `--seeds 43 44 --risk-ceilings <ceiling>` and a fresh output root.
This remains a predictor-based selection experiment. Do not claim measured
safety, publish the new shortlist as a release, or promote it automatically.

## Authorship audit in parallel

```bash
uv run --no-sync python scripts/audit_authorship_readiness.py \
  --out sweep_results/authorship-readiness-v1
cat sweep_results/authorship-readiness-v1/REPORT.md
```

Resolve missing committed predictor metadata and dataset provenance before the
fresh-clone release check described in `AUTHORSHIP_READINESS.md`.
Public visibility and a completed official submission
remain necessary external steps; running these scripts does not submit anything.
