# Frozen pool diagnosis

This is the first stage of the next iteration, not a new training experiment.
It reads the completed external pilot and its original cached internal controls.
It verifies completion inventories, source identities, scored-pool hashes and
the original reference hash. No models are loaded, no predictors are run, and
no original files are modified. The new run manifest freezes this comparison.

Run on SSH after pulling main:

```bash
uv run --no-sync python -u scripts/audit_generator_pools.py \
  --source sweep_results/external-pilot-v2 \
  --out sweep_results/pool-accounting-v1
```

No GPU is needed. Pairwise grouping of qualifying sequences can take time.
The audit uses each prefix from the original recipe and never adds draws.
Use a new output directory after any code/input change. Completed identical
audits are hash-verified and reused. Existing source results need not be rerun
just because the current code commit changed.

## Outputs

- `run.json`: input hashes, origins, thresholds, code identity and limitations.
- `pool_accounting.csv`: mutually exclusive draw accounting and qualifying yield.
- `length_strata.csv`: the same categories within length bins; invalid sequences
  outside 8..50 are present in overall accounting, not these bins.
- `qualifying_families.csv`: deterministic greedy representative groups at
  Levenshtein ratio >=0.8, among unique non-reference qualifying sequences.
  This file is omitted if there are no qualifying sequences in any cell.
- `complete.json`: output integrity inventory.

Reference membership takes precedence over repetition: repeated reference
sequences count as exact-reference draws, not repeated-nonreference draws.
`reference_unique` is an additional diagnostic and must not be added to the
four mutually exclusive categories. Invalid draws remain in yield denominators.

Qualifying thresholds are activity >=0.6 and both risk scores <=0.5, matching
the current development diagnostics. Evaluation risk participates in this
diagnostic definition, so these results are NOT independent confirmation.
Grouping uses lexicographic sequence order and first matching representative;
it is not connected-component clustering or biological family identification.
Its group counts can differ from prior score-ordered separation diagnostics.

## Print the primary result

```bash
uv run --no-sync python - <<'PY'
import pandas as pd
root = "sweep_results/pool-accounting-v1"
df = pd.read_csv(f"{root}/pool_accounting.csv")
df = df[df.draws == df.draws.max()]
print(df.to_string(index=False))
cols = ["invalid", "exact_reference", "repeated_nonreference", "unique_nonreference",
        "qualifying_yield_per_1000", "family_yield_per_1000"]
print(df.groupby("method")[cols].agg(["mean", "std"]).to_string())
PY
```

Decide the next intervention from the diagnosis: reference reproduction,
novel-sequence repetition, concentration within similarity groups, or length
concentration. Near-reference similarity, reward-training-data overlap,
independent evaluator validation, selector changes, and new post-training
remain separate future stages. Do not interpret this audit as implementing
or validating those stages.
