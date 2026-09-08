# Conversion and metadata reconciliation

Run on SSH from the repository root:

```bash
git pull --ff-only origin main
uv run --no-sync python scripts/reconcile_observation_metadata.py \
  --source sweep_results/label-observations-v1 \
  --out sweep_results/metadata-reconciliation-v1
```

CPU only; no downloads, model runs, or dependency changes. Requires the completed
observation audit and the same `data/raw/dbaasp/peptides.csv` snapshot. All source
completion hashes are checked. Existing output directories are rejected.

Share the console summary. `report.json` records hashes and limitations;
`activity_reconciliation.csv` and `hemolysis_reconciliation.csv` retain source
line references for joining back to the original audit's full provenance.

The `prior_disagreements` histogram partitions the previously reported conversion
disagreements. `relative_difference`, `implied_mw_da`, `sequence_mw_da` and
`expected_um` support investigating conversion discrepancies. Molar conversions
use explicit unit factors; sequence-mass calculations remain estimates, not
verified molecular weights. Operator changes are reported independently.

Exact matches to `dbaasp_id`/`dbaaspId` are reported as alias candidates. Prefixes
are not stripped and ambiguous aliases are not selected. No sequence replacement
occurs. Metadata coverage reports unknown fields separately from recorded values;
even recorded values remain uninterpreted. No modification or terminal chemistry
is inferred from missing metadata. No external metadata is fetched.

This diagnostic step does not resolve donor/strain/condition conflicts, produce
new training labels, or establish generalizability. The next label-policy decision
must use the actual reconciliation results. Preserve existing training snapshots
and the already-inspected test results.
