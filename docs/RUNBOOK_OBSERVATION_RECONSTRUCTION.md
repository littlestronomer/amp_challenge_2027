# Observation reconstruction audit

Run on the SSH machine from the repository root:

```bash
git pull --ff-only origin main
uv run --no-sync python scripts/audit_label_observations.py \
  --out sweep_results/label-observations-v1
```

CPU only. No new dependencies, model downloads, training, or changes to the
existing CSVs. Requires raw DBAASP `peptides.csv`, `activity.csv`, and
`hemolysis_raw.csv`, plus all three existing processed label CSVs. Missing files
or required columns cause an error. Use a new output directory for each run.

Outputs:

- `report.json`: input hashes, code identity, syntax/conversion/evidence counts,
  issues and comparison statuses.
- `activity_observations.csv`, `hemolysis_observations.csv`: every source row,
  target, sequence, available peptide metadata, local reference identifier,
  original raw row, parsed syntax, estimated concentration, and issue flags.
- `label_comparison.csv`: every old label compared with grouped diagnostic
  evidence, with its source line retained. Conflicts and ambiguous evidence
  remain explicit.
- `complete.json`: hashes of the completed output files.

## Deliberately conservative rules

Activity uses the original `assay`, never silently replaces it with a normalized
field. Comparisons flag normalization disagreements and lost range/uncertainty
structure. Ranges, ± errors, qualitative notes, unsupported formats and unknown
units are masked. No standard-deviation/confidence-interval meaning is guessed.

Mass conversion estimates require a standard nonempty sequence. Missing or
unrecognized modification metadata blocks those estimates from becoming label
evidence; the estimate remains visible to audit historical conversion errors.
The audit does not assert that a sequence-derived mass is experimentally known.

Activity target matching uses the existing panel mapping. Diagnostic positive
cutoffs are 16 µM for panel and 4 µM for binary, with negative evidence >32 µM.
The binary report pools observations across genera and does NOT reproduce the
historical minimum-MIC aggregation. A disagreement is not proof of a wrong label.

Hemolysis requires erythrocyte/RBC targets and explicit percent-hemolysis bands.
The entire band must satisfy the existing 40% risk / 30% low-lysis boundary,
and the concentration bound must imply <=128 / >=128 µM respectively. `active`
means risk evidence, not antimicrobial activity. Cytotoxicity, qualitative
notes, IC50/MHC and unspecified endpoints are not converted into safety labels.
Donor, species and conditions remain in raw rows; they are not interchangeable
replicates. Reference numbers remain peptide-local, not global study identifiers.

This is a diagnostic report, not replacement training data. Share the console
summary before deciding the label aggregation policy. Preserve the existing
generalization test as already inspected; this audit creates no new holdout.
