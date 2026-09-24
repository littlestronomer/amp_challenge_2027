# Training provenance and unresolved historical links

The machine-readable [lineage ledger](release/TRAINING_LINEAGE.json) pins all
five deployed model files and their candidate data snapshots. Model hashes were
computed from the committed files; SSH data hashes were supplied by the author
or already pinned in the provisional reconstruction protocol. These are distinct
evidence types. No historical input manifest has been invented or backdated.

| Component | Candidate snapshot | Evidence and remaining gap |
|---|---|---|
| Primary generator | `data/processed/generative.csv` | SSH snapshot hash recorded. Historical notes describe seed 44 and the promoted generator; original split and promotion-to-input linkage remain unverified. |
| Charge-conditioned generator | Same candidate CSV | Architecture config confirms charge conditioning, not training data. Original split, seed and promotion linkage remain unverified. |
| Binary activity | `data/processed/activity_labels.csv` | Prior reconstruction protocol pins the CSV and records frozen run member 2, seed 44. Prior audit reports tensor identity. Original training-time CSV hash/split absent. |
| Panel activity | `data/processed/activity_labels_full.csv` | SSH hash matches the previously pinned reconstruction candidate; frozen run member 1, seed 43 is recorded. Original training-time CSV hash/split absent. |
| Hemolysis | `data/processed/hemolysis_labels.csv` | SSH hash matches the previously pinned candidate; frozen run member 0, seed 42 is recorded. Original training-time CSV hash/split absent. |

The repository already records the missing historical predictor split manifests in
[the reconstruction runbook](RUNBOOK_REWARD_RECONSTRUCTION.md). Reconstructing a
split with the current trainer and recorded seed provides a retrospective
diagnostic only. Neither a similar AUROC nor a matching member proves which CSV
was used. Newer generator training code writes input/split hashes; its presence
today does not show that the historical deployed models used that version.

## Collect the remaining SSH evidence

From the repository root, run this once in a new output directory:

```bash
uv run --no-sync python scripts/collect_training_lineage.py \
  --out sweep_results/training-lineage-v1
cat sweep_results/training-lineage-v1/REPORT.md
```

This uses CPU and the standard library. It compares all five deployed weights
and candidate CSVs with the ledger, reports CSV row counts/source annotations,
inventories raw/processed files and original JSON manifests, and looks for
byte-identical model copies under `checkpoint/`. It also compares inference
files and the existing fresh-clone FASTAs against the validated baseline.
CSV sequence sets are compared to the reference without exporting peptide rows;
set equality is a current-corpus check, not proof of historical training use.
It does not download, regenerate, retrain, unpickle models or upload files.
Keep the original files intact and return the report for review.

If no contemporaneous manifests are recovered, retain the explicit limitation
in the release. Do not label a newly generated split as historical evidence.
Prospective retraining would create new weights and require a new evaluation;
it is not necessary merely to make a provenance checklist look complete.

## Source completeness

The generator CSV has a `source_dbs` column in the documented writer. Its actual
values on SSH will help reconcile the competition reference/MarLys subset.
MarLys itself aggregates 13 databases. Thus “separate DRAMP downloads were used
only in development” does not exclude indirect DRAMP ancestry in generators.
See [source acknowledgements and terms](THIRD_PARTY_NOTICES.md).

The author declared no private training data on 2026-09-24. The exact public-source
inventory and historical checkpoint-input linkage remain unresolved. Download
integrity, data-source attribution, training lineage and biological validation
are separate claims.
