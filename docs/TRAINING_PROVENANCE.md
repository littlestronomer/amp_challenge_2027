# Training provenance and unresolved historical links

The machine-readable [lineage ledger](release/TRAINING_LINEAGE.json) pins all
five deployed model files and their candidate data snapshots. Model hashes were
computed from the committed files; SSH data hashes were supplied by the author
or already pinned in the provisional reconstruction protocol. These are distinct
evidence types. No historical input manifest has been invented or backdated.

| Component | Candidate snapshot | Evidence and remaining gap |
|---|---|---|
| Primary generator | `data/processed/generative.csv` | SSH reports byte identity to `checkpoint/generator-e100_p10-seed44/model.pt` and a matching candidate CSV. Original training-time input/split linkage remains unverified. |
| Charge-conditioned generator | Same candidate CSV | SSH reports byte identity to `checkpoint/generator-e100_p10-cond/model.pt` and a matching candidate CSV. Original training-time input/split linkage remains unverified. |
| Binary activity | `data/processed/activity_labels.csv` | Prior reconstruction protocol pins the CSV and records frozen run member 2, seed 44. Latest SSH report also confirms byte identity to the original run and member copies, and a matching candidate CSV. Original training-time CSV hash/split absent. |
| Panel activity | `data/processed/activity_labels_full.csv` | SSH hash matches the previously pinned reconstruction candidate; frozen run member 1, seed 43 is recorded. Original training-time CSV hash/split absent. |
| Hemolysis | `data/processed/hemolysis_labels.csv` | SSH hash matches the previously pinned candidate; frozen run member 0, seed 42 is recorded. Original training-time CSV hash/split absent. |

The repository already records the missing historical predictor split manifests in
[the reconstruction runbook](RUNBOOK_REWARD_RECONSTRUCTION.md). Reconstructing a
split with the current trainer and recorded seed provides a retrospective
diagnostic only. Neither a similar AUROC nor a matching member proves which CSV
was used. Newer generator training code writes input/split hashes; its presence
today does not show that the historical deployed models used that version.

## SSH collection reviewed on 2026-09-25

The author supplied the completed report from the command below. Its findings are
preserved in [SSH_LINEAGE_REPORT.json](release/SSH_LINEAGE_REPORT.json). No repeat
collection or generation is needed for this documentation update. To reproduce
the collection later, choose a new output directory:

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
Keep the original files intact. The reported scan listed configuration files but
no original split or training manifests; it does not rule out records elsewhere.

If no contemporaneous manifests are recovered, retain the explicit limitation
in the release. Do not label a newly generated split as historical evidence.
Prospective retraining would create new weights and require a new evaluation;
it is not necessary merely to make a provenance checklist look complete.

## Source completeness

The current generator CSV has 39,448 rows and 39,448 unique sequences; its
sequence set equals the competition reference. All 13 upstream names occur in
its `source_dbs` annotations, according to the supplied report:

| Source annotation | Membership count |
|---|---:|
| AMPDB | 1,270 |
| APD | 2,041 |
| BaAMPs | 182 |
| CAMP | 11,898 |
| CancerPPD | 396 |
| CyBase | 169 |
| DADP | 601 |
| DBAASP | 14,496 |
| DRAMP | 18,395 |
| InverPep | 424 |
| ParaPep | 124 |
| SATPdb | 9,740 |
| dbAMP | 20,933 |

Counts can overlap because a sequence can name multiple databases. These are
CSV annotations, not independently checked database memberships. The separate
DRAMP downloads remain declared development-only, while indirect DRAMP ancestry
is explicitly disclosed. See [source acknowledgements](THIRD_PARTY_NOTICES.md).

The author declared no private training data on 2026-09-24. Source annotations of the current generator CSV are now inventoried; historical
checkpoint-input linkage and exhaustive public-source use across training remain
unverified. Download
integrity, data-source attribution, training lineage and biological validation
are separate claims.
